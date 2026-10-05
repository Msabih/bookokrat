//! Microphone recording through an external command-line recorder.
//!
//! The recorder is detected at runtime (PipeWire first, then ALSA, SoX and
//! FFmpeg) and writes 16 kHz mono 16-bit WAV, which the transcription API
//! accepts directly and which keeps a minute of speech under 2 MB.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

pub const SAMPLE_RATE: u32 = 16_000;
const WAV_HEADER_LEN: u64 = 44;
/// 100 ms of 16-bit mono audio.
const METER_WINDOW_BYTES: u64 = (SAMPLE_RATE as u64) / 10 * 2;
const METER_INTERVAL: Duration = Duration::from_millis(100);
const GRACEFUL_STOP_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecorderTool {
    PwRecord,
    Arecord,
    Rec,
    Sox,
    Ffmpeg,
}

impl RecorderTool {
    /// Auto-detection order: PipeWire's native client first (talks to the
    /// session's default source), then ALSA, then SoX, then FFmpeg (which
    /// needs a platform-specific input device).
    pub const DETECTION_ORDER: [RecorderTool; 5] = [
        RecorderTool::PwRecord,
        RecorderTool::Arecord,
        RecorderTool::Rec,
        RecorderTool::Sox,
        RecorderTool::Ffmpeg,
    ];

    pub fn program(self) -> &'static str {
        match self {
            RecorderTool::PwRecord => "pw-record",
            RecorderTool::Arecord => "arecord",
            RecorderTool::Rec => "rec",
            RecorderTool::Sox => "sox",
            RecorderTool::Ffmpeg => "ffmpeg",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "pw-record" | "pipewire" => Some(RecorderTool::PwRecord),
            "arecord" | "alsa" => Some(RecorderTool::Arecord),
            "rec" => Some(RecorderTool::Rec),
            "sox" => Some(RecorderTool::Sox),
            "ffmpeg" => Some(RecorderTool::Ffmpeg),
            _ => None,
        }
    }

    pub fn args(self, output: &Path) -> Vec<String> {
        let out = output.to_string_lossy().into_owned();
        let rate = SAMPLE_RATE.to_string();
        let v = |items: &[&str]| items.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        match self {
            RecorderTool::PwRecord => {
                let mut a = v(&["--rate", &rate, "--channels", "1", "--format", "s16"]);
                a.push(out);
                a
            }
            RecorderTool::Arecord => {
                let mut a = v(&["-q", "-f", "S16_LE", "-r", &rate, "-c", "1", "-t", "wav"]);
                a.push(out);
                a
            }
            RecorderTool::Rec => {
                let mut a = v(&[
                    "-q",
                    "-r",
                    &rate,
                    "-c",
                    "1",
                    "-b",
                    "16",
                    "-e",
                    "signed-integer",
                ]);
                a.push(out);
                a
            }
            RecorderTool::Sox => {
                let mut a = v(&[
                    "-q",
                    "-d",
                    "-r",
                    &rate,
                    "-c",
                    "1",
                    "-b",
                    "16",
                    "-e",
                    "signed-integer",
                ]);
                a.push(out);
                a
            }
            RecorderTool::Ffmpeg => {
                let mut a = v(&["-hide_banner", "-loglevel", "error", "-nostdin", "-y"]);
                a.extend(v(ffmpeg_input_args()));
                a.extend(v(&["-ac", "1", "-ar", &rate, "-c:a", "pcm_s16le"]));
                a.push(out);
                a
            }
        }
    }
}

#[cfg(target_os = "macos")]
fn ffmpeg_input_args() -> &'static [&'static str] {
    &["-f", "avfoundation", "-i", ":0"]
}

#[cfg(target_os = "windows")]
fn ffmpeg_input_args() -> &'static [&'static str] {
    &["-f", "dshow", "-i", "audio=default"]
}

#[cfg(not(any(target_os = "macos", target_os = "windows")))]
fn ffmpeg_input_args() -> &'static [&'static str] {
    &["-f", "pulse", "-i", "default"]
}

/// How a recording will be made.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecorderPlan {
    Tool {
        tool: RecorderTool,
        program: PathBuf,
    },
    /// User-supplied shell command; `{file}` is replaced by the output path
    /// (appended when absent). The command must write WAV/MP3/… and stop
    /// cleanly on SIGINT.
    Custom { command: String },
}

impl RecorderPlan {
    pub fn label(&self) -> String {
        match self {
            RecorderPlan::Tool { tool, .. } => tool.program().to_string(),
            RecorderPlan::Custom { command } => {
                let first = command.split_whitespace().next().unwrap_or("custom");
                format!("custom ({first})")
            }
        }
    }

    pub fn command(&self, output: &Path) -> (String, Vec<String>) {
        match self {
            RecorderPlan::Tool { tool, program } => {
                (program.to_string_lossy().into_owned(), tool.args(output))
            }
            RecorderPlan::Custom { command } => {
                let quoted = shell_quote(&output.to_string_lossy());
                let line = if command.contains("{file}") {
                    command.replace("{file}", &quoted)
                } else {
                    format!("{command} {quoted}")
                };
                if cfg!(windows) {
                    ("cmd".to_string(), vec!["/C".to_string(), line])
                } else {
                    ("sh".to_string(), vec!["-c".to_string(), line])
                }
            }
        }
    }

    /// Built-in tools write 16-bit PCM WAV, which the level meter can read.
    fn writes_pcm_wav(&self) -> bool {
        matches!(self, RecorderPlan::Tool { .. })
    }
}

fn shell_quote(s: &str) -> String {
    if cfg!(windows) {
        format!("\"{s}\"")
    } else {
        format!("'{}'", s.replace('\'', "'\\''"))
    }
}

/// Choose a recorder. `preference` is the settings override: empty/`auto`
/// means auto-detect, a known tool name forces that tool, anything else is a
/// custom shell command.
pub fn plan_recorder(
    preference: Option<&str>,
    find: &dyn Fn(&str) -> Option<PathBuf>,
) -> Result<RecorderPlan, String> {
    let preference = preference.map(str::trim).filter(|p| !p.is_empty());
    match preference {
        None => {}
        Some(p) if p.eq_ignore_ascii_case("auto") => {}
        Some(p) => {
            if let Some(tool) = RecorderTool::from_name(p) {
                return find(tool.program())
                    .map(|program| RecorderPlan::Tool { tool, program })
                    .ok_or_else(|| format!("Recorder '{}' not found in PATH", tool.program()));
            }
            return Ok(RecorderPlan::Custom {
                command: p.to_string(),
            });
        }
    }
    for tool in RecorderTool::DETECTION_ORDER {
        if let Some(program) = find(tool.program()) {
            return Ok(RecorderPlan::Tool { tool, program });
        }
    }
    Err("No audio recorder found: install pw-record (PipeWire), arecord (ALSA), sox or ffmpeg, or set a recorder command in Settings".to_string())
}

/// Locate an executable in `PATH`.
pub fn find_in_path(program: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let candidate = dir.join(program);
        if is_executable(&candidate) {
            return Some(candidate);
        }
        #[cfg(windows)]
        {
            let exe = dir.join(format!("{program}.exe"));
            if is_executable(&exe) {
                return Some(exe);
            }
        }
    }
    None
}

#[cfg(unix)]
fn is_executable(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    std::fs::metadata(path)
        .map(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
        .unwrap_or(false)
}

#[cfg(not(unix))]
fn is_executable(path: &Path) -> bool {
    path.is_file()
}

/// Map a window of 16-bit little-endian PCM to a 0.0–1.0 loudness level
/// (RMS on a -55 dBFS..0 dBFS scale).
pub fn pcm_level(bytes: &[u8]) -> f32 {
    let samples = bytes.len() / 2;
    if samples == 0 {
        return 0.0;
    }
    let sum: f64 = bytes
        .chunks_exact(2)
        .map(|c| {
            let s = i16::from_le_bytes([c[0], c[1]]) as f64;
            s * s
        })
        .sum();
    let rms = (sum / samples as f64).sqrt();
    if rms < 1.0 {
        return 0.0;
    }
    let db = 20.0 * (rms / 32768.0).log10();
    (((db + 55.0) / 55.0) as f32).clamp(0.0, 1.0)
}

/// Starts recordings. Behind a trait so tests never touch a microphone.
pub trait AudioRecorder: Send + Sync {
    fn plan(&self, preference: Option<&str>) -> Result<RecorderPlan, String>;
    fn start(
        &self,
        plan: &RecorderPlan,
        output: &Path,
        log: &Path,
    ) -> Result<Box<dyn ActiveRecording>, String>;
}

/// A recording in progress.
pub trait ActiveRecording: Send {
    /// Current input level (0.0–1.0), if the recorder's output can be metered.
    fn level(&self) -> Option<f32>;
    /// `Err` with a user-facing reason if the recorder died on its own.
    fn check_alive(&mut self) -> Result<(), String>;
    /// Stop gracefully so the file is finalized. Blocks until the recorder
    /// exits; call it off the UI thread.
    fn finish(self: Box<Self>) -> Result<(), String>;
    /// Stop immediately, discarding the recording. May block briefly.
    fn abort(self: Box<Self>);
}

/// Real recorder: spawns the detected command-line tool.
pub struct SystemRecorder;

impl AudioRecorder for SystemRecorder {
    fn plan(&self, preference: Option<&str>) -> Result<RecorderPlan, String> {
        plan_recorder(preference, &find_in_path)
    }

    fn start(
        &self,
        plan: &RecorderPlan,
        output: &Path,
        log: &Path,
    ) -> Result<Box<dyn ActiveRecording>, String> {
        let (program, args) = plan.command(output);
        let stderr = File::create(log)
            .map(Stdio::from)
            .unwrap_or_else(|_| Stdio::null());
        let mut cmd = Command::new(&program);
        cmd.args(&args)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(stderr);
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            // Own process group: SIGINT reaches the whole pipeline of a
            // custom `sh -c` command, and terminal signals never reach it.
            cmd.process_group(0);
        }
        let child = cmd.spawn().map_err(|e| {
            log::error!("Failed to start recorder {program}: {e}");
            format!("Failed to start {}: {e}", plan.label())
        })?;
        log::info!("Recording with {} (pid {})", plan.label(), child.id());

        let stop = Arc::new(AtomicBool::new(false));
        let (level, meter) = if plan.writes_pcm_wav() {
            let level = Arc::new(AtomicU32::new(0f32.to_bits()));
            let handle = spawn_meter(output.to_path_buf(), level.clone(), stop.clone());
            (Some(level), handle)
        } else {
            (None, None)
        };

        Ok(Box::new(ProcessRecording {
            child,
            label: plan.label(),
            log: log.to_path_buf(),
            level,
            stop,
            meter,
        }))
    }
}

fn spawn_meter(
    path: PathBuf,
    level: Arc<AtomicU32>,
    stop: Arc<AtomicBool>,
) -> Option<JoinHandle<()>> {
    std::thread::Builder::new()
        .name("voice-meter".into())
        .spawn(move || {
            let mut buf = vec![0u8; METER_WINDOW_BYTES as usize];
            while !stop.load(Ordering::Relaxed) {
                std::thread::sleep(METER_INTERVAL);
                if let Some(value) = read_tail_level(&path, &mut buf) {
                    level.store(value.to_bits(), Ordering::Relaxed);
                }
            }
        })
        .map_err(|e| log::warn!("Failed to start level meter thread: {e}"))
        .ok()
}

fn read_tail_level(path: &Path, buf: &mut [u8]) -> Option<f32> {
    let mut file = File::open(path).ok()?;
    let len = file.metadata().ok()?.len();
    if len <= WAV_HEADER_LEN {
        return None;
    }
    let window = (len - WAV_HEADER_LEN).min(buf.len() as u64) & !1;
    if window == 0 {
        return None;
    }
    // Align to sample boundaries (the WAV header length is even).
    let start = (len - window) & !1;
    file.seek(SeekFrom::Start(start)).ok()?;
    let n = file.read(&mut buf[..window as usize]).ok()?;
    Some(pcm_level(&buf[..n & !1]))
}

struct ProcessRecording {
    child: Child,
    label: String,
    log: PathBuf,
    level: Option<Arc<AtomicU32>>,
    stop: Arc<AtomicBool>,
    meter: Option<JoinHandle<()>>,
}

impl ProcessRecording {
    fn signal(&self, signal: i32) {
        #[cfg(unix)]
        {
            // Negative pid: the whole process group created at spawn.
            let pgid = self.child.id() as i32;
            unsafe {
                libc::kill(-pgid, signal);
            }
        }
        #[cfg(not(unix))]
        {
            let _ = signal;
        }
    }

    fn wait_with_timeout(&mut self, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        loop {
            match self.child.try_wait() {
                Ok(Some(_)) => return true,
                Ok(None) if Instant::now() < deadline => {
                    std::thread::sleep(Duration::from_millis(20));
                }
                Ok(None) => return false,
                Err(e) => {
                    log::warn!("Failed to wait for recorder: {e}");
                    return false;
                }
            }
        }
    }

    fn kill_now(&mut self) {
        #[cfg(unix)]
        self.signal(libc::SIGKILL);
        let _ = self.child.kill();
        let _ = self.child.wait();
    }

    fn stop_meter(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.meter.take() {
            let _ = handle.join();
        }
    }

    fn log_tail(&self) -> Option<String> {
        let text = std::fs::read_to_string(&self.log).ok()?;
        let line = text.lines().rev().find(|l| !l.trim().is_empty())?.trim();
        Some(line.chars().take(160).collect())
    }
}

impl ActiveRecording for ProcessRecording {
    fn level(&self) -> Option<f32> {
        self.level
            .as_ref()
            .map(|l| f32::from_bits(l.load(Ordering::Relaxed)))
    }

    fn check_alive(&mut self) -> Result<(), String> {
        match self.child.try_wait() {
            Ok(None) => Ok(()),
            Ok(Some(status)) => {
                let detail = self.log_tail().unwrap_or_else(|| status.to_string());
                log::error!("Recorder {} exited early ({status}): {detail}", self.label);
                Err(format!("{} stopped: {detail}", self.label))
            }
            Err(e) => {
                log::error!("Failed to query recorder {}: {e}", self.label);
                Err(format!("{} failed: {e}", self.label))
            }
        }
    }

    fn finish(mut self: Box<Self>) -> Result<(), String> {
        #[cfg(unix)]
        self.signal(libc::SIGINT);
        #[cfg(not(unix))]
        let _ = self.child.kill();
        if !self.wait_with_timeout(GRACEFUL_STOP_TIMEOUT) {
            log::warn!("Recorder {} ignored SIGINT; killing it", self.label);
            self.kill_now();
        }
        self.stop_meter();
        Ok(())
    }

    fn abort(mut self: Box<Self>) {
        self.kill_now();
        self.stop_meter();
    }
}

impl Drop for ProcessRecording {
    fn drop(&mut self) {
        if matches!(self.child.try_wait(), Ok(None)) {
            self.kill_now();
        }
        self.stop.store(true, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn finder(available: &'static [&'static str]) -> impl Fn(&str) -> Option<PathBuf> {
        move |name| {
            available
                .contains(&name)
                .then(|| PathBuf::from(format!("/usr/bin/{name}")))
        }
    }

    #[test]
    fn auto_detection_prefers_pipewire() {
        let plan = plan_recorder(None, &finder(&["sox", "arecord", "pw-record"])).unwrap();
        assert_eq!(plan.label(), "pw-record");
    }

    #[test]
    fn auto_detection_falls_back_in_order() {
        let find = finder(&["ffmpeg", "sox", "arecord"]);
        assert_eq!(plan_recorder(None, &find).unwrap().label(), "arecord");
        let find = finder(&["ffmpeg", "sox"]);
        assert_eq!(plan_recorder(Some("auto"), &find).unwrap().label(), "sox");
        let find = finder(&["ffmpeg", "rec", "sox"]);
        assert_eq!(plan_recorder(Some("  "), &find).unwrap().label(), "rec");
        let find = finder(&["ffmpeg"]);
        assert_eq!(plan_recorder(None, &find).unwrap().label(), "ffmpeg");
    }

    #[test]
    fn no_recorder_is_an_error() {
        assert!(plan_recorder(None, &finder(&[])).is_err());
    }

    #[test]
    fn named_override_forces_tool() {
        let find = finder(&["pw-record", "ffmpeg"]);
        assert_eq!(
            plan_recorder(Some("ffmpeg"), &find).unwrap().label(),
            "ffmpeg"
        );
        assert!(plan_recorder(Some("arecord"), &find).is_err());
    }

    #[test]
    fn custom_command_substitutes_file() {
        let plan = plan_recorder(Some("parecord --file-format=wav {file}"), &finder(&[])).unwrap();
        let (program, args) = plan.command(Path::new("/tmp/a b/x.wav"));
        if cfg!(unix) {
            assert_eq!(program, "sh");
            assert_eq!(args[1], "parecord --file-format=wav '/tmp/a b/x.wav'");
        }
        let plan = RecorderPlan::Custom {
            command: "myrec".into(),
        };
        let (_, args) = plan.command(Path::new("/tmp/x.wav"));
        assert!(
            args.last().unwrap().ends_with("x.wav'") || args.last().unwrap().ends_with("x.wav\"")
        );
    }

    #[test]
    fn tool_args_end_with_output_path() {
        for tool in RecorderTool::DETECTION_ORDER {
            let args = tool.args(Path::new("/tmp/out.wav"));
            assert_eq!(args.last().map(String::as_str), Some("/tmp/out.wav"));
            assert!(args.iter().any(|a| a == "16000"), "{tool:?} sets 16 kHz");
        }
    }

    #[test]
    fn pcm_level_scales_with_loudness() {
        assert_eq!(pcm_level(&[]), 0.0);
        assert_eq!(pcm_level(&[0; 64]), 0.0);
        let quiet: Vec<u8> = std::iter::repeat_n(100i16.to_le_bytes(), 32)
            .flatten()
            .collect();
        let loud: Vec<u8> = std::iter::repeat_n(20000i16.to_le_bytes(), 32)
            .flatten()
            .collect();
        let q = pcm_level(&quiet);
        let l = pcm_level(&loud);
        assert!(q > 0.0 && q < l && l <= 1.0, "quiet={q} loud={l}");
    }

    #[test]
    fn tail_level_reads_wav_data() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.wav");
        let mut data = vec![0u8; WAV_HEADER_LEN as usize];
        data.extend(std::iter::repeat_n(16000i16.to_le_bytes(), 4000).flatten());
        std::fs::write(&path, &data).unwrap();
        let mut buf = vec![0u8; METER_WINDOW_BYTES as usize];
        let level = read_tail_level(&path, &mut buf).unwrap();
        assert!(level > 0.8, "level={level}");
        std::fs::write(&path, vec![0u8; 20]).unwrap();
        assert!(read_tail_level(&path, &mut buf).is_none());
    }
}
