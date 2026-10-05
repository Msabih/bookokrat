//! Voice notes: record from the microphone, transcribe through an
//! OpenAI-compatible endpoint, hand the text to the comment input.
//!
//! `Dictation` is polled from the UI loop; the recorder runs as its own
//! process and stopping / transcribing happen on a worker thread, so the UI
//! never blocks.

pub mod recorder;
pub mod transcriber;

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

use recorder::{ActiveRecording, AudioRecorder};
use transcriber::{Transcriber, TranscriptionRequest};

pub const OPENAI_ENDPOINT: &str = "https://api.openai.com/v1";
pub const OPENAI_DEFAULT_MODEL: &str = "gpt-transcribe";
/// Recordings longer than this are stopped automatically (the OpenAI
/// endpoint takes files up to 25 MB; 10 minutes of 16 kHz WAV is ~19 MB).
const MAX_RECORDING: Duration = Duration::from_secs(600);

/// Transcription settings resolved from config + environment.
#[derive(Clone, PartialEq)]
pub struct VoiceConfig {
    pub endpoint: String,
    pub model: String,
    pub language: Option<String>,
    pub prompt: Option<String>,
    pub api_key: Option<String>,
    pub recorder: Option<String>,
}

impl std::fmt::Debug for VoiceConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VoiceConfig")
            .field("endpoint", &self.endpoint)
            .field("model", &self.model)
            .field("language", &self.language)
            .field("api_key_set", &self.api_key.is_some())
            .field("recorder", &self.recorder)
            .finish_non_exhaustive()
    }
}

fn non_empty(value: &Option<String>) -> Option<String> {
    value
        .as_deref()
        .map(str::trim)
        .filter(|v| !v.is_empty() && !v.eq_ignore_ascii_case("auto"))
        .map(str::to_string)
}

impl VoiceConfig {
    /// `env` looks up environment variables (injected for tests).
    pub fn resolve(
        settings: &crate::settings::Settings,
        env: &dyn Fn(&str) -> Option<String>,
    ) -> Self {
        let endpoint = non_empty(&settings.transcribe_endpoint)
            .unwrap_or_else(|| OPENAI_ENDPOINT.to_string())
            .trim_end_matches('/')
            .to_string();
        let is_openai = endpoint.contains("api.openai.com");
        let model = non_empty(&settings.transcribe_model).unwrap_or_else(|| {
            if is_openai {
                OPENAI_DEFAULT_MODEL.to_string()
            } else if endpoint.contains("groq.com") {
                "whisper-large-v3-turbo".to_string()
            } else {
                "whisper-1".to_string()
            }
        });
        let key_var = if endpoint.contains("groq.com") {
            "GROQ_API_KEY"
        } else {
            "OPENAI_API_KEY"
        };
        let api_key = non_empty(&settings.transcribe_api_key)
            .or_else(|| env(key_var).filter(|k| !k.trim().is_empty()));
        Self {
            endpoint,
            model,
            language: non_empty(&settings.transcribe_language),
            prompt: non_empty(&settings.transcribe_prompt),
            api_key,
            recorder: non_empty(&settings.voice_recorder),
        }
    }

    pub fn transcriptions_url(&self) -> String {
        format!("{}/audio/transcriptions", self.endpoint)
    }

    pub fn models_url(&self) -> String {
        format!("{}/models", self.endpoint)
    }

    /// Hosted APIs need a key; a local server usually does not.
    pub fn needs_key(&self) -> bool {
        !(self.endpoint.contains("127.0.0.1") || self.endpoint.contains("localhost"))
    }
}

/// What the UI should show / do after a poll.
#[derive(Debug, Clone, PartialEq)]
pub enum DictationEvent {
    /// Recognized text to insert at the cursor.
    Text(String),
    Error(String),
    Cancelled,
}

enum Phase {
    Idle,
    Recording {
        recording: Box<dyn ActiveRecording>,
        started: Instant,
        audio: PathBuf,
        workdir: tempfile::TempDir,
        config: VoiceConfig,
    },
    Transcribing {
        started: Instant,
        rx: std::sync::mpsc::Receiver<Result<String, String>>,
        cancel: Arc<AtomicBool>,
    },
}

/// Status line content while dictation is active.
#[derive(Debug, Clone, PartialEq)]
pub enum DictationStatus {
    Recording {
        elapsed: Duration,
        level: Option<f32>,
    },
    Transcribing {
        elapsed: Duration,
        model: String,
    },
}

pub struct Dictation {
    recorder: Arc<dyn AudioRecorder>,
    transcriber: Arc<dyn Transcriber>,
    phase: Phase,
    model: String,
}

impl Dictation {
    pub fn new(recorder: Arc<dyn AudioRecorder>, transcriber: Arc<dyn Transcriber>) -> Self {
        Self {
            recorder,
            transcriber,
            phase: Phase::Idle,
            model: String::new(),
        }
    }

    pub fn system() -> Self {
        Self::new(
            Arc::new(recorder::SystemRecorder),
            Arc::new(transcriber::CurlTranscriber::default()),
        )
    }

    pub fn is_active(&self) -> bool {
        !matches!(self.phase, Phase::Idle)
    }

    pub fn is_recording(&self) -> bool {
        matches!(self.phase, Phase::Recording { .. })
    }

    pub fn status(&self) -> Option<DictationStatus> {
        match &self.phase {
            Phase::Idle => None,
            Phase::Recording {
                recording, started, ..
            } => Some(DictationStatus::Recording {
                elapsed: started.elapsed(),
                level: recording.level(),
            }),
            Phase::Transcribing { started, .. } => Some(DictationStatus::Transcribing {
                elapsed: started.elapsed(),
                model: self.model.clone(),
            }),
        }
    }

    /// Start recording. Fails fast (before touching the microphone) when the
    /// endpoint needs a key and none is configured.
    pub fn start(&mut self, config: VoiceConfig) -> Result<(), String> {
        if self.is_active() {
            return Ok(());
        }
        if config.needs_key() && config.api_key.is_none() {
            return Err(
                "No transcription API key: set it in Settings (Space+s) or OPENAI_API_KEY"
                    .to_string(),
            );
        }
        let plan = self.recorder.plan(config.recorder.as_deref())?;
        let workdir = tempfile::Builder::new()
            .prefix("bookokrat-voice-")
            .tempdir()
            .map_err(|e| {
                log::error!("Failed to create voice temp dir: {e}");
                format!("Failed to create a temp dir: {e}")
            })?;
        let audio = workdir.path().join("note.wav");
        let recording = self
            .recorder
            .start(&plan, &audio, &workdir.path().join("recorder.log"))?;
        log::info!("Dictation started ({config:?}, recorder {})", plan.label());
        self.model = config.model.clone();
        self.phase = Phase::Recording {
            recording,
            started: Instant::now(),
            audio,
            workdir,
            config,
        };
        Ok(())
    }

    /// Stop recording and transcribe in the background.
    pub fn stop(&mut self) {
        let Phase::Recording {
            recording,
            audio,
            workdir,
            config,
            ..
        } = std::mem::replace(&mut self.phase, Phase::Idle)
        else {
            return;
        };
        let (tx, rx) = std::sync::mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let worker_cancel = cancel.clone();
        let transcriber = self.transcriber.clone();
        let spawned = std::thread::Builder::new()
            .name("voice-transcribe".into())
            .spawn(move || {
                let result = recording.finish().and_then(|()| {
                    let request = TranscriptionRequest {
                        audio,
                        api_key: config.api_key.clone().unwrap_or_default(),
                        model: config.model.clone(),
                        language: config.language.clone(),
                        prompt: config.prompt.clone(),
                    };
                    transcriber.transcribe_at(
                        &config.transcriptions_url(),
                        &request,
                        workdir.path(),
                        &worker_cancel,
                    )
                });
                drop(workdir);
                let _ = tx.send(result);
            });
        match spawned {
            Ok(_) => {
                self.phase = Phase::Transcribing {
                    started: Instant::now(),
                    rx,
                    cancel,
                }
            }
            Err(e) => log::error!("Failed to start transcription thread: {e}"),
        }
    }

    /// Discard the recording / abandon the transcription.
    pub fn cancel(&mut self) {
        match std::mem::replace(&mut self.phase, Phase::Idle) {
            Phase::Recording { recording, .. } => recording.abort(),
            Phase::Transcribing { cancel, .. } => {
                cancel.store(true, std::sync::atomic::Ordering::Relaxed)
            }
            Phase::Idle => {}
        }
    }

    /// Call from the UI loop. Returns an event when something finished.
    pub fn poll(&mut self) -> Option<DictationEvent> {
        match &mut self.phase {
            Phase::Idle => None,
            Phase::Recording {
                recording, started, ..
            } => {
                if let Err(e) = recording.check_alive() {
                    self.phase = Phase::Idle;
                    return Some(DictationEvent::Error(e));
                }
                if started.elapsed() >= MAX_RECORDING {
                    self.stop();
                }
                None
            }
            Phase::Transcribing { rx, .. } => match rx.try_recv() {
                Ok(result) => {
                    self.phase = Phase::Idle;
                    Some(match result {
                        Ok(text) => DictationEvent::Text(text),
                        Err(e) if e == "Cancelled" => DictationEvent::Cancelled,
                        Err(e) => DictationEvent::Error(e),
                    })
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => None,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    self.phase = Phase::Idle;
                    Some(DictationEvent::Error(
                        "Transcription worker stopped unexpectedly".to_string(),
                    ))
                }
            },
        }
    }

    /// One-line status for the HUD, e.g. `● REC 0:07 ▮▮▮▯▯ — Ctrl+t: transcribe, Esc: cancel`.
    pub fn status_line(&self, toggle_key: &str) -> Option<String> {
        Some(match self.status()? {
            DictationStatus::Recording { elapsed, level } => {
                let secs = elapsed.as_secs();
                let meter = level
                    .map(|l| {
                        let n = (l * 8.0).round() as usize;
                        format!(" {}{}", "▮".repeat(n), "▯".repeat(8 - n.min(8)))
                    })
                    .unwrap_or_default();
                format!(
                    "● REC {}:{:02}{meter}   {toggle_key}: transcribe   Esc: cancel",
                    secs / 60,
                    secs % 60
                )
            }
            DictationStatus::Transcribing { elapsed, model } => {
                format!(
                    "Transcribing with {model}… {}s   Esc: cancel",
                    elapsed.as_secs()
                )
            }
        })
    }
}

impl Drop for Dictation {
    fn drop(&mut self) {
        self.cancel();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::settings::Settings;
    use std::path::Path;
    use std::sync::Mutex;

    fn env_none(_: &str) -> Option<String> {
        None
    }

    #[test]
    fn resolve_defaults_to_openai() {
        let cfg = VoiceConfig::resolve(&Settings::default(), &env_none);
        assert_eq!(cfg.endpoint, OPENAI_ENDPOINT);
        assert_eq!(cfg.model, OPENAI_DEFAULT_MODEL);
        assert_eq!(cfg.language, None);
        assert!(cfg.needs_key());
        assert_eq!(
            cfg.transcriptions_url(),
            "https://api.openai.com/v1/audio/transcriptions"
        );
    }

    #[test]
    fn resolve_uses_settings_then_env_for_key() {
        let env = |k: &str| (k == "OPENAI_API_KEY").then(|| "sk-env".to_string());
        let mut s = Settings::default();
        assert_eq!(
            VoiceConfig::resolve(&s, &env).api_key.as_deref(),
            Some("sk-env")
        );
        s.transcribe_api_key = Some("sk-config".into());
        s.transcribe_language = Some(" ur ".into());
        s.transcribe_model = Some("whisper-1".into());
        let cfg = VoiceConfig::resolve(&s, &env);
        assert_eq!(cfg.api_key.as_deref(), Some("sk-config"));
        assert_eq!(cfg.language.as_deref(), Some("ur"));
        assert_eq!(cfg.model, "whisper-1");
        assert!(!format!("{cfg:?}").contains("sk-config"));
    }

    #[test]
    fn groq_and_local_endpoints() {
        let mut s = Settings::default();
        s.transcribe_endpoint = Some("https://api.groq.com/openai/v1/".into());
        let env = |k: &str| (k == "GROQ_API_KEY").then(|| "gsk".to_string());
        let cfg = VoiceConfig::resolve(&s, &env);
        assert_eq!(cfg.model, "whisper-large-v3-turbo");
        assert_eq!(cfg.api_key.as_deref(), Some("gsk"));
        assert_eq!(
            cfg.transcriptions_url(),
            "https://api.groq.com/openai/v1/audio/transcriptions"
        );

        s.transcribe_endpoint = Some("http://127.0.0.1:8178/v1".into());
        let cfg = VoiceConfig::resolve(&s, &env_none);
        assert!(!cfg.needs_key());
    }

    struct FakeRecording {
        finished: Arc<Mutex<bool>>,
    }
    impl ActiveRecording for FakeRecording {
        fn level(&self) -> Option<f32> {
            Some(0.5)
        }
        fn check_alive(&mut self) -> Result<(), String> {
            Ok(())
        }
        fn finish(self: Box<Self>) -> Result<(), String> {
            *self.finished.lock().unwrap() = true;
            Ok(())
        }
        fn abort(self: Box<Self>) {}
    }

    struct FakeRecorder {
        finished: Arc<Mutex<bool>>,
    }
    impl AudioRecorder for FakeRecorder {
        fn plan(&self, _: Option<&str>) -> Result<recorder::RecorderPlan, String> {
            Ok(recorder::RecorderPlan::Custom {
                command: "fake".into(),
            })
        }
        fn start(
            &self,
            _: &recorder::RecorderPlan,
            _: &Path,
            _: &Path,
        ) -> Result<Box<dyn ActiveRecording>, String> {
            Ok(Box::new(FakeRecording {
                finished: self.finished.clone(),
            }))
        }
    }

    struct FakeTranscriber {
        seen_url: Arc<Mutex<String>>,
    }
    impl Transcriber for FakeTranscriber {
        fn transcribe(
            &self,
            request: &TranscriptionRequest,
            _: &Path,
            _: &AtomicBool,
        ) -> Result<String, String> {
            Ok(format!(
                "said in {}",
                request.language.as_deref().unwrap_or("?")
            ))
        }
        fn transcribe_at(
            &self,
            url: &str,
            request: &TranscriptionRequest,
            workdir: &Path,
            cancel: &AtomicBool,
        ) -> Result<String, String> {
            *self.seen_url.lock().unwrap() = url.to_string();
            self.transcribe(request, workdir, cancel)
        }
        fn verify(&self, _: &str, _: &str, _: &Path, _: &AtomicBool) -> Result<(), String> {
            Ok(())
        }
    }

    #[test]
    fn record_stop_transcribe_flow() {
        let finished = Arc::new(Mutex::new(false));
        let seen_url = Arc::new(Mutex::new(String::new()));
        let mut d = Dictation::new(
            Arc::new(FakeRecorder {
                finished: finished.clone(),
            }),
            Arc::new(FakeTranscriber {
                seen_url: seen_url.clone(),
            }),
        );
        let mut s = Settings::default();
        s.transcribe_api_key = Some("k".into());
        s.transcribe_language = Some("ur".into());
        d.start(VoiceConfig::resolve(&s, &env_none)).unwrap();
        assert!(d.is_recording());
        assert!(d.status_line("Ctrl+t").unwrap().starts_with("● REC 0:00"));
        d.stop();
        let mut event = None;
        for _ in 0..200 {
            event = d.poll();
            if event.is_some() {
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(event, Some(DictationEvent::Text("said in ur".into())));
        assert!(*finished.lock().unwrap());
        assert!(
            seen_url
                .lock()
                .unwrap()
                .ends_with("/v1/audio/transcriptions")
        );
        assert!(!d.is_active());
    }

    #[test]
    fn start_without_key_fails_before_recording() {
        let mut d = Dictation::new(
            Arc::new(FakeRecorder {
                finished: Arc::new(Mutex::new(false)),
            }),
            Arc::new(FakeTranscriber {
                seen_url: Arc::new(Mutex::new(String::new())),
            }),
        );
        let err = d
            .start(VoiceConfig::resolve(&Settings::default(), &env_none))
            .unwrap_err();
        assert!(err.contains("API key"));
        assert!(!d.is_active());
    }
}
