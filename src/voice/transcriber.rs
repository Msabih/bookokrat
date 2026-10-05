//! Speech-to-text through the OpenAI transcription endpoint.
//!
//! Requests are made by spawning `curl` (no HTTP stack in the binary). The
//! API key is fed to curl on stdin as a header (`-H @-`), so it never appears
//! on a command line visible to other processes.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

pub const TRANSCRIPTIONS_URL: &str = "https://api.openai.com/v1/audio/transcriptions";
pub const MODELS_URL: &str = "https://api.openai.com/v1/models";

#[derive(Clone)]
pub struct TranscriptionRequest {
    pub audio: PathBuf,
    pub api_key: String,
    pub model: String,
    /// Comma-separated ISO-639 language hint(s), e.g. `en` or `en,fr`.
    pub language: Option<String>,
    /// Context text that helps with names and domain vocabulary.
    pub prompt: Option<String>,
}

impl std::fmt::Debug for TranscriptionRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TranscriptionRequest")
            .field("audio", &self.audio)
            .field("model", &self.model)
            .field("language", &self.language)
            .field(
                "prompt_chars",
                &self.prompt.as_ref().map(|p| p.chars().count()),
            )
            .finish_non_exhaustive()
    }
}

/// Behind a trait so tests never make network requests.
pub trait Transcriber: Send + Sync {
    /// Blocking; called from a worker thread. `workdir` is a private
    /// scratch directory for the request.
    fn transcribe(
        &self,
        request: &TranscriptionRequest,
        workdir: &Path,
        cancel: &AtomicBool,
    ) -> Result<String, String>;

    /// Like `transcribe`, against an explicit endpoint URL (OpenAI-compatible
    /// APIs: OpenAI, Groq, a local Whisper server).
    fn transcribe_at(
        &self,
        url: &str,
        request: &TranscriptionRequest,
        workdir: &Path,
        cancel: &AtomicBool,
    ) -> Result<String, String> {
        let _ = url;
        self.transcribe(request, workdir, cancel)
    }

    /// Check that the key is accepted and the model exists.
    fn verify(
        &self,
        api_key: &str,
        model: &str,
        workdir: &Path,
        cancel: &AtomicBool,
    ) -> Result<(), String>;
}

/// gpt-transcribe takes `languages[]`; older models take a single `language`.
fn uses_language_list(model: &str) -> bool {
    model.starts_with("gpt-transcribe")
}

fn supports_prompt(model: &str) -> bool {
    !model.contains("diarize")
}

fn curl_file_arg(path: &Path) -> String {
    let p = path
        .to_string_lossy()
        .replace('\\', "\\\\")
        .replace('"', "\\\"");
    format!("file=@\"{p}\";type={}", audio_mime(path))
}

fn audio_mime(path: &Path) -> &'static str {
    match path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .as_deref()
    {
        Some("mp3") | Some("mpga") | Some("mpeg") => "audio/mpeg",
        Some("m4a") | Some("mp4") => "audio/mp4",
        Some("webm") => "audio/webm",
        Some("ogg") | Some("oga") => "audio/ogg",
        Some("flac") => "audio/flac",
        _ => "audio/wav",
    }
}

/// Multipart form arguments for curl. Text fields use `--form-string` so
/// values starting with `@` or `<` are never read as files.
pub fn transcription_form_args(request: &TranscriptionRequest) -> Vec<String> {
    let mut args = vec!["-F".to_string(), curl_file_arg(&request.audio)];
    let mut field = |name: &str, value: &str| {
        args.push("--form-string".to_string());
        args.push(format!("{name}={value}"));
    };
    field("model", &request.model);
    field("response_format", "json");
    if let Some(lang) = request.language.as_deref() {
        let codes: Vec<&str> = lang
            .split([',', ' '])
            .map(str::trim)
            .filter(|c| !c.is_empty())
            .collect();
        if uses_language_list(&request.model) {
            for code in codes {
                field("languages[]", code);
            }
        } else if let Some(code) = codes.first() {
            field("language", code);
        }
    }
    if let Some(prompt) = request.prompt.as_deref()
        && supports_prompt(&request.model)
        && !prompt.trim().is_empty()
    {
        field("prompt", prompt);
    }
    args
}

#[derive(serde::Deserialize)]
struct TextResponse {
    text: String,
}

#[derive(serde::Deserialize)]
struct ErrorBody {
    error: ErrorDetail,
}

#[derive(serde::Deserialize)]
struct ErrorDetail {
    message: String,
}

pub fn api_error_message(status: u16, body: &str) -> String {
    let detail = serde_json::from_str::<ErrorBody>(body)
        .map(|b| b.error.message)
        .ok()
        .filter(|m| !m.trim().is_empty());
    let prefix = match status {
        401 => "OpenAI rejected the API key",
        403 => "OpenAI denied access",
        404 => "Not found",
        413 => "Recording too large",
        429 => "OpenAI rate limit or quota exceeded",
        500..=599 => "OpenAI server error",
        _ => "OpenAI request failed",
    };
    match detail {
        Some(d) => format!("{prefix} (HTTP {status}): {d}"),
        None => format!("{prefix} (HTTP {status})"),
    }
}

pub fn parse_transcription_response(status: u16, body: &str) -> Result<String, String> {
    if !(200..300).contains(&status) {
        return Err(api_error_message(status, body));
    }
    let text = match serde_json::from_str::<TextResponse>(body) {
        Ok(r) => r.text,
        Err(_) if !body.trim_start().starts_with('{') => body.to_string(),
        Err(e) => return Err(format!("Unexpected transcription response: {e}")),
    };
    let text = text.trim().to_string();
    if text.is_empty() {
        Err("No speech recognized".to_string())
    } else {
        Ok(text)
    }
}

/// Real transcriber backed by `curl`.
pub struct CurlTranscriber {
    pub transcriptions_url: String,
    pub models_url: String,
}

impl Default for CurlTranscriber {
    fn default() -> Self {
        Self {
            transcriptions_url: TRANSCRIPTIONS_URL.to_string(),
            models_url: MODELS_URL.to_string(),
        }
    }
}

impl CurlTranscriber {
    fn run(
        &self,
        api_key: &str,
        mut args: Vec<String>,
        url: &str,
        workdir: &Path,
        max_time_secs: u32,
        cancel: &AtomicBool,
    ) -> Result<(u16, String), String> {
        let response_path = workdir.join("response.json");
        let mut base = vec![
            "-sS".to_string(),
            "--connect-timeout".to_string(),
            "15".to_string(),
            "--max-time".to_string(),
            max_time_secs.to_string(),
        ];
        // The key goes to curl on stdin (`-H @-`), never on the command line.
        // Local servers take no key: send no Authorization header at all.
        if !api_key.is_empty() {
            base.extend(["-H".to_string(), "@-".to_string()]);
        }
        base.extend([
            "-o".to_string(),
            response_path.to_string_lossy().into_owned(),
            "-w".to_string(),
            "%{http_code}".to_string(),
        ]);
        base.append(&mut args);
        base.push(url.to_string());

        let mut child = Command::new("curl")
            .args(&base)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| {
                log::error!("Failed to spawn curl: {e}");
                if e.kind() == std::io::ErrorKind::NotFound {
                    "curl not found in PATH (needed for transcription)".to_string()
                } else {
                    format!("Failed to run curl: {e}")
                }
            })?;

        if let Some(mut stdin) = child.stdin.take()
            && !api_key.is_empty()
        {
            let header = format!("Authorization: Bearer {api_key}\n");
            if let Err(e) = stdin.write_all(header.as_bytes()) {
                log::error!("Failed to pass auth header to curl: {e}");
            }
        }

        loop {
            match child.try_wait() {
                Ok(Some(_)) => break,
                Ok(None) => {
                    if cancel.load(Ordering::Relaxed) {
                        let _ = child.kill();
                        let _ = child.wait();
                        return Err("Cancelled".to_string());
                    }
                    std::thread::sleep(Duration::from_millis(50));
                }
                Err(e) => {
                    log::error!("Failed to wait for curl: {e}");
                    let _ = child.kill();
                    return Err(format!("curl failed: {e}"));
                }
            }
        }

        let output = child.wait_with_output().map_err(|e| {
            log::error!("Failed to collect curl output: {e}");
            format!("curl failed: {e}")
        })?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let msg = stderr.trim();
            log::error!("curl exited with {}: {msg}", output.status);
            return Err(if msg.is_empty() {
                format!("Network request failed ({})", output.status)
            } else {
                format!("Network request failed: {msg}")
            });
        }
        let status: u16 = String::from_utf8_lossy(&output.stdout)
            .trim()
            .parse()
            .unwrap_or(0);
        let body = std::fs::read_to_string(&response_path).unwrap_or_default();
        Ok((status, body))
    }
}

impl Transcriber for CurlTranscriber {
    fn transcribe(
        &self,
        request: &TranscriptionRequest,
        workdir: &Path,
        cancel: &AtomicBool,
    ) -> Result<String, String> {
        self.transcribe_at(&self.transcriptions_url, request, workdir, cancel)
    }

    fn transcribe_at(
        &self,
        url: &str,
        request: &TranscriptionRequest,
        workdir: &Path,
        cancel: &AtomicBool,
    ) -> Result<String, String> {
        log::info!("Transcribing {request:?} via {url}");
        let (status, body) = self.run(
            &request.api_key,
            transcription_form_args(request),
            url,
            workdir,
            300,
            cancel,
        )?;
        let result = parse_transcription_response(status, &body);
        match &result {
            Ok(text) => log::info!("Transcription done ({} chars)", text.chars().count()),
            Err(e) => log::error!("Transcription failed: {e}"),
        }
        result
    }

    fn verify(
        &self,
        api_key: &str,
        model: &str,
        workdir: &Path,
        cancel: &AtomicBool,
    ) -> Result<(), String> {
        let url = format!("{}/{}", self.models_url.trim_end_matches('/'), model);
        let (status, body) = self.run(api_key, Vec::new(), &url, workdir, 20, cancel)?;
        if (200..300).contains(&status) {
            Ok(())
        } else if status == 404 {
            Err(format!("Model '{model}' is not available to this API key"))
        } else {
            Err(api_error_message(status, &body))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(model: &str, language: Option<&str>, prompt: Option<&str>) -> TranscriptionRequest {
        TranscriptionRequest {
            audio: PathBuf::from("/tmp/dir/audio.wav"),
            api_key: "sk-secret".into(),
            model: model.into(),
            language: language.map(Into::into),
            prompt: prompt.map(Into::into),
        }
    }

    fn pairs(args: &[String]) -> Vec<(String, String)> {
        args.chunks(2)
            .map(|c| (c[0].clone(), c[1].clone()))
            .collect()
    }

    #[test]
    fn form_args_for_gpt_transcribe_use_language_list() {
        let args = transcription_form_args(&request(
            "gpt-transcribe",
            Some("en, fr"),
            Some("@not a file"),
        ));
        let p = pairs(&args);
        assert_eq!(
            p[0],
            (
                "-F".into(),
                "file=@\"/tmp/dir/audio.wav\";type=audio/wav".into()
            )
        );
        assert!(p.contains(&("--form-string".into(), "model=gpt-transcribe".into())));
        assert!(p.contains(&("--form-string".into(), "languages[]=en".into())));
        assert!(p.contains(&("--form-string".into(), "languages[]=fr".into())));
        assert!(!args.iter().any(|a| a.starts_with("language=")));
        assert!(p.contains(&("--form-string".into(), "prompt=@not a file".into())));
        assert!(!args.iter().any(|a| a.contains("sk-secret")));
    }

    #[test]
    fn form_args_for_older_models_use_single_language() {
        let args = transcription_form_args(&request("whisper-1", Some("de,en"), None));
        assert!(args.contains(&"language=de".to_string()));
        assert!(!args.iter().any(|a| a.starts_with("languages[]")));
        assert!(!args.iter().any(|a| a.starts_with("prompt=")));
    }

    #[test]
    fn diarize_model_gets_no_prompt() {
        let args =
            transcription_form_args(&request("gpt-4o-transcribe-diarize", None, Some("context")));
        assert!(!args.iter().any(|a| a.starts_with("prompt=")));
    }

    #[test]
    fn parses_success_and_errors() {
        assert_eq!(
            parse_transcription_response(200, r#"{"text":"  Hello world. "}"#),
            Ok("Hello world.".to_string())
        );
        assert_eq!(
            parse_transcription_response(200, "plain text"),
            Ok("plain text".to_string())
        );
        assert!(parse_transcription_response(200, r#"{"text":"   "}"#).is_err());
        let err = parse_transcription_response(
            401,
            r#"{"error":{"message":"Incorrect API key provided","type":"invalid_request_error"}}"#,
        )
        .unwrap_err();
        assert!(
            err.contains("401") && err.contains("Incorrect API key"),
            "{err}"
        );
        let err = parse_transcription_response(500, "<html>").unwrap_err();
        assert!(err.contains("server error"), "{err}");
    }

    #[test]
    fn debug_never_prints_key() {
        let text = format!("{:?}", request("m", None, Some("p")));
        assert!(!text.contains("sk-secret"));
    }
}
