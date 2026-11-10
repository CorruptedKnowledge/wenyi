//! Native diagnostic controls work without the loopback server or the React tree.
use serde_json::{json, Map, Value};
use std::{
    collections::BTreeMap,
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};
use tauri::Manager;

use crate::{
    diagnostics::{Diagnostics, Event, Range, Source, Status},
    native_export::{Saved, Saves},
};

#[derive(Default)]
pub struct ExportControl {
    busy: AtomicBool,
    cancel: AtomicBool,
}

fn trusted(window: &tauri::WebviewWindow) -> Result<(), crate::ipc_error::IpcError> {
    if window.label() == "main" && window.url().is_ok_and(|url| crate::trusted(&url)) {
        Ok(())
    } else {
        Err(crate::ipc_error::IpcError::new(
            "diagnostics_untrusted_window",
            "Desktop diagnostics are unavailable.",
        ))
    }
}

#[tauri::command]
pub async fn desktop_diagnostics_status(
    window: tauri::WebviewWindow,
) -> Result<Status, crate::ipc_error::IpcError> {
    trusted(&window)?;
    let diagnostics = window.state::<Diagnostics>().inner().clone();
    tauri::async_runtime::spawn_blocking(move || diagnostics.status())
        .await
        .map_err(|_| {
            crate::ipc_error::IpcError::new(
                "diagnostics_unavailable",
                "Cannot inspect diagnostics.",
            )
        })
}

#[tauri::command]
pub fn desktop_diagnostics_record(
    window: tauri::WebviewWindow,
    event: Event,
    attributes: Option<BTreeMap<String, Value>>,
) -> Result<Option<String>, crate::ipc_error::IpcError> {
    trusted(&window)?;
    Ok(window.state::<Diagnostics>().record(
        Source::Frontend,
        event,
        Value::Object(request_failure(event, attributes)),
    ))
}

/// Keep what the renderer may say about an event: nothing, except how a request failed.
///
/// Every other frontend event stays a bare name. A failed request may add its class,
/// status, method, route shape and the engine's identities; the recorder still bounds
/// each value, so none of them can hold a path, a message or other free text.
fn request_failure(
    event: Event,
    attributes: Option<BTreeMap<String, Value>>,
) -> Map<String, Value> {
    if event != Event::FrontendRequestFailed {
        return Map::new();
    }
    attributes
        .unwrap_or_default()
        .into_iter()
        .filter(|(key, _)| {
            matches!(
                key.as_str(),
                "category" | "status_code" | "method" | "route" | "request_id" | "error_id"
            )
        })
        .collect()
}

#[tauri::command]
pub fn desktop_diagnostics_detail(
    window: tauri::WebviewWindow,
    enabled: bool,
) -> Result<(), crate::ipc_error::IpcError> {
    trusted(&window)?;
    window.state::<Diagnostics>().detail(enabled);
    Ok(())
}

#[tauri::command]
pub fn desktop_diagnostics_cancel(
    window: tauri::WebviewWindow,
) -> Result<(), crate::ipc_error::IpcError> {
    trusted(&window)?;
    window
        .state::<Arc<ExportControl>>()
        .cancel
        .store(true, Ordering::SeqCst);
    Ok(())
}

#[tauri::command]
pub async fn desktop_diagnostics_clear(
    window: tauri::WebviewWindow,
) -> Result<(), crate::ipc_error::IpcError> {
    trusted(&window)?;
    let diagnostics = window.state::<Diagnostics>().inner().clone();
    let control = window.state::<Arc<ExportControl>>().inner().clone();
    tauri::async_runtime::spawn_blocking(move || {
        let _guard = Admission::new(&control)?;
        if rfd::MessageDialog::new().set_title("Clear application logs?")
            .set_description("Only Desktop diagnostic logs will be removed. Projects, translations and usage are kept.")
            .set_buttons(rfd::MessageButtons::YesNo).show() == rfd::MessageDialogResult::Yes {
            diagnostics.clear()?;
        }
        Ok(())
    }).await.map_err(|_| "Could not clear diagnostics.".to_string())?
}

#[tauri::command]
pub async fn desktop_diagnostics_open(
    window: tauri::WebviewWindow,
) -> Result<(), crate::ipc_error::IpcError> {
    trusted(&window)?;
    let diagnostics = window.state::<Diagnostics>().inner().clone();
    tauri::async_runtime::spawn_blocking(move || {
        let path = diagnostics
            .status()
            .directory
            .ok_or("Application logs are only in memory.")?;
        #[cfg(windows)]
        let mut command = std::process::Command::new("explorer.exe");
        #[cfg(target_os = "macos")]
        let mut command = std::process::Command::new("open");
        #[cfg(not(any(windows, target_os = "macos")))]
        let mut command = std::process::Command::new("xdg-open");
        command
            .arg(path)
            .spawn()
            .map_err(|_| "Cannot open the log folder.".to_string())?;
        Ok(())
    })
    .await
    .map_err(|_| "Cannot open the log folder.".to_string())?
}

struct Admission<'a>(&'a ExportControl);
impl<'a> Admission<'a> {
    fn new(control: &'a ExportControl) -> Result<Self, crate::ipc_error::IpcError> {
        if control
            .busy
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return Err(crate::ipc_error::IpcError::new(
                "diagnostics_busy",
                "A diagnostic operation is already running.",
            ));
        }
        control.cancel.store(false, Ordering::SeqCst);
        Ok(Self(control))
    }
}
impl Drop for Admission<'_> {
    fn drop(&mut self) {
        self.0.busy.store(false, Ordering::SeqCst);
    }
}

fn save(app: &tauri::AppHandle, range: Range) -> Result<Option<Saved>, crate::ipc_error::IpcError> {
    let control = app.state::<Arc<ExportControl>>();
    let _guard = Admission::new(&control)?;
    let diagnostics = app.state::<Diagnostics>();
    let mut roots: Vec<PathBuf> = app
        .state::<crate::Engine>()
        .workspace
        .clone()
        .into_iter()
        .collect();
    if let Some(path) = diagnostics.status().directory {
        roots.push(path.into());
    }
    // An unavailable workspace must not prevent saving a memory-only failure report.
    // Destination admission still rejects links and protects every reachable internal root.
    roots.retain(|path| path.is_dir());
    let result = crate::native_export::save_diagnostics(
        &app.state::<Saves>(),
        &roots,
        || {
            diagnostics
                .bundle(range, || control.cancel.load(Ordering::SeqCst))
                .map_err(|message| {
                    crate::ipc_error::IpcError::new("diagnostics_collection_failed", message)
                })
        },
        || control.cancel.load(Ordering::SeqCst),
    );
    match &result {
        Ok(Some(_)) => {
            diagnostics.record(Source::Native, Event::ExportCompleted, json!({}));
        }
        Err(_) if !control.cancel.load(Ordering::SeqCst) => {
            diagnostics.record(Source::Native, Event::ExportFailed, json!({}));
        }
        _ => {}
    }
    result
}

#[tauri::command]
pub async fn desktop_diagnostics_export(
    window: tauri::WebviewWindow,
    range: Range,
) -> Result<Option<Saved>, crate::ipc_error::IpcError> {
    trusted(&window)?;
    let app = window.app_handle().clone();
    tauri::async_runtime::spawn_blocking(move || save(&app, range))
        .await
        .map_err(|_| {
            crate::ipc_error::IpcError::new(
                "diagnostics_interrupted",
                "Diagnostic export stopped unexpectedly.",
            )
        })?
}

pub fn export_from_menu(app: &tauri::AppHandle) {
    let app = app.clone();
    tauri::async_runtime::spawn_blocking(move || match save(&app, Range::Day) {
        Ok(Some(_)) => {
            rfd::MessageDialog::new().set_title("Wenyi diagnostics")
                    .set_description("Diagnostic report saved. Review it before sharing; nothing was uploaded. Missing sources are listed in manifest.json.")
                    .show();
        }
        Err(error) => {
            rfd::MessageDialog::new()
                .set_title("Wenyi diagnostics")
                .set_description(error.message)
                .set_level(rfd::MessageLevel::Error)
                .show();
        }
        _ => {}
    });
}

pub fn startup_failure(diagnostics: &Diagnostics) {
    diagnostics.record(Source::Native, Event::EngineFailed, json!({}));
    if rfd::MessageDialog::new().set_title("Wenyi Desktop")
        .set_description("Desktop could not start. Save a diagnostic report? No document content or credentials are included.")
        .set_buttons(rfd::MessageButtons::YesNo).show() == rfd::MessageDialogResult::Yes {
        let roots = diagnostics.status().directory.into_iter().map(PathBuf::from).collect::<Vec<_>>();
        if crate::native_export::save_diagnostics(&Saves::default(), &roots,
            || diagnostics.bundle(Range::Day, || false).map_err(|message| {
                crate::ipc_error::IpcError::new("diagnostics_collection_failed", message)
            }), || false).is_err() {
            rfd::MessageDialog::new().set_title("Wenyi Desktop")
                .set_description("Could not save the diagnostic report. Check destination permissions and free space.")
                .set_level(rfd::MessageLevel::Error).show();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_a_failed_request_may_describe_itself() {
        let sent = BTreeMap::from([
            ("category".to_string(), json!("timeout")),
            ("method".to_string(), json!("GET")),
            ("route".to_string(), json!("/projects/:id/stats")),
            ("message".to_string(), json!("fake-private-text")),
            ("duration_ms".to_string(), json!(30000)),
        ]);
        let kept = request_failure(Event::FrontendRequestFailed, Some(sent.clone()));
        assert_eq!(
            Value::Object(kept),
            json!({"category": "timeout", "method": "GET", "route": "/projects/:id/stats"})
        );
        assert!(request_failure(Event::FrontendError, Some(sent)).is_empty());
        assert!(request_failure(Event::FrontendRequestFailed, None).is_empty());
    }
}
