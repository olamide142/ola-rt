#![doc = include_str!("../README.md")]

mod app;
mod backend;
mod local;
mod model;
mod ui;

use std::{
    io,
    time::{Duration, Instant},
};

use crossterm::event::{self, Event, KeyEventKind};
use ratatui::DefaultTerminal;

pub use app::{App, InputMode, Screen, UiCommand};
pub use backend::{DisconnectedBackend, UiBackend, UiBackendError};
pub use local::LocalBackend;
pub use model::{
    ChatMessage, ChatRole, ConnectionState, RunState, RuntimeSummary, WorkItem, WorkspaceSnapshot,
};

/// Starts the fullscreen terminal UI with one pluggable data backend.
///
/// Terminal initialization, restoration, and panic cleanup are delegated to
/// Ratatui's current Crossterm-backed runner.
pub fn run<B: UiBackend>(mut backend: B) -> io::Result<()> {
    ratatui::run(|terminal| run_loop(terminal, &mut backend))
}

fn run_loop<B: UiBackend>(terminal: &mut DefaultTerminal, backend: &mut B) -> io::Result<()> {
    let snapshot = backend
        .snapshot()
        .unwrap_or_else(|error| WorkspaceSnapshot {
            notice: Some(error.to_string()),
            ..WorkspaceSnapshot::default()
        });
    let mut app = App::new(snapshot);
    let poll_interval = Duration::from_millis(250);
    let refresh_interval = Duration::from_secs(1);
    let mut last_refresh = Instant::now();

    while !app.should_quit() {
        terminal.draw(|frame| ui::render(frame, &app))?;
        if event::poll(poll_interval)?
            && let Event::Key(key) = event::read()?
            && key.kind == KeyEventKind::Press
        {
            let command = app.handle_key(key);
            execute(command, &mut app, backend);
        }
        if last_refresh.elapsed() >= refresh_interval {
            match backend.snapshot() {
                Ok(snapshot) => app.replace_snapshot(snapshot),
                Err(error) => app.set_status(error.to_string()),
            }
            last_refresh = Instant::now();
        }
    }
    Ok(())
}

fn execute<B: UiBackend>(command: UiCommand, app: &mut App, backend: &mut B) {
    let result = match command {
        UiCommand::None | UiCommand::Quit => return,
        UiCommand::Refresh => backend.snapshot().map(|snapshot| {
            app.replace_snapshot(snapshot);
            app.set_status("refreshed");
        }),
        UiCommand::SubmitPrompt(prompt) => backend.submit_prompt(&prompt).and_then(|()| {
            backend.snapshot().map(|snapshot| {
                app.replace_snapshot(snapshot);
                app.set_status("prompt submitted");
            })
        }),
        UiCommand::StartBenchmark(id) => backend.start_benchmark(&id).and_then(|()| {
            backend.snapshot().map(|snapshot| {
                app.replace_snapshot(snapshot);
                app.set_status(format!("started benchmark {id}"));
            })
        }),
        UiCommand::StartExperiment(id) => backend.start_experiment(&id).and_then(|()| {
            backend.snapshot().map(|snapshot| {
                app.replace_snapshot(snapshot);
                app.set_status(format!("started experiment {id}"));
            })
        }),
    };
    if let Err(error) = result {
        app.set_status(error.to_string());
    }
}

#[cfg(test)]
mod tests {
    use ratatui::{Terminal, backend::TestBackend};

    use super::*;

    #[test]
    fn complete_shell_renders_to_a_test_terminal() {
        let backend = TestBackend::new(120, 36);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = App::new(DisconnectedBackend::default().snapshot().unwrap());

        for screen in Screen::ALL {
            app.screen = screen;
            app.input_mode = if screen == Screen::Chat {
                InputMode::Editing
            } else {
                InputMode::Navigation
            };
            terminal.draw(|frame| ui::render(frame, &app)).unwrap();
        }

        let rendered = terminal.backend().buffer();
        assert!(format!("{rendered:?}").contains("Experiments"));
    }

    #[test]
    fn compact_terminal_uses_the_size_fallback() {
        let backend = TestBackend::new(50, 12);
        let mut terminal = Terminal::new(backend).unwrap();
        let app = App::new(WorkspaceSnapshot::default());

        terminal.draw(|frame| ui::render(frame, &app)).unwrap();

        assert!(
            format!("{:?}", terminal.backend().buffer()).contains("at least 64"),
            "small terminals should render guidance instead of overflowing layouts"
        );
    }
}
