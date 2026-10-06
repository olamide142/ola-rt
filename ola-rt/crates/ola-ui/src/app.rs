use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use crate::WorkspaceSnapshot;

/// Top-level screen selected in the application shell.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Screen {
    /// Runtime health and recent activity.
    Overview,
    /// LLM transcript and prompt editor.
    Chat,
    /// Benchmark catalog and results.
    Benchmarks,
    /// Experiment catalog and results.
    Experiments,
}

impl Screen {
    pub(crate) const ALL: [Self; 4] = [
        Self::Overview,
        Self::Chat,
        Self::Benchmarks,
        Self::Experiments,
    ];

    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Overview => "Overview",
            Self::Chat => "Chat",
            Self::Benchmarks => "Benchmarks",
            Self::Experiments => "Experiments",
        }
    }
}

/// Whether keys navigate the shell or edit the chat prompt.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum InputMode {
    /// Global navigation shortcuts are active.
    #[default]
    Navigation,
    /// Character keys edit the chat input.
    Editing,
}

/// Side effect requested by the pure application state machine.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum UiCommand {
    /// No integration work is required.
    None,
    /// Exit the terminal application.
    Quit,
    /// Fetch a fresh backend snapshot.
    Refresh,
    /// Submit a complete model prompt.
    SubmitPrompt(String),
    /// Start one benchmark by stable ID.
    StartBenchmark(String),
    /// Start one experiment by stable ID.
    StartExperiment(String),
}

/// Pure, testable state for the OlaRT terminal application.
#[derive(Clone, Debug)]
pub struct App {
    pub(crate) snapshot: WorkspaceSnapshot,
    pub(crate) screen: Screen,
    pub(crate) input_mode: InputMode,
    pub(crate) prompt: String,
    pub(crate) prompt_cursor: usize,
    pub(crate) benchmark_index: usize,
    pub(crate) experiment_index: usize,
    pub(crate) status: String,
    should_quit: bool,
}

impl App {
    /// Creates the shell around an initial coherent backend snapshot.
    pub fn new(snapshot: WorkspaceSnapshot) -> Self {
        Self {
            snapshot,
            screen: Screen::Overview,
            input_mode: InputMode::Navigation,
            prompt: String::new(),
            prompt_cursor: 0,
            benchmark_index: 0,
            experiment_index: 0,
            status: "ready".into(),
            should_quit: false,
        }
    }

    /// Returns whether the event loop should stop.
    pub const fn should_quit(&self) -> bool {
        self.should_quit
    }

    /// Replaces external data while preserving local navigation and input.
    pub fn replace_snapshot(&mut self, snapshot: WorkspaceSnapshot) {
        self.snapshot = snapshot;
        self.clamp_selections();
    }

    /// Replaces the footer status message.
    pub fn set_status(&mut self, status: impl Into<String>) {
        self.status = status.into();
    }

    /// Applies one key press and returns any requested integration command.
    pub fn handle_key(&mut self, key: KeyEvent) -> UiCommand {
        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
            self.should_quit = true;
            return UiCommand::Quit;
        }
        if self.input_mode == InputMode::Editing {
            return self.handle_editor_key(key);
        }
        self.handle_navigation_key(key)
    }

    fn handle_navigation_key(&mut self, key: KeyEvent) -> UiCommand {
        match key.code {
            KeyCode::Char('q') => {
                self.should_quit = true;
                UiCommand::Quit
            }
            KeyCode::Char('r') => UiCommand::Refresh,
            KeyCode::Tab | KeyCode::Right => {
                self.select_relative(1);
                UiCommand::None
            }
            KeyCode::BackTab | KeyCode::Left => {
                self.select_relative(-1);
                UiCommand::None
            }
            KeyCode::Char('1') => self.select_screen(Screen::Overview),
            KeyCode::Char('2') => self.select_screen(Screen::Chat),
            KeyCode::Char('3') => self.select_screen(Screen::Benchmarks),
            KeyCode::Char('4') => self.select_screen(Screen::Experiments),
            KeyCode::Up => {
                self.select_item(-1);
                UiCommand::None
            }
            KeyCode::Down => {
                self.select_item(1);
                UiCommand::None
            }
            KeyCode::Char('i') if self.screen == Screen::Chat => {
                self.input_mode = InputMode::Editing;
                UiCommand::None
            }
            KeyCode::Enter => self.activate(),
            _ => UiCommand::None,
        }
    }

    fn handle_editor_key(&mut self, key: KeyEvent) -> UiCommand {
        match key.code {
            KeyCode::Esc => {
                self.input_mode = InputMode::Navigation;
                UiCommand::None
            }
            KeyCode::Enter if !self.prompt.trim().is_empty() => {
                let prompt = self.prompt.trim().to_owned();
                self.prompt.clear();
                self.prompt_cursor = 0;
                self.input_mode = InputMode::Navigation;
                UiCommand::SubmitPrompt(prompt)
            }
            KeyCode::Char(character) => {
                self.prompt.insert(self.prompt_cursor, character);
                self.prompt_cursor += character.len_utf8();
                UiCommand::None
            }
            KeyCode::Backspace if self.prompt_cursor > 0 => {
                let previous = self.prompt[..self.prompt_cursor]
                    .char_indices()
                    .next_back()
                    .map_or(0, |(index, _)| index);
                self.prompt.drain(previous..self.prompt_cursor);
                self.prompt_cursor = previous;
                UiCommand::None
            }
            KeyCode::Delete if self.prompt_cursor < self.prompt.len() => {
                let next = self.prompt[self.prompt_cursor..]
                    .char_indices()
                    .nth(1)
                    .map_or(self.prompt.len(), |(index, _)| self.prompt_cursor + index);
                self.prompt.drain(self.prompt_cursor..next);
                UiCommand::None
            }
            KeyCode::Left if self.prompt_cursor > 0 => {
                self.prompt_cursor = self.prompt[..self.prompt_cursor]
                    .char_indices()
                    .next_back()
                    .map_or(0, |(index, _)| index);
                UiCommand::None
            }
            KeyCode::Right if self.prompt_cursor < self.prompt.len() => {
                self.prompt_cursor = self.prompt[self.prompt_cursor..]
                    .char_indices()
                    .nth(1)
                    .map_or(self.prompt.len(), |(index, _)| self.prompt_cursor + index);
                UiCommand::None
            }
            KeyCode::Home => {
                self.prompt_cursor = 0;
                UiCommand::None
            }
            KeyCode::End => {
                self.prompt_cursor = self.prompt.len();
                UiCommand::None
            }
            _ => UiCommand::None,
        }
    }

    fn activate(&mut self) -> UiCommand {
        match self.screen {
            Screen::Overview => UiCommand::Refresh,
            Screen::Chat => {
                self.input_mode = InputMode::Editing;
                UiCommand::None
            }
            Screen::Benchmarks => self
                .snapshot
                .benchmarks
                .get(self.benchmark_index)
                .map_or(UiCommand::None, |item| {
                    UiCommand::StartBenchmark(item.id.clone())
                }),
            Screen::Experiments => self
                .snapshot
                .experiments
                .get(self.experiment_index)
                .map_or(UiCommand::None, |item| {
                    UiCommand::StartExperiment(item.id.clone())
                }),
        }
    }

    fn select_screen(&mut self, screen: Screen) -> UiCommand {
        self.screen = screen;
        UiCommand::None
    }

    fn select_relative(&mut self, delta: isize) {
        let current = Screen::ALL
            .iter()
            .position(|screen| *screen == self.screen)
            .unwrap_or(0);
        let len = Screen::ALL.len() as isize;
        self.screen = Screen::ALL[(current as isize + delta).rem_euclid(len) as usize];
    }

    fn select_item(&mut self, delta: isize) {
        match self.screen {
            Screen::Benchmarks => {
                self.benchmark_index =
                    move_selection(self.benchmark_index, self.snapshot.benchmarks.len(), delta);
            }
            Screen::Experiments => {
                self.experiment_index = move_selection(
                    self.experiment_index,
                    self.snapshot.experiments.len(),
                    delta,
                );
            }
            Screen::Overview | Screen::Chat => {}
        }
    }

    fn clamp_selections(&mut self) {
        self.benchmark_index =
            clamp_selection(self.benchmark_index, self.snapshot.benchmarks.len());
        self.experiment_index =
            clamp_selection(self.experiment_index, self.snapshot.experiments.len());
    }
}

fn move_selection(current: usize, len: usize, delta: isize) -> usize {
    if len == 0 {
        return 0;
    }
    (current as isize + delta).rem_euclid(len as isize) as usize
}

fn clamp_selection(current: usize, len: usize) -> usize {
    current.min(len.saturating_sub(1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{RunState, WorkItem};

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    #[test]
    fn chat_editor_handles_unicode_and_emits_a_trimmed_prompt() {
        let mut app = App::new(WorkspaceSnapshot::default());
        app.handle_key(key(KeyCode::Char('2')));
        app.handle_key(key(KeyCode::Enter));
        app.handle_key(key(KeyCode::Char(' ')));
        app.handle_key(key(KeyCode::Char('λ')));
        app.handle_key(key(KeyCode::Char(' ')));

        assert_eq!(
            app.handle_key(key(KeyCode::Enter)),
            UiCommand::SubmitPrompt("λ".into())
        );
        assert_eq!(app.input_mode, InputMode::Navigation);
        assert!(app.prompt.is_empty());
    }

    #[test]
    fn benchmark_selection_wraps_and_uses_stable_ids() {
        let item = |id: &str| WorkItem {
            id: id.into(),
            name: id.into(),
            description: String::new(),
            state: RunState::Ready,
            last_result: None,
        };
        let mut app = App::new(WorkspaceSnapshot {
            benchmarks: vec![item("first"), item("second")],
            ..WorkspaceSnapshot::default()
        });
        app.handle_key(key(KeyCode::Char('3')));
        app.handle_key(key(KeyCode::Up));

        assert_eq!(
            app.handle_key(key(KeyCode::Enter)),
            UiCommand::StartBenchmark("second".into())
        );
    }
}
