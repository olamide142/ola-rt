use ratatui::{
    Frame,
    layout::{Alignment, Constraint, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Gauge, Paragraph, Row, Table, Tabs, Wrap},
};

use crate::{App, ChatRole, ConnectionState, InputMode, RunState, Screen, WorkItem};

const ACCENT: Color = Color::Rgb(104, 180, 255);
const MUTED: Color = Color::Rgb(125, 135, 150);
const GOOD: Color = Color::Rgb(95, 215, 150);
const WARN: Color = Color::Rgb(245, 190, 90);
const BAD: Color = Color::Rgb(245, 105, 120);

pub(crate) fn render(frame: &mut Frame<'_>, app: &App) {
    let area = frame.area();
    if area.width < 64 || area.height < 18 {
        render_too_small(frame, area);
        return;
    }
    let [header, content, footer] = Layout::vertical([
        Constraint::Length(4),
        Constraint::Min(10),
        Constraint::Length(2),
    ])
    .areas(area);
    render_header(frame, header, app);
    match app.screen {
        Screen::Overview => render_overview(frame, content, app),
        Screen::Chat => render_chat(frame, content, app),
        Screen::Benchmarks => render_work(frame, content, app, true),
        Screen::Experiments => render_work(frame, content, app, false),
    }
    render_footer(frame, footer, app);
}

fn render_header(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let [brand, tabs] =
        Layout::horizontal([Constraint::Length(24), Constraint::Min(20)]).areas(area);
    frame.render_widget(
        Paragraph::new(vec![
            Line::from(Span::styled(
                " OLA / RT",
                Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
            )),
            Line::from(Span::styled(
                " inference workbench",
                Style::default().fg(MUTED),
            )),
        ])
        .block(Block::default().borders(Borders::BOTTOM)),
        brand,
    );
    let titles = Screen::ALL
        .into_iter()
        .enumerate()
        .map(|(index, screen)| Line::from(format!(" {} {} ", index + 1, screen.label())))
        .collect::<Vec<_>>();
    let selected = Screen::ALL
        .iter()
        .position(|screen| *screen == app.screen)
        .unwrap_or(0);
    frame.render_widget(
        Tabs::new(titles)
            .select(selected)
            .style(Style::default().fg(MUTED))
            .highlight_style(Style::default().fg(ACCENT).add_modifier(Modifier::BOLD))
            .divider("│")
            .block(Block::default().borders(Borders::BOTTOM)),
        tabs,
    );
}

fn render_overview(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let [summary, lower] =
        Layout::vertical([Constraint::Length(9), Constraint::Min(5)]).areas(area);
    let [runtime, memory, integration] = Layout::horizontal([
        Constraint::Percentage(33),
        Constraint::Percentage(34),
        Constraint::Percentage(33),
    ])
    .areas(summary);
    let data = &app.snapshot.runtime;
    let connection_color = connection_color(&data.connection);
    frame.render_widget(
        Paragraph::new(vec![
            metric_line("status", data.connection.label(), connection_color),
            metric_line("backend", fallback(&data.backend), Color::White),
            metric_line("models", &data.resident_models.to_string(), Color::White),
            metric_line("queued", &data.queued_requests.to_string(), Color::White),
            metric_line("active", &data.active_requests.to_string(), Color::White),
        ])
        .block(panel(" Runtime ")),
        runtime,
    );
    let ratio = if data.memory_capacity_bytes == 0 {
        0.0
    } else {
        data.memory_used_bytes as f64 / data.memory_capacity_bytes as f64
    }
    .clamp(0.0, 1.0);
    let [memory_text, gauge] = Layout::vertical([Constraint::Min(3), Constraint::Length(3)])
        .margin(1)
        .areas(memory);
    frame.render_widget(panel(" Memory "), memory);
    frame.render_widget(
        Paragraph::new(vec![
            metric_line("used", &format_bytes(data.memory_used_bytes), Color::White),
            metric_line(
                "capacity",
                &format_bytes(data.memory_capacity_bytes),
                Color::White,
            ),
            metric_line("KV cache", &format_bytes(data.kv_cache_bytes), ACCENT),
        ]),
        memory_text,
    );
    frame.render_widget(
        Gauge::default()
            .ratio(ratio)
            .label(format!("{:>3}%", (ratio * 100.0).round() as u8))
            .gauge_style(Style::default().fg(ACCENT)),
        gauge,
    );
    frame.render_widget(
        Paragraph::new(vec![
            metric_line(
                "LLM",
                app.snapshot
                    .selected_model
                    .as_deref()
                    .unwrap_or("not selected"),
                if app.snapshot.selected_model.is_some() {
                    GOOD
                } else {
                    MUTED
                },
            ),
            metric_line(
                "benchmarks",
                &app.snapshot.benchmarks.len().to_string(),
                Color::White,
            ),
            metric_line(
                "experiments",
                &app.snapshot.experiments.len().to_string(),
                Color::White,
            ),
            Line::from(""),
            Line::from(Span::styled(
                app.snapshot
                    .notice
                    .as_deref()
                    .unwrap_or("all integrations ready"),
                Style::default().fg(WARN),
            )),
        ])
        .wrap(Wrap { trim: true })
        .block(panel(" Integrations ")),
        integration,
    );
    let [activity, architecture] =
        Layout::horizontal([Constraint::Percentage(58), Constraint::Percentage(42)]).areas(lower);
    let activity_lines = if app.snapshot.activity.is_empty() {
        vec![Line::from(Span::styled(
            "No activity yet",
            Style::default().fg(MUTED),
        ))]
    } else {
        app.snapshot
            .activity
            .iter()
            .rev()
            .map(|entry| Line::from(format!("• {entry}")))
            .collect()
    };
    frame.render_widget(
        Paragraph::new(activity_lines)
            .wrap(Wrap { trim: true })
            .block(panel(" Activity ")),
        activity,
    );
    frame.render_widget(
        Paragraph::new(vec![
            Line::from("UI backend"),
            Line::from(Span::styled(
                "    ↓ snapshots + commands",
                Style::default().fg(MUTED),
            )),
            Line::from("OlaRT runtime / model / bench"),
            Line::from(Span::styled(
                "    ↓ backend-neutral state",
                Style::default().fg(MUTED),
            )),
            Line::from("Burn / future executors"),
        ])
        .alignment(Alignment::Center)
        .block(panel(" Extension boundary ")),
        architecture,
    );
}

fn render_chat(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let [model, transcript, input] = Layout::vertical([
        Constraint::Length(3),
        Constraint::Min(6),
        Constraint::Length(4),
    ])
    .areas(area);
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(" model  ", Style::default().fg(MUTED)),
            Span::styled(
                app.snapshot
                    .selected_model
                    .as_deref()
                    .unwrap_or("not connected"),
                Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
            ),
        ]))
        .block(panel(" LLM connection ")),
        model,
    );
    let messages = if app.snapshot.chat.is_empty() {
        vec![Line::from(Span::styled(
            "No messages yet. Press Enter or i to write a prompt.",
            Style::default().fg(MUTED),
        ))]
    } else {
        app.snapshot
            .chat
            .iter()
            .flat_map(|message| {
                let (label, color) = match message.role {
                    ChatRole::User => ("you", ACCENT),
                    ChatRole::Assistant => ("model", GOOD),
                    ChatRole::System => ("system", WARN),
                };
                [
                    Line::from(Span::styled(
                        label,
                        Style::default().fg(color).add_modifier(Modifier::BOLD),
                    )),
                    Line::from(message.content.clone()),
                    Line::from(""),
                ]
            })
            .collect()
    };
    frame.render_widget(
        Paragraph::new(messages)
            .wrap(Wrap { trim: false })
            .block(panel(" Transcript ")),
        transcript,
    );
    let input_style = if app.input_mode == InputMode::Editing {
        Style::default().fg(Color::White)
    } else {
        Style::default().fg(MUTED)
    };
    frame.render_widget(
        Paragraph::new(format!("> {}", app.prompt))
            .style(input_style)
            .block(panel(if app.input_mode == InputMode::Editing {
                " Prompt · editing "
            } else {
                " Prompt · Enter to edit "
            })),
        input,
    );
    if app.input_mode == InputMode::Editing {
        let prefix_width = Line::from(app.prompt[..app.prompt_cursor].to_owned()).width() as u16;
        frame.set_cursor_position((input.x + 3 + prefix_width, input.y + 1));
    }
}

fn render_work(frame: &mut Frame<'_>, area: Rect, app: &App, benchmarks: bool) {
    let (items, selected, title) = if benchmarks {
        (
            &app.snapshot.benchmarks,
            app.benchmark_index,
            " Benchmarks ",
        )
    } else {
        (
            &app.snapshot.experiments,
            app.experiment_index,
            " Experiments ",
        )
    };
    if items.is_empty() {
        frame.render_widget(
            Paragraph::new("No entries are exposed by the current UI backend.")
                .style(Style::default().fg(MUTED))
                .alignment(Alignment::Center)
                .block(panel(title)),
            area,
        );
        return;
    }
    let rows = items
        .iter()
        .enumerate()
        .map(|(index, item)| work_row(item, index == selected));
    frame.render_widget(
        Table::new(
            rows,
            [
                Constraint::Length(22),
                Constraint::Length(13),
                Constraint::Percentage(45),
                Constraint::Percentage(25),
            ],
        )
        .header(
            Row::new(["Name", "State", "Description", "Last result"])
                .style(Style::default().fg(MUTED).add_modifier(Modifier::BOLD))
                .bottom_margin(1),
        )
        .column_spacing(2)
        .block(panel(title)),
        area,
    );
}

fn work_row(item: &WorkItem, selected: bool) -> Row<'static> {
    let state_color = match item.state {
        RunState::Ready | RunState::Succeeded => GOOD,
        RunState::Running => WARN,
        RunState::Failed => BAD,
        RunState::Unavailable => MUTED,
    };
    let marker = if selected { "› " } else { "  " };
    let style = if selected {
        Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(state_color)
    };
    Row::new([
        format!("{marker}{}", item.name),
        item.state.label().to_owned(),
        item.description.clone(),
        item.last_result.clone().unwrap_or_else(|| "—".into()),
    ])
    .style(style)
    .height(2)
}

fn render_footer(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let shortcuts = if app.input_mode == InputMode::Editing {
        " Enter submit  Esc cancel  ←/→ move  Ctrl-C quit "
    } else {
        " Tab switch  ↑/↓ select  Enter open/run  r refresh  q quit "
    };
    let [status, help] =
        Layout::horizontal([Constraint::Percentage(45), Constraint::Percentage(55)]).areas(area);
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(" status  ", Style::default().fg(MUTED)),
            Span::raw(&app.status),
        ]))
        .block(Block::default().borders(Borders::TOP)),
        status,
    );
    frame.render_widget(
        Paragraph::new(shortcuts)
            .style(Style::default().fg(MUTED))
            .alignment(Alignment::Right)
            .block(Block::default().borders(Borders::TOP)),
        help,
    );
}

fn render_too_small(frame: &mut Frame<'_>, area: Rect) {
    frame.render_widget(
        Paragraph::new("OlaRT UI needs a terminal of at least 64 × 18")
            .alignment(Alignment::Center)
            .block(panel(" OlaRT ")),
        area,
    );
}

fn panel(title: &'static str) -> Block<'static> {
    Block::bordered()
        .title(title)
        .border_style(Style::default().fg(Color::DarkGray))
}

fn metric_line(label: &str, value: &str, value_color: Color) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("{label:<10}"), Style::default().fg(MUTED)),
        Span::styled(value.to_owned(), Style::default().fg(value_color)),
    ])
}

fn fallback(value: &str) -> &str {
    if value.is_empty() { "unknown" } else { value }
}

fn connection_color(connection: &ConnectionState) -> Color {
    match connection {
        ConnectionState::Connected => GOOD,
        ConnectionState::Busy => WARN,
        ConnectionState::Disconnected => MUTED,
        ConnectionState::Failed(_) => BAD,
    }
}

fn format_bytes(bytes: u64) -> String {
    const KIB: u64 = 1024;
    const MIB: u64 = KIB * 1024;
    const GIB: u64 = MIB * 1024;
    if bytes >= GIB {
        format!("{:.1} GiB", bytes as f64 / GIB as f64)
    } else if bytes >= MIB {
        format!("{:.1} MiB", bytes as f64 / MIB as f64)
    } else if bytes >= KIB {
        format!("{:.1} KiB", bytes as f64 / KIB as f64)
    } else {
        format!("{bytes} B")
    }
}
