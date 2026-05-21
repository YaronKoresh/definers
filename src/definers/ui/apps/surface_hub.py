from __future__ import annotations

from dataclasses import dataclass
from html import escape


@dataclass(frozen=True, slots=True)
class SurfaceCard:
    launcher: str
    title: str
    description: str
    outcomes: tuple[str, ...]


def _render_hero_html(*, eyebrow: str, title: str, description: str) -> str:
    return (
        "<div>"
        '<p style="color:#bdf6ff;font-size:11px;font-weight:700;'
        'letter-spacing:3px;margin:0 0 10px 0;">' + escape(eyebrow) + "</p>"
        '<div style="color:#f9fcff;font-size:26px;font-weight:700;margin:0 0 8px 0;">'
        + escape(title)
        + "</div>"
        '<p style="color:#c8d3e2;font-size:14px;line-height:1.55;margin:0;">'
        + escape(description)
        + "</p></div>"
    )


def _render_card_html(card: SurfaceCard) -> str:
    outcomes_markup = "".join(
        f'<li style="color:#c2cfe0;font-size:13px;">{escape(outcome)}</li>'
        for outcome in card.outcomes
    )
    return (
        '<span style="background:rgba(39,184,234,0.18);color:#bdf6ff;'
        "padding:3px 8px;border-radius:8px;font-size:11px;font-weight:700;"
        'letter-spacing:1.5px;">'
        + escape(card.launcher)
        + "</span>"
        + '<div style="color:#eef6ff;font-size:15px;font-weight:700;margin:8px 0 4px 0;">'
        + escape(card.title)
        + "</div>"
        + '<p style="color:#c2cfe0;font-size:13px;line-height:1.5;margin:0 0 8px 0;">'
        + escape(card.description)
        + "</p>"
        + f'<ul style="margin:0 0 8px 0;padding-left:18px;">{outcomes_markup}</ul>'
        + '<div style="background:rgba(84,226,255,0.08);padding:6px 10px;'
        "border-radius:6px;font-family:Consolas,monospace;font-size:12px;"
        'color:#7a8fa8;">definers start ' + escape(card.launcher) + "</div>"
    )


def launch_surface_hub(
    *,
    app_title: str,
    eyebrow: str,
    title: str,
    description: str,
    cards: tuple[SurfaceCard, ...],
    legacy_command: str | None = None,
):
    from definers.ui.gui_shared import launch_blocks, qt_app
    from definers.ui.qt_widgets import (
        CardFrame,
        HeroCard,
        HtmlLabel,
        Row,
        Section,
        build_main_window,
    )

    qt_app()
    body = Section()
    body.addWidget(
        HeroCard(
            _render_hero_html(
                eyebrow=eyebrow,
                title=title,
                description=description,
            )
        )
    )

    current_row: Row | None = None
    for index, card in enumerate(cards):
        if index % 2 == 0:
            current_row = Row()
            body.addWidget(current_row)
        frame = CardFrame()
        frame.addWidget(HtmlLabel(_render_card_html(card)))
        assert current_row is not None
        current_row.addWidget(frame)

    if legacy_command is not None:
        body.addWidget(
            HtmlLabel(
                '<p style="color:#c2cfe0;font-size:13px;">'
                "Need the full advanced workbench? Run <strong>"
                '<span style="color:#bdf6ff;">'
                + escape(legacy_command)
                + "</span></strong>.</p>"
            )
        )

    window = build_main_window(app_title, body)
    launch_blocks(window)


__all__ = ["SurfaceCard", "launch_surface_hub"]
