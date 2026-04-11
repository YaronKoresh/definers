from __future__ import annotations

from dataclasses import dataclass
from html import escape


@dataclass(frozen=True, slots=True)
class SurfaceCard:
    launcher: str
    title: str
    description: str
    outcomes: tuple[str, ...]


HUB_CSS = """
.surface-hub {
    display: grid;
    gap: 18px;
}

.surface-hub__hero {
    position: relative;
    overflow: hidden;
    padding: 30px 34px;
    border-radius: 28px;
    border: 1px solid rgba(23, 32, 51, 0.12);
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.94) 0%, rgba(233, 245, 243, 0.84) 52%, rgba(229, 236, 241, 0.88) 100%);
    box-shadow: 0 18px 60px rgba(15, 23, 42, 0.12);
}

.surface-hub__hero::after {
    content: "";
    position: absolute;
    inset: auto -8% -35% auto;
    width: 260px;
    height: 260px;
    border-radius: 999px;
    background: radial-gradient(circle, rgba(15, 118, 110, 0.18), rgba(21, 94, 117, 0));
}

.surface-hub__eyebrow {
    margin: 0 0 10px 0;
    color: #155e75;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
}

.surface-hub__hero h1 {
    margin: 0;
    color: #172033;
    font-size: clamp(2.1rem, 3vw, 3.4rem);
    line-height: 1.02;
}

.surface-hub__hero p {
    max-width: 70ch;
    margin: 14px 0 0 0;
    color: #5b6679;
    line-height: 1.6;
}

.surface-hub__grid {
    display: grid;
    gap: 18px;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
}

.surface-card {
    display: grid;
    gap: 12px;
    padding: 22px;
    border-radius: 24px;
    border: 1px solid rgba(23, 32, 51, 0.12);
    background: rgba(255, 255, 255, 0.84);
    box-shadow: 0 18px 60px rgba(15, 23, 42, 0.08);
}

.surface-card__launcher {
    display: inline-flex;
    width: fit-content;
    padding: 6px 10px;
    border-radius: 999px;
    background: rgba(15, 118, 110, 0.12);
    color: #0f766e;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.surface-card h2 {
    margin: 0;
    color: #172033;
    font-size: 1.35rem;
}

.surface-card p,
.surface-card li,
.surface-card code {
    color: #334155;
}

.surface-card ul {
    margin: 0;
    padding-left: 18px;
}

.surface-card code {
    padding: 10px 12px;
    border-radius: 16px;
    background: #eef7f5;
    font-size: 0.92rem;
}

.surface-hub__legacy {
    margin: 0;
    padding: 16px 18px;
    border-radius: 20px;
    border: 1px solid rgba(23, 32, 51, 0.12);
    background: rgba(255, 255, 255, 0.74);
    color: #334155;
}
"""


def _render_hero_html(*, eyebrow: str, title: str, description: str) -> str:
    return (
        '<section class="surface-hub surface-hub__hero">'
        f'<p class="surface-hub__eyebrow">{escape(eyebrow)}</p>'
        f"<h1>{escape(title)}</h1>"
        f"<p>{escape(description)}</p>"
        "</section>"
    )


def _render_cards_html(cards: tuple[SurfaceCard, ...]) -> str:
    card_markup = []
    for card in cards:
        outcomes_markup = "".join(
            f"<li>{escape(outcome)}</li>" for outcome in card.outcomes
        )
        card_markup.append(
            '<article class="surface-card">'
            f'<div class="surface-card__launcher">{escape(card.launcher)}</div>'
            f"<h2>{escape(card.title)}</h2>"
            f"<p>{escape(card.description)}</p>"
            f"<ul>{outcomes_markup}</ul>"
            f"<code>definers start {escape(card.launcher)}</code>"
            "</article>"
        )
    return (
        '<section class="surface-hub surface-hub__grid">'
        + "".join(card_markup)
        + "</section>"
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
    import gradio as gr

    from definers.ui.gradio_shared import (
        css as shared_css,
        launch_blocks,
        theme as shared_theme,
    )

    with gr.Blocks(title=app_title) as app:
        gr.HTML(
            _render_hero_html(
                eyebrow=eyebrow,
                title=title,
                description=description,
            )
        )
        gr.HTML(_render_cards_html(cards))
        if legacy_command is not None:
            gr.HTML(
                f'<p class="surface-hub__legacy">Need the full advanced workbench? Run <strong>{escape(legacy_command)}</strong>.</p>'
            )
    launch_blocks(
        app,
        custom_css=shared_css() + "\n" + HUB_CSS,
        custom_theme=shared_theme(),
    )


__all__ = ["SurfaceCard", "launch_surface_hub"]
