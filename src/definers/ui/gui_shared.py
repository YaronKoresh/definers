from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

from PySide6.QtCore import (
    QAbstractListModel,
    QModelIndex,
    QObject,
    QSize,
    Qt,
    QThread,
    Signal,
    Slot,
)
from PySide6.QtGui import QColor, QPalette, QTextOption
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListView,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QStyledItemDelegate,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

ChatHandler = object


def qt_app() -> QApplication:

    existing = QApplication.instance()
    if existing is not None:
        return existing
    application = QApplication(sys.argv if sys.argv else ["definers"])
    application.setApplicationName("Definers")
    application.setOrganizationName("Definers")
    application.setStyle("Fusion")
    application.setPalette(qt_palette())
    application.setStyleSheet(qt_stylesheet())
    return application


def qt_palette() -> QPalette:

    palette = QPalette()
    palette.setColor(QPalette.Window, QColor("#111b2d"))
    palette.setColor(QPalette.WindowText, QColor("#eef6ff"))
    palette.setColor(QPalette.Base, QColor("#17223a"))
    palette.setColor(QPalette.AlternateBase, QColor("#1b2742"))
    palette.setColor(QPalette.ToolTipBase, QColor("#17223a"))
    palette.setColor(QPalette.ToolTipText, QColor("#eef6ff"))
    palette.setColor(QPalette.Text, QColor("#eef6ff"))
    palette.setColor(QPalette.Button, QColor("#1e2d46"))
    palette.setColor(QPalette.ButtonText, QColor("#eef6ff"))
    palette.setColor(QPalette.Highlight, QColor("#27b8ea"))
    palette.setColor(QPalette.HighlightedText, QColor("#01111f"))
    palette.setColor(QPalette.Link, QColor("#54e2ff"))
    palette.setColor(QPalette.PlaceholderText, QColor("#6a80a0"))
    return palette


def qt_stylesheet() -> str:

    return """
/* ── Root ──────────────────────────────────────────────────────────── */
QWidget {
    background-color: #111b2d;
    color: #eef6ff;
    font-family: "Heebo", "Assistant", "Segoe UI", "Inter", "Helvetica Neue", "Arial", sans-serif;
    font-size: 15px;
}
QMainWindow, QDialog {
    background-color: #111b2d;
}
QScrollArea {
    background-color: #111b2d;
    border: none;
}
QScrollArea > QWidget > QWidget {
    background-color: #111b2d;
}

/* ── Semantic named frames ──────────────────────────────────────────── */
QFrame#DefinersCard {
    background-color: #17223a;
    border: 1px solid rgba(120, 145, 175, 0.20);
    border-left: 3px solid rgba(84, 226, 255, 0.35);
    border-radius: 12px;
    padding: 0;
}
QFrame#DefinersHero {
    background-color: #17223a;
    border: 1px solid rgba(84, 226, 255, 0.28);
    border-radius: 16px;
    padding: 0;
}

/* ── Labels (MarkdownLabel / plain QLabel) ─────────────────────────── */
QLabel {
    color: #eef6ff;
    background-color: transparent;
}

/* ── Buttons ────────────────────────────────────────────────────────── */
QPushButton {
    background-color: #1e2d46;
    color: #d8e8f8;
    border: 1px solid rgba(120, 145, 175, 0.30);
    border-radius: 9px;
    padding: 10px 22px;
    font-weight: 600;
    font-size: 15px;
    min-height: 36px;
}
QPushButton:hover {
    background-color: #233553;
    border-color: rgba(84, 226, 255, 0.50);
    color: #eef6ff;
}
QPushButton:pressed {
    background-color: #17223a;
    border-color: rgba(84, 226, 255, 0.65);
}
QPushButton:disabled {
    color: #3e5060;
    background-color: #141c2e;
    border-color: rgba(120, 145, 175, 0.10);
}
QPushButton[definersPrimary="true"] {
    background-color: #1aa0d6;
    color: #01111f;
    border: 1px solid rgba(189, 246, 255, 0.55);
    font-weight: 700;
    letter-spacing: 0.3px;
}
QPushButton[definersPrimary="true"]:hover {
    background-color: #27b8ea;
    border-color: #bdf6ff;
    color: #000d18;
}
QPushButton[definersPrimary="true"]:pressed {
    background-color: #0d7da8;
    border-color: rgba(84, 226, 255, 0.70);
}
QPushButton[definersPrimary="true"]:disabled {
    background-color: #112535;
    color: rgba(1, 17, 31, 0.50);
    border-color: rgba(84, 226, 255, 0.15);
}

/* ── Text inputs ────────────────────────────────────────────────────── */
QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox {
    background-color: #141c2e;
    color: #eef6ff;
    border: 1px solid rgba(120, 145, 175, 0.22);
    border-radius: 9px;
    padding: 9px 12px;
    selection-background-color: #27b8ea;
    selection-color: #01111f;
    min-height: 28px;
    font-size: 15px;
}
QLineEdit:hover, QTextEdit:hover, QPlainTextEdit:hover,
QSpinBox:hover, QDoubleSpinBox:hover {
    border-color: rgba(84, 226, 255, 0.30);
}
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus,
QSpinBox:focus, QDoubleSpinBox:focus {
    border-color: rgba(84, 226, 255, 0.62);
    background-color: #17223a;
    outline: none;
}

/* ── Combo box ──────────────────────────────────────────────────────── */
QComboBox {
    background-color: #141c2e;
    color: #eef6ff;
    border: 1px solid rgba(120, 145, 175, 0.22);
    border-radius: 9px;
    padding: 9px 36px 9px 12px;
    min-height: 28px;
    font-size: 15px;
}
QComboBox:hover {
    border-color: rgba(84, 226, 255, 0.30);
}
QComboBox:focus {
    border-color: rgba(84, 226, 255, 0.62);
    background-color: #17223a;
}
QComboBox::drop-down {
    border: none;
    width: 28px;
}
QComboBox::down-arrow {
    width: 10px;
    height: 10px;
}
QComboBox QAbstractItemView {
    background-color: #17223a;
    color: #eef6ff;
    selection-background-color: #27b8ea;
    selection-color: #01111f;
    border: 1px solid rgba(84, 226, 255, 0.25);
    border-radius: 8px;
    padding: 4px;
    outline: none;
}

/* ── Tab widget ─────────────────────────────────────────────────────── */
QTabWidget::pane {
    background-color: #161e30;
    border: 1px solid rgba(120, 145, 175, 0.20);
    border-radius: 12px;
    top: -1px;
}
QTabBar {
    qproperty-drawBase: 0;
}
QTabBar::tab {
    background-color: transparent;
    color: #7a8fa8;
    padding: 11px 22px;
    border: 1px solid transparent;
    border-bottom: none;
    border-top-left-radius: 9px;
    border-top-right-radius: 9px;
    margin-right: 3px;
    font-weight: 600;
    font-size: 14px;
    letter-spacing: 0.2px;
}
QTabBar::tab:selected {
    background-color: #17223a;
    color: #eef6ff;
    border-color: rgba(84, 226, 255, 0.35);
    border-bottom-color: #161e30;
}
QTabBar::tab:hover:!selected {
    color: #c2cfe0;
    background-color: rgba(84, 226, 255, 0.06);
}

/* ── Progress bar ───────────────────────────────────────────────────── */
QProgressBar {
    background-color: #17223a;
    border: 1px solid rgba(120, 145, 175, 0.20);
    border-radius: 5px;
    height: 8px;
    text-align: center;
    color: transparent;
    max-height: 8px;
}
QProgressBar::chunk {
    background-color: #27b8ea;
    border-radius: 5px;
}
QProgressBar[definersState="success"]::chunk { background-color: #4ade80; }
QProgressBar[definersState="error"]::chunk { background-color: #f87171; }
QProgressBar[definersState="idle"]::chunk {
    background-color: rgba(84, 226, 255, 0.18);
}

/* ── Sliders ────────────────────────────────────────────────────────── */
QSlider::groove:horizontal {
    height: 5px;
    background: #233553;
    border-radius: 3px;
    margin: 1px 0;
}
QSlider::sub-page:horizontal {
    background: qlineargradient(
        x1:0, y1:0, x2:1, y2:0,
        stop:0 #1aa0d6, stop:1 #27b8ea
    );
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #eef6ff;
    border: 2px solid #27b8ea;
    width: 14px;
    height: 14px;
    margin: -6px 0;
    border-radius: 8px;
}
QSlider::handle:horizontal:hover {
    background: #ffffff;
    border-color: #bdf6ff;
}
QSlider::handle:horizontal:pressed {
    background: #bdf6ff;
}

/* ── Checkboxes / radio buttons ─────────────────────────────────────── */
QCheckBox, QRadioButton {
    color: #c2cfe0;
    spacing: 8px;
    font-size: 15px;
}
QCheckBox::indicator, QRadioButton::indicator {
    width: 17px;
    height: 17px;
    border: 1px solid rgba(120, 145, 175, 0.45);
    border-radius: 5px;
    background-color: #141c2e;
}
QCheckBox::indicator:hover, QRadioButton::indicator:hover {
    border-color: rgba(84, 226, 255, 0.50);
}
QCheckBox::indicator:checked {
    background-color: #27b8ea;
    border-color: #1aa0d6;
}
QRadioButton::indicator { border-radius: 9px; }
QRadioButton::indicator:checked {
    background-color: #27b8ea;
    border-color: #1aa0d6;
}

/* ── MarkdownLabel (inherits from QLabel) ───────────────────────────── */
MarkdownLabel {
    color: #c2cfe0;
    background-color: transparent;
    font-size: 15px;
    line-height: 1.65;
    padding: 0 6px;
}

/* ── Cards / hero sections ─────────────────────────────────────────── */
HeroCard {
    background-color: #17223a;
    border: 1px solid rgba(84, 226, 255, 0.22);
    border-radius: 16px;
    padding: 22px 24px;
}

/* ── Labeled widgets ───────────────────────────────────────────────── */
LabeledWidget {
    color: #7a8fa8;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.5px;
    background: transparent;
}

/* ── Divider ───────────────────────────────────────────────────────── */
DividerLabel {
    border-bottom: 1px solid rgba(84, 226, 255, 0.22);
    padding-bottom: 10px;
    margin-bottom: 2px;
}

/* ── Scrollbars ─────────────────────────────────────────────────────── */
QScrollBar:vertical {
    background: transparent;
    width: 10px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background: rgba(120, 145, 175, 0.22);
    border-radius: 5px;
    min-height: 28px;
}
QScrollBar::handle:vertical:hover { background: rgba(84, 226, 255, 0.40); }
QScrollBar:horizontal {
    background: transparent;
    height: 10px;
    margin: 0;
}
QScrollBar::handle:horizontal {
    background: rgba(120, 145, 175, 0.22);
    border-radius: 5px;
    min-width: 28px;
}
QScrollBar::handle:horizontal:hover { background: rgba(84, 226, 255, 0.40); }
QScrollBar::add-line, QScrollBar::sub-line {
    background: transparent; height: 0; width: 0;
}
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }

/* ── Group box ──────────────────────────────────────────────────────── */
QGroupBox {
    border: 1px solid rgba(120, 145, 175, 0.20);
    border-radius: 10px;
    margin-top: 16px;
    padding: 10px 10px 8px 10px;
    color: #eef6ff;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 8px;
    color: #bdf6ff;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
}

/* ── Accordion ──────────────────────────────────────────────────────── */
Accordion::indicator {
    width: 14px; height: 14px;
    border: 2px solid rgba(84,226,255,0.22); border-radius: 3px;
}
Accordion::indicator:hover {
    border: 2px solid rgba(84,226,255,0.42);
}

Accordion::indicator:checked {
    background: #1aa0d6; border: 2px solid #fff;
}

/* ── List / tree / table views ──────────────────────────────────────── */
QListView, QListWidget, QTreeView, QTableView {
    background-color: #17223a;
    color: #eef6ff;
    border: 1px solid rgba(120, 145, 175, 0.18);
    border-radius: 10px;
    alternate-background-color: #1b2742;
    gridline-color: rgba(120, 145, 175, 0.12);
    outline: none;
}
QListView::item, QListWidget::item {
    padding: 10px 12px;
    border-radius: 6px;
    border-bottom: 1px solid rgba(120, 145, 175, 0.08);
}
QListView::item:selected, QListWidget::item:selected {
    background-color: rgba(27, 160, 214, 0.22);
    color: #eef6ff;
    border: 1px solid rgba(84, 226, 255, 0.30);
}
QListView::item:hover, QListWidget::item:hover {
    background-color: rgba(84, 226, 255, 0.06);
}
QHeaderView::section {
    background-color: #1e2d46;
    color: #bdf6ff;
    border: none;
    border-bottom: 1px solid rgba(84, 226, 255, 0.20);
    padding: 8px 10px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.5px;
}

/* ── Menus ──────────────────────────────────────────────────────────── */
QMenu {
    background-color: #17223a;
    color: #eef6ff;
    border: 1px solid rgba(84, 226, 255, 0.22);
    border-radius: 10px;
    padding: 6px;
}
QMenu::item {
    padding: 8px 20px;
    border-radius: 6px;
}
QMenu::item:selected {
    background-color: rgba(27, 160, 214, 0.28);
    color: #eef6ff;
}
QMenu::separator {
    height: 1px;
    background: rgba(120, 145, 175, 0.18);
    margin: 4px 10px;
}

/* ── Status / tool bars ─────────────────────────────────────────────── */
QStatusBar {
    background-color: #0e162a;
    color: #7a8fa8;
    border-top: 1px solid rgba(120, 145, 175, 0.15);
}
QToolBar {
    background-color: #161e30;
    border-bottom: 1px solid rgba(120, 145, 175, 0.15);
    spacing: 4px;
    padding: 4px;
}

/* ── Tooltip ────────────────────────────────────────────────────────── */
QToolTip {
    background-color: #17223a;
    color: #eef6ff;
    border: 1px solid rgba(84, 226, 255, 0.30);
    border-radius: 6px;
    padding: 7px 12px;
    font-size: 14px;
}
"""


def _format_bytes(value: int | None) -> str | None:
    if value is None:
        return None
    try:
        normalized = max(int(value), 0)
    except Exception:
        return None
    suffixes = ("B", "KiB", "MiB", "GiB", "TiB")
    scaled = float(normalized)
    suffix = suffixes[0]
    for suffix in suffixes:
        if scaled < 1024.0 or suffix == suffixes[-1]:
            break
        scaled /= 1024.0
    if suffix == "B":
        return f"{int(scaled)} {suffix}"
    return f"{scaled:.1f} {suffix}"


@dataclass
class ProgressState:
    title: str = "Workspace ready"
    state: str = "idle"
    detail: str = "Choose an action to begin."
    steps: tuple[str, ...] = ()
    active_step: int | None = None
    activity_completed: int | None = None
    activity_total: int | None = None
    bytes_downloaded: int | None = None
    bytes_total: int | None = None

    @property
    def fraction(self) -> float:
        steps = self.steps
        normalized_state = (self.state or "idle").lower()
        activity_fraction: float | None = None
        if self.activity_total and self.activity_completed is not None:
            activity_fraction = max(
                0.0,
                min(
                    float(self.activity_completed) / float(self.activity_total),
                    1.0,
                ),
            )
        elif self.bytes_total and self.bytes_downloaded is not None:
            activity_fraction = max(
                0.0,
                min(
                    float(self.bytes_downloaded) / float(self.bytes_total),
                    1.0,
                ),
            )
        if steps:
            if normalized_state == "success":
                return 1.0
            if normalized_state == "idle":
                return 0.0
            bounded = max(1, min(int(self.active_step or 1), len(steps)))
            completed = bounded - 1
            resolved_fraction = (
                activity_fraction
                if activity_fraction is not None
                and normalized_state == "running"
                else 0.0
                if normalized_state == "running"
                else 1.0
            )
            return (completed + resolved_fraction) / len(steps)
        if normalized_state == "running" and activity_fraction is not None:
            return activity_fraction
        if normalized_state == "success":
            return 1.0
        return 0.0


def _progress_markup(
    title: str,
    state: str = "idle",
    detail: str | None = None,
    steps: tuple[str, ...] | list[str] | None = None,
    active_step: int | None = None,
    activity_completed: int | None = None,
    activity_total: int | None = None,
    bytes_downloaded: int | None = None,
    bytes_total: int | None = None,
) -> str:

    normalized_state = str(state or "idle").strip().lower()
    detail_text = (
        "Choose an action to begin."
        if detail is None
        else str(detail).strip() or "Choose an action to begin."
    )
    state_label = {
        "idle": "Ready",
        "running": "Running",
        "success": "Done",
        "error": "Failed",
    }.get(normalized_state, "Ready")
    resolved_steps = tuple(
        str(step).strip() for step in tuple(steps or ()) if str(step).strip()
    )
    completed_count = 0
    current_step_label: str | None = None
    remaining_steps: tuple[str, ...] = resolved_steps
    if resolved_steps:
        if normalized_state == "success":
            completed_count = len(resolved_steps)
            remaining_steps = ()
        elif normalized_state != "idle":
            bounded = max(1, min(int(active_step or 1), len(resolved_steps)))
            completed_count = bounded - 1
            current_step_label = resolved_steps[bounded - 1]
            remaining_steps = resolved_steps[bounded:]
    done_text = (
        ", ".join(resolved_steps[:completed_count])
        if completed_count
        else "Nothing completed yet"
    )
    current_text = current_step_label or (
        "Waiting to start"
        if normalized_state == "idle"
        else "Completed"
        if normalized_state == "success"
        else detail_text
    )
    if normalized_state == "running" and activity_total:
        completed = max(int(activity_completed or 0), 0)
        current_text = f"{current_text} ({completed}/{int(activity_total)})"
    elif normalized_state == "running" and bytes_total:
        downloaded_text = _format_bytes(bytes_downloaded)
        total_text = _format_bytes(bytes_total)
        if downloaded_text and total_text:
            current_text = f"{current_text} ({downloaded_text} / {total_text})"
    remaining_text = (
        ", ".join(remaining_steps)
        if remaining_steps
        else "Start a workflow to load steps"
        if not resolved_steps and normalized_state == "idle"
        else "Nothing left"
    )
    return (
        f"<b>{escape(state_label)}</b> &mdash; "
        f"<b>{escape(str(title or 'Workspace status'))}</b><br/>"
        f"<span>{escape(detail_text)}</span><br/>"
        f"<small>Done: {escape(done_text)} &middot; "
        f"Now: {escape(current_text)} &middot; "
        f"Left: {escape(remaining_text)}</small>"
    )


class ProgressTracker(QFrame):
    def __init__(
        self,
        title: str = "Workspace ready",
        detail: str = "Choose an action to begin.",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("DefinersCard")
        self.setFrameShape(QFrame.NoFrame)
        self._state = ProgressState(title=title, detail=detail)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(8)
        self._badge = QLabel("Ready", self)
        self._badge.setStyleSheet(
            "color: #bdf6ff; font-size: 11px; font-weight: 700;"
            " letter-spacing: 2.5px; background: transparent;"
        )
        self._title_label = QLabel(title, self)
        self._title_label.setStyleSheet(
            "color: #eef6ff; font-size: 17px; font-weight: 700;"
            " background: transparent;"
        )
        self._count_label = QLabel("", self)
        self._count_label.setStyleSheet(
            "color: #7a8fa8; font-size: 13px; background: transparent;"
        )
        self._count_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        header.addWidget(self._badge)
        header.addWidget(self._title_label, stretch=1)
        header.addWidget(self._count_label)
        layout.addLayout(header)

        self._detail_label = QLabel(detail, self)
        self._detail_label.setWordWrap(True)
        self._detail_label.setStyleSheet(
            "color: #7a8fa8; font-size: 14px; background: transparent;"
        )
        layout.addWidget(self._detail_label)

        summary_row = QHBoxLayout()
        summary_row.setContentsMargins(0, 0, 0, 0)
        summary_row.setSpacing(8)
        self._done_cell = self._make_summary_cell(
            "Done", "Nothing completed yet"
        )
        self._now_cell = self._make_summary_cell("Now", "Waiting to start")
        self._left_cell = self._make_summary_cell(
            "Left", "Start a workflow to load steps"
        )
        for cell, _ in (self._done_cell, self._now_cell, self._left_cell):
            summary_row.addWidget(cell, stretch=1)
        layout.addLayout(summary_row)

        self._bar = QProgressBar(self)
        self._bar.setRange(0, 1000)
        self._bar.setValue(0)
        self._bar.setTextVisible(False)
        self._bar.setProperty("definersState", "idle")
        layout.addWidget(self._bar)

        self._steps_label = QLabel("", self)
        self._steps_label.setWordWrap(True)
        self._steps_label.setStyleSheet(
            "color: #7a8fa8; font-size: 13px; background: transparent;"
        )
        self._steps_label.setVisible(False)
        layout.addWidget(self._steps_label)

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

    def _make_summary_cell(
        self,
        eyebrow: str,
        placeholder: str,
    ) -> tuple[QFrame, QLabel]:
        cell = QFrame(self)
        cell.setObjectName("DefinersCard")
        cell_layout = QVBoxLayout(cell)
        cell_layout.setContentsMargins(8, 6, 8, 6)
        cell_layout.setSpacing(2)
        eyebrow_label = QLabel(eyebrow, cell)
        eyebrow_label.setStyleSheet(
            "color: #7a8fa8; font-size: 11px; font-weight: 700;"
            " letter-spacing: 1.5px; background: transparent;"
        )
        value_label = QLabel(placeholder, cell)
        value_label.setWordWrap(True)
        value_label.setStyleSheet(
            "color: #eef6ff; font-size: 14px; background: transparent;"
        )
        cell_layout.addWidget(eyebrow_label)
        cell_layout.addWidget(value_label)
        return cell, value_label

    @Slot(dict)
    def apply_update(self, payload: dict[str, Any]) -> None:

        if not isinstance(payload, dict):
            return
        current = self._state
        new_steps = payload.get("steps", current.steps)
        if new_steps is None:
            new_steps = ()
        self._state = ProgressState(
            title=payload.get("title", current.title) or current.title,
            state=payload.get("state", current.state) or current.state,
            detail=payload.get("detail", current.detail) or current.detail,
            steps=tuple(new_steps),
            active_step=payload.get("active_step", current.active_step),
            activity_completed=payload.get(
                "activity_completed", current.activity_completed
            ),
            activity_total=payload.get(
                "activity_total", current.activity_total
            ),
            bytes_downloaded=payload.get(
                "bytes_downloaded", current.bytes_downloaded
            ),
            bytes_total=payload.get("bytes_total", current.bytes_total),
        )
        self._render()

    def state(self) -> ProgressState:
        return self._state

    def reset(self) -> None:
        self._state = ProgressState(
            title=self._state.title,
            detail="Choose an action to begin.",
        )
        self._render()

    def _render(self) -> None:
        st = self._state
        normalized_state = (st.state or "idle").lower()
        state_label = {
            "idle": "Ready",
            "running": "Running",
            "success": "Done",
            "error": "Failed",
        }.get(normalized_state, "Ready")
        self._badge.setText(state_label)
        self._title_label.setText(st.title or "Workspace status")
        self._detail_label.setText(st.detail or "Choose an action to begin.")
        steps = tuple(s for s in (st.steps or ()) if str(s).strip())
        completed_count = 0
        current_label: str | None = None
        remaining: tuple[str, ...] = steps
        if steps:
            if normalized_state == "success":
                completed_count = len(steps)
                remaining = ()
            elif normalized_state != "idle":
                bounded = max(1, min(int(st.active_step or 1), len(steps)))
                completed_count = bounded - 1
                current_label = steps[bounded - 1]
                remaining = steps[bounded:]
        if steps and normalized_state == "running" and st.activity_total:
            self._count_label.setText(
                f"{int(st.active_step or 0)}/{len(steps)} · "
                f"{max(int(st.activity_completed or 0), 0)}/"
                f"{int(st.activity_total)}"
            )
        elif steps:
            self._count_label.setText(f"{completed_count}/{len(steps)}")
        else:
            self._count_label.setText("")
        done_text = (
            ", ".join(steps[:completed_count])
            if completed_count
            else "Nothing completed yet"
        )
        current_text = current_label or (
            "Waiting to start"
            if normalized_state == "idle"
            else "Completed"
            if normalized_state == "success"
            else (st.detail or "Working")
        )
        if normalized_state == "running" and st.activity_total:
            completed_n = max(int(st.activity_completed or 0), 0)
            current_text = (
                f"{current_text} ({completed_n}/{int(st.activity_total)})"
            )
        elif normalized_state == "running" and st.bytes_total:
            downloaded_text = _format_bytes(st.bytes_downloaded)
            total_text = _format_bytes(st.bytes_total)
            if downloaded_text and total_text:
                current_text = (
                    f"{current_text} ({downloaded_text} / {total_text})"
                )
        remaining_text = (
            ", ".join(remaining)
            if remaining
            else "Start a workflow to load steps"
            if not steps and normalized_state == "idle"
            else "Nothing left"
        )
        self._done_cell[1].setText(done_text)
        self._now_cell[1].setText(current_text)
        self._left_cell[1].setText(remaining_text)

        fraction = max(0.0, min(st.fraction, 1.0))
        self._bar.setValue(int(round(fraction * 1000)))
        self._bar.setProperty("definersState", normalized_state)
        self._bar.style().unpolish(self._bar)
        self._bar.style().polish(self._bar)

        if steps:
            entries = []
            for index, step_label in enumerate(steps, start=1):
                marker = "•"
                if normalized_state == "success" or index <= completed_count:
                    marker = "✓"
                elif index == completed_count + 1:
                    marker = "✗" if normalized_state == "error" else "▶"
                entries.append(f"{marker} {step_label}")
            self._steps_label.setText("  ".join(entries))
            self._steps_label.setVisible(True)
        else:
            self._steps_label.setVisible(False)


def init_progress_tracker(
    title: str = "Workspace ready",
    detail: str = "Choose an action to begin.",
) -> ProgressTracker:

    return ProgressTracker(title=title, detail=detail)


def _status_value_text(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def status_card_markdown(
    title: str,
    detail: str | None = None,
    items: tuple[tuple[str, object], ...] | list[tuple[str, object]] = (),
) -> str:

    normalized_title = str(title or "Status").strip() or "Status"
    normalized_detail = str(detail or "").strip() or "Ready."
    lines = [f"## {normalized_title}", normalized_detail]
    for label, value in items:
        normalized_label = str(label).strip()
        if not normalized_label:
            continue
        lines.append(f"- **{normalized_label}:** {_status_value_text(value)}")
    return "\n".join(lines)


class StatusCard(QFrame):
    def __init__(
        self,
        title: str = "Status",
        detail: str = "Ready.",
        items: Sequence[tuple[str, object]] = (),
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("DefinersCard")
        self.setFrameShape(QFrame.NoFrame)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(6)
        self._title_label = QLabel(title, self)
        self._title_label.setStyleSheet(
            "color: #eef6ff; font-size: 16px; font-weight: 700;"
            " background: transparent;"
        )
        self._detail_label = QLabel(detail, self)
        self._detail_label.setWordWrap(True)
        self._detail_label.setStyleSheet(
            "color: #7a8fa8; font-size: 14px; background: transparent;"
        )
        self._items_label = QLabel("", self)
        self._items_label.setWordWrap(True)
        self._items_label.setTextFormat(Qt.RichText)
        self._items_label.setStyleSheet(
            "color: #c2cfe0; font-size: 14px; background: transparent;"
        )
        layout.addWidget(self._title_label)
        layout.addWidget(self._detail_label)
        layout.addWidget(self._items_label)
        self.update_card(title, detail, items)

    def update_card(
        self,
        title: str,
        detail: str | None = None,
        items: Sequence[tuple[str, object]] = (),
    ) -> None:
        self._title_label.setText(str(title or "Status"))
        self._detail_label.setText(str(detail or "Ready."))
        rendered_items: list[str] = []
        for label, value in items or ():
            label_text = str(label).strip()
            if not label_text:
                continue
            rendered_items.append(
                f"<div><b>{escape(label_text)}:</b> "
                f"{escape(_status_value_text(value))}</div>"
            )
        if rendered_items:
            self._items_label.setText("".join(rendered_items))
            self._items_label.setVisible(True)
        else:
            self._items_label.setText("")
            self._items_label.setVisible(False)

    @Slot(dict)
    def apply_update(self, payload: dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            return
        self.update_card(
            payload.get("title", self._title_label.text()),
            payload.get("detail", self._detail_label.text()),
            payload.get("items", ()) or (),
        )


def init_status_card(
    title: str = "Status",
    detail: str = "Ready.",
    items: tuple[tuple[str, object], ...] | list[tuple[str, object]] = (),
) -> StatusCard:
    return StatusCard(title=title, detail=detail, items=tuple(items))


def _open_directory(path: str) -> None:
    if sys.platform.startswith("win") and hasattr(os, "startfile"):
        os.startfile(path)
        return
    if sys.platform == "darwin":
        subprocess.Popen(["open", path])
        return
    subprocess.Popen(["xdg-open", path])


def _managed_output_folder_path(section: str | None = None) -> str:
    from definers.system.output_paths import (
        managed_output_dir,
        managed_output_root,
    )

    if section is None or not str(section).strip():
        return managed_output_root()
    return managed_output_dir(section)


def open_managed_output_root(section: str | None = None) -> dict[str, Any]:

    output_root = Path(_managed_output_folder_path(section))
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        _open_directory(str(output_root))
    except Exception as error:
        return {
            "path": str(output_root),
            "detail": f"Could not open the folder automatically: {error}",
            "state": "error",
        }
    return {
        "path": str(output_root),
        "detail": "Opened in the system file manager.",
        "state": "open",
    }


class OutputFolderToolbar(QFrame):
    opened = Signal(dict)

    def __init__(
        self,
        button_label: str = "Open Outputs Folder",
        section: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("DefinersCard")
        self.setFrameShape(QFrame.NoFrame)
        self._section = section
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(12)
        self._button = QPushButton(button_label, self)
        self._button.setMinimumWidth(200)
        self._button.clicked.connect(self._on_clicked)
        layout.addWidget(self._button)

        info = QVBoxLayout()
        info.setContentsMargins(0, 0, 0, 0)
        info.setSpacing(2)
        self._path_label = QLabel(_managed_output_folder_path(section), self)
        self._path_label.setStyleSheet(
            "color: #eef6ff; font-size: 14px; font-weight: 600;"
            " background: transparent;"
        )
        self._path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        detail_text = (
            "Artifacts produced by this GUI are saved here."
            if section is not None and str(section).strip()
            else "Artifacts produced by the GUI are saved here."
        )
        self._detail_label = QLabel(detail_text, self)
        self._detail_label.setStyleSheet(
            "color: #7a8fa8; font-size: 13px; background: transparent;"
        )
        self._detail_label.setWordWrap(True)
        info.addWidget(self._path_label)
        info.addWidget(self._detail_label)
        layout.addLayout(info, stretch=1)

    @property
    def button(self) -> QPushButton:
        return self._button

    @Slot()
    def _on_clicked(self) -> None:
        status = open_managed_output_root(self._section)
        self._path_label.setText(status.get("path", ""))
        self._detail_label.setText(status.get("detail", ""))
        self.opened.emit(status)


def init_output_folder_controls(
    button_label: str = "Open Outputs Folder",
    *,
    section: str | None = None,
) -> tuple[QPushButton, OutputFolderToolbar]:

    toolbar = OutputFolderToolbar(button_label=button_label, section=section)
    return toolbar.button, toolbar


def progress_update(
    title: str,
    state: str = "idle",
    detail: str | None = None,
    steps: tuple[str, ...] | list[str] | None = None,
    active_step: int | None = None,
    activity_completed: int | None = None,
    activity_total: int | None = None,
    bytes_downloaded: int | None = None,
    bytes_total: int | None = None,
) -> dict[str, Any]:

    return {
        "title": title,
        "state": state,
        "detail": detail,
        "steps": tuple(steps or ()),
        "active_step": active_step,
        "activity_completed": activity_completed,
        "activity_total": activity_total,
        "bytes_downloaded": bytes_downloaded,
        "bytes_total": bytes_total,
    }


def _coerce_outputs(outputs: Any) -> list[Any]:
    if outputs is None:
        return []
    if isinstance(outputs, (list, tuple)):
        return list(outputs)
    return [outputs]


def _coerce_result_values(result: Any, output_count: int) -> tuple[Any, ...]:
    if output_count == 0:
        return ()
    if output_count == 1:
        return (result,)
    if not isinstance(result, (list, tuple)):
        raise TypeError("Expected a list or tuple of outputs")
    if len(result) != output_count:
        raise ValueError("Unexpected number of outputs returned")
    return tuple(result)


def _activity_detail(fallback_detail: str, activity_snapshot: Any) -> str:
    if activity_snapshot is None:
        return fallback_detail
    activity_message = str(getattr(activity_snapshot, "message", "")).strip()
    return activity_message or fallback_detail


class StateHolder:
    __slots__ = ("value",)

    def __init__(self, value: Any = None) -> None:
        self.value = value

    def get_value(self) -> Any:
        return self.value

    def set_value(self, value: Any) -> None:
        self.value = value


def read_widget_value(widget: Any) -> Any:

    if widget is None:
        return None
    if isinstance(widget, StateHolder):
        return widget.value
    if hasattr(widget, "get_value"):
        return widget.get_value()
    if hasattr(widget, "value") and callable(widget.value):
        try:
            return widget.value()
        except TypeError:
            pass
    if hasattr(widget, "toPlainText"):
        return widget.toPlainText()
    if hasattr(widget, "currentText"):
        return widget.currentText()
    if hasattr(widget, "isChecked"):
        return widget.isChecked()
    if hasattr(widget, "text"):
        return widget.text()
    return None


def write_widget_value(widget: Any, value: Any) -> None:

    if widget is None:
        return
    if isinstance(widget, StateHolder):
        widget.value = value
        return
    if isinstance(value, dict):
        if hasattr(widget, "apply_update"):
            widget.apply_update(value)
            return
    if hasattr(widget, "set_value"):
        widget.set_value(value)
        return
    if hasattr(widget, "setPlainText"):
        widget.setPlainText("" if value is None else str(value))
        return
    if hasattr(widget, "setCurrentText"):
        widget.setCurrentText("" if value is None else str(value))
        return
    if hasattr(widget, "setChecked"):
        widget.setChecked(bool(value))
        return
    if hasattr(widget, "setValue"):
        try:
            widget.setValue(value)
            return
        except TypeError:
            pass
    if hasattr(widget, "setText"):
        widget.setText("" if value is None else str(value))


class ProgressWorker(QObject):
    progress = Signal(dict)
    result = Signal(tuple)
    error = Signal(object)
    finished = Signal()

    def __init__(
        self,
        handler: Callable[..., Any],
        args: tuple[Any, ...] = (),
        *,
        output_count: int,
        action_label: str,
        steps: Sequence[str] | None = None,
        running_detail: str | None = None,
        success_detail: str | None = None,
        poll_interval_seconds: float = 0.12,
    ) -> None:
        super().__init__()
        self._handler = handler
        self._args = tuple(args)
        self._output_count = output_count
        self._action_label = action_label
        self._steps = tuple(steps or ()) or (
            "Validate request",
            "Run workflow",
            "Publish result",
        )
        self._running_detail = running_detail
        self._success_detail = success_detail
        self._poll = poll_interval_seconds

    @Slot()
    def run(self) -> None:
        from definers.system.download_activity import (
            create_download_activity_task,
            get_download_activity_snapshot,
            resolve_download_activity_task,
            wait_for_download_activity_task,
        )

        try:
            self.progress.emit(
                progress_update(
                    self._action_label,
                    "running",
                    "Checking the request.",
                    steps=self._steps,
                    active_step=1,
                )
            )
            activity_task = create_download_activity_task(
                self._handler, *self._args
            )
            running_step = min(2, len(self._steps))
            self.progress.emit(
                progress_update(
                    self._action_label,
                    "running",
                    self._running_detail or "Working on your request.",
                    steps=self._steps,
                    active_step=running_step,
                )
            )
            last_sequence = -1
            while True:
                task_done = wait_for_download_activity_task(
                    activity_task, self._poll
                )
                snapshot = get_download_activity_snapshot(
                    activity_task.scope_id
                )
                if snapshot is not None and snapshot.sequence != last_sequence:
                    last_sequence = snapshot.sequence
                    self.progress.emit(
                        progress_update(
                            self._action_label,
                            "running",
                            _activity_detail(
                                self._running_detail
                                or "Working on your request.",
                                snapshot,
                            ),
                            steps=self._steps,
                            active_step=running_step,
                            activity_completed=getattr(
                                snapshot, "completed", None
                            ),
                            activity_total=getattr(snapshot, "total", None),
                            bytes_downloaded=getattr(
                                snapshot, "bytes_downloaded", None
                            ),
                            bytes_total=getattr(snapshot, "bytes_total", None),
                        )
                    )
                if task_done:
                    break
            try:
                raw_result, _ = resolve_download_activity_task(activity_task)
            except Exception as error:
                snapshot = getattr(error, "download_activity_snapshot", None)
                self.progress.emit(
                    progress_update(
                        self._action_label,
                        "error",
                        _activity_detail(str(error), snapshot),
                        steps=self._steps,
                        active_step=running_step,
                        activity_completed=getattr(snapshot, "completed", None),
                        activity_total=getattr(snapshot, "total", None),
                        bytes_downloaded=getattr(
                            snapshot, "bytes_downloaded", None
                        ),
                        bytes_total=getattr(snapshot, "bytes_total", None),
                    )
                )
                self.error.emit(error)
                return
            try:
                values = _coerce_result_values(raw_result, self._output_count)
            except Exception as conversion_error:
                self.progress.emit(
                    progress_update(
                        self._action_label,
                        "error",
                        str(conversion_error),
                        steps=self._steps,
                        active_step=running_step,
                    )
                )
                self.error.emit(conversion_error)
                return
            if len(self._steps) > 2:
                self.progress.emit(
                    progress_update(
                        self._action_label,
                        "running",
                        "Publishing the result.",
                        steps=self._steps,
                        active_step=len(self._steps),
                    )
                )
            self.progress.emit(
                progress_update(
                    self._action_label,
                    "success",
                    self._success_detail or "Result is ready.",
                    steps=self._steps,
                    active_step=len(self._steps),
                )
            )
            self.result.emit(values)
        except Exception as unexpected:
            self.progress.emit(
                progress_update(
                    self._action_label,
                    "error",
                    str(unexpected),
                    steps=self._steps,
                )
            )
            self.error.emit(unexpected)
        finally:
            self.finished.emit()


@dataclass
class _ActiveBinding:
    thread: QThread
    worker: ProgressWorker


def bind_progress_click(
    button: QPushButton,
    handler: Callable[..., Any],
    *,
    progress_output: ProgressTracker,
    inputs: Sequence[Any] | None = None,
    outputs: Sequence[Any] | None = None,
    action_label: str,
    steps: Sequence[str] | None = None,
    running_detail: str | None = None,
    success_detail: str | None = None,
) -> Callable[[], None]:

    resolved_inputs = list(inputs or ())
    resolved_outputs = list(outputs or ())
    output_count = len(resolved_outputs)
    binding: dict[str, _ActiveBinding | None] = {"current": None}

    def _on_progress(payload: dict[str, Any]) -> None:
        if progress_output is not None:
            progress_output.apply_update(payload)

    def _on_result(values: tuple[Any, ...]) -> None:
        for widget, value in zip(resolved_outputs, values):
            write_widget_value(widget, value)

    def _on_error(exc: BaseException) -> None:
        return

    def _on_finished() -> None:
        button.setEnabled(True)
        active = binding["current"]
        if active is not None:
            active.thread.quit()
            active.thread.wait()
            active.worker.deleteLater()
            active.thread.deleteLater()
        binding["current"] = None

    def _trigger() -> None:
        if binding["current"] is not None:
            return
        args = tuple(read_widget_value(widget) for widget in resolved_inputs)
        thread = QThread()
        worker = ProgressWorker(
            handler,
            args,
            output_count=output_count,
            action_label=action_label,
            steps=steps,
            running_detail=running_detail,
            success_detail=success_detail,
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(_on_progress)
        worker.result.connect(_on_result)
        worker.error.connect(_on_error)
        worker.finished.connect(_on_finished)
        binding["current"] = _ActiveBinding(thread=thread, worker=worker)
        button.setEnabled(False)
        thread.start()

    button.clicked.connect(_trigger)
    return _trigger


@dataclass
class ChatMessage:
    role: str
    content: str


class _ChatModel(QAbstractListModel):
    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._messages: list[ChatMessage] = []

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return len(self._messages)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:
        if not index.isValid() or not (0 <= index.row() < len(self._messages)):
            return None
        message = self._messages[index.row()]
        if role == Qt.DisplayRole:
            return f"{message.role.capitalize()}: {message.content}"
        if role == Qt.UserRole:
            return message
        return None

    def append(self, message: ChatMessage) -> None:
        row = len(self._messages)
        self.beginInsertRows(QModelIndex(), row, row)
        self._messages.append(message)
        self.endInsertRows()

    def replace_last(self, content: str) -> None:
        if not self._messages:
            return
        row = len(self._messages) - 1
        self._messages[row] = ChatMessage(
            role=self._messages[row].role,
            content=content,
        )
        index = self.index(row, 0)
        self.dataChanged.emit(index, index, [Qt.DisplayRole])

    def messages(self) -> list[ChatMessage]:
        return list(self._messages)


class _ChatDelegate(QStyledItemDelegate):
    def sizeHint(self, option, index):
        size = super().sizeHint(option, index)
        return QSize(size.width(), max(size.height() + 16, 56))


class _ChatStreamWorker(QObject):
    chunk = Signal(str)
    finished = Signal()
    failed = Signal(object)

    def __init__(
        self,
        stream_handler: Callable[[str, list[tuple[str, str]]], Iterable[str]],
        message: str,
        history: list[tuple[str, str]],
    ) -> None:
        super().__init__()
        self._stream_handler = stream_handler
        self._message = message
        self._history = history

    @Slot()
    def run(self) -> None:
        try:
            iterator = self._stream_handler(self._message, self._history)
            for partial in iterator:
                self.chunk.emit("" if partial is None else str(partial))
        except Exception as error:
            self.failed.emit(error)
        finally:
            self.finished.emit()


class ChatPanel(QFrame):
    def __init__(
        self,
        title: str,
        stream_handler: Callable[[str, list[tuple[str, str]]], Iterable[str]],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("DefinersCard")
        self.setFrameShape(QFrame.NoFrame)
        self._stream_handler = stream_handler
        self._active_thread: QThread | None = None
        self._active_worker: _ChatStreamWorker | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)

        title_label = QLabel(title, self)
        title_label.setStyleSheet(
            "color: #eef6ff; font-size: 18px; font-weight: 700;"
            " letter-spacing: 0.2px; background: transparent;"
        )
        layout.addWidget(title_label)

        self._model = _ChatModel(self)
        self._view = QListView(self)
        self._view.setModel(self._model)
        self._view.setItemDelegate(_ChatDelegate(self._view))
        self._view.setWordWrap(True)
        self._view.setUniformItemSizes(False)
        self._view.setSelectionMode(QListView.NoSelection)
        layout.addWidget(self._view, stretch=1)

        input_row = QHBoxLayout()
        input_row.setContentsMargins(0, 0, 0, 0)
        input_row.setSpacing(8)
        self._input = QTextEdit(self)
        self._input.setPlaceholderText("Type a message and press Enter…")
        self._input.setFixedHeight(82)
        self._input.setWordWrapMode(QTextOption.WrapAtWordBoundaryOrAnywhere)
        input_row.addWidget(self._input, stretch=1)

        button_column = QVBoxLayout()
        button_column.setSpacing(6)
        self._send_button = QPushButton("Send", self)
        self._send_button.setProperty("definersPrimary", True)
        self._send_button.clicked.connect(self._on_send)
        self._clear_button = QPushButton("Clear", self)
        self._clear_button.clicked.connect(self._on_clear)
        button_column.addWidget(self._send_button)
        button_column.addWidget(self._clear_button)
        button_column.addStretch(1)
        input_row.addLayout(button_column)
        layout.addLayout(input_row)

    @Slot()
    def _on_clear(self) -> None:
        if self._active_thread is not None:
            return
        self._model = _ChatModel(self)
        self._view.setModel(self._model)

    @Slot()
    def _on_send(self) -> None:
        if self._active_thread is not None:
            return
        text = self._input.toPlainText().strip()
        if not text:
            return
        self._input.clear()
        self._model.append(ChatMessage(role="user", content=text))
        self._model.append(ChatMessage(role="assistant", content="…"))
        history = [
            (message.role, message.content)
            for message in self._model.messages()[:-2]
        ]
        self._send_button.setEnabled(False)
        thread = QThread()
        worker = _ChatStreamWorker(
            self._stream_handler,
            text,
            history,
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.chunk.connect(self._on_chunk)
        worker.failed.connect(self._on_failure)
        worker.finished.connect(self._on_stream_finished)
        self._active_thread = thread
        self._active_worker = worker
        thread.start()

    @Slot(str)
    def _on_chunk(self, partial: str) -> None:
        self._model.replace_last(partial or "…")
        self._view.scrollToBottom()

    @Slot(object)
    def _on_failure(self, error: BaseException) -> None:
        self._model.replace_last(f"⚠ Chat failure: {error}")

    @Slot()
    def _on_stream_finished(self) -> None:
        if self._active_thread is not None:
            self._active_thread.quit()
            self._active_thread.wait()
        if self._active_worker is not None:
            self._active_worker.deleteLater()
        if self._active_thread is not None:
            self._active_thread.deleteLater()
        self._active_thread = None
        self._active_worker = None
        self._send_button.setEnabled(True)
        self._view.scrollToBottom()


def init_chat(
    title: str,
    stream_handler: Callable[[str, list[tuple[str, str]]], Iterable[str]],
) -> ChatPanel:

    return ChatPanel(title=title, stream_handler=stream_handler)


def launch_blocks(
    window: QWidget | QMainWindow,
    *,
    custom_css: str | None = None,
) -> int:

    application = qt_app()
    if custom_css:
        existing = application.styleSheet() or ""
        application.setStyleSheet(existing + "\n" + custom_css)
    window.show()
    return int(application.exec())
