from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from PySide6.QtCore import QEvent, QSize, Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class MarkdownLabel(QLabel):
    def __init__(self, text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setTextFormat(Qt.MarkdownText)
        self.setWordWrap(True)
        self.setText(text)


class HtmlLabel(QLabel):
    def __init__(self, html: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setTextFormat(Qt.RichText)
        self.setWordWrap(True)
        self.setText(html)


class HeroCard(HtmlLabel):
    def __init__(self, html: str = "", parent: QWidget | None = None) -> None:
        super().__init__(html, parent)
        self.setObjectName("DefinersHero")


class DividerLabel(MarkdownLabel):
    def __init__(self, text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(text, parent)


class Section(QFrame):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.NoFrame)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        self._layout = layout

    def addWidget(self, widget: QWidget) -> None:
        self._layout.addWidget(widget)

    def addLayout(self, layout) -> None:
        self._layout.addLayout(layout)

    def layout_target(self) -> QVBoxLayout:
        return self._layout


class CardFrame(Section):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("DefinersCard")


class HeroSection(Section):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("DefinersHero")
        self._layout.setContentsMargins(22, 22, 22, 22)
        self._layout.setSpacing(10)


class Row(QFrame):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.NoFrame)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        self._layout = layout

    def addWidget(self, widget: QWidget) -> None:
        self._layout.addWidget(widget, 0)

    def addLayout(self, layout) -> None:
        self._layout.addLayout(layout, 0)


class Column(Section):
    pass


class Accordion(QGroupBox):
    def __init__(
        self,
        title: str,
        *,
        open: bool = True,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(title, parent)
        self.setCheckable(True)
        self.setChecked(bool(open))
        body = QVBoxLayout(self)
        body.setContentsMargins(10, 14, 10, 10)
        body.setSpacing(8)
        self._body = body
        self.toggled.connect(self._on_toggled)

    def _on_toggled(self, checked: bool) -> None:
        for i in range(self._body.count()):
            item = self._body.itemAt(i)
            widget = item.widget()
            if widget is not None:
                widget.setVisible(checked)

    def addWidget(self, widget: QWidget) -> None:
        self._body.addWidget(widget)
        widget.setVisible(self.isChecked())

    def addLayout(self, layout) -> None:
        self._body.addLayout(layout)


class LabeledWidget(QFrame):
    def __init__(
        self,
        label: str,
        *,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.NoFrame)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)
        self._label = QLabel(label, self)
        self._label.setStyleSheet(
            "color: #7a8fa8;"
            " font-size: 11px;"
            " font-weight: 700;"
            " letter-spacing: 1.5px;"
            " background: transparent;"
        )
        outer.addWidget(self._label)
        self._outer = outer
        self._label_text = label

    def set_label(self, text: str) -> None:
        self._label_text = text
        self._label.setText(text)


class LabeledTextbox(LabeledWidget):
    def __init__(
        self,
        label: str,
        *,
        value: str = "",
        placeholder: str = "",
        lines: int = 4,
        max_length: int | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(label, parent=parent)
        self._editor = QTextEdit(self)
        self._editor.setPlaceholderText(placeholder)
        self._editor.setPlainText(value)
        self._editor.setAcceptRichText(False)
        fm_height = self._editor.fontMetrics().lineSpacing()
        self._editor.setMinimumHeight(int(fm_height * max(int(lines), 1) + 16))
        self._max_length = max_length
        if max_length is not None:
            self._editor.textChanged.connect(self._enforce_max_length)
        self._outer.addWidget(self._editor)

    def _enforce_max_length(self) -> None:
        if self._max_length is None:
            return
        current = self._editor.toPlainText()
        if len(current) > self._max_length:
            cursor = self._editor.textCursor()
            position = cursor.position()
            self._editor.blockSignals(True)
            self._editor.setPlainText(current[: self._max_length])
            cursor.setPosition(min(position, self._max_length))
            self._editor.setTextCursor(cursor)
            self._editor.blockSignals(False)

    def value(self) -> str:
        return self._editor.toPlainText()

    def set_value(self, value: Any) -> None:
        self._editor.setPlainText("" if value is None else str(value))

    def apply_update(self, update: dict[str, Any]) -> None:
        if "value" in update:
            self.set_value(update["value"])
        if "visible" in update:
            self.setVisible(bool(update["visible"]))
        if "interactive" in update:
            self._editor.setReadOnly(not bool(update["interactive"]))
        if "label" in update:
            self.set_label(str(update["label"]))


class LabeledLineEdit(LabeledWidget):
    def __init__(
        self,
        label: str,
        *,
        value: str = "",
        placeholder: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(label, parent=parent)
        self._editor = QLineEdit(self)
        self._editor.setPlaceholderText(placeholder)
        self._editor.setText(value)
        self._outer.addWidget(self._editor)

    def value(self) -> str:
        return self._editor.text()

    def set_value(self, value: Any) -> None:
        self._editor.setText("" if value is None else str(value))

    def apply_update(self, update: dict[str, Any]) -> None:
        if "value" in update:
            self.set_value(update["value"])
        if "visible" in update:
            self.setVisible(bool(update["visible"]))
        if "interactive" in update:
            self._editor.setReadOnly(not bool(update["interactive"]))
        if "label" in update:
            self.set_label(str(update["label"]))


class LabeledCombo(LabeledWidget):
    changed = Signal(str)

    def __init__(
        self,
        label: str,
        *,
        choices: Iterable[str] = (),
        value: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(label, parent=parent)
        self._combo = QComboBox(self)
        self._combo.addItems([str(c) for c in choices])
        if value is not None and self._combo.findText(str(value)) >= 0:
            self._combo.setCurrentText(str(value))
        elif value is not None:
            pass
        self._combo.currentTextChanged.connect(self.changed.emit)
        self._outer.addWidget(self._combo)

    def value(self) -> str:
        return self._combo.currentText()

    def set_value(self, value: Any) -> None:
        if value is None:
            return
        self._combo.setCurrentText(str(value))

    def set_choices(self, choices: Iterable[str]) -> None:
        current = self._combo.currentText()
        self._combo.blockSignals(True)
        self._combo.clear()
        self._combo.addItems([str(c) for c in choices])
        if current and self._combo.findText(current) >= 0:
            self._combo.setCurrentText(current)
        self._combo.blockSignals(False)

    def apply_update(self, update: dict[str, Any]) -> None:
        if "choices" in update:
            self.set_choices(update["choices"] or [])
        if "value" in update:
            self.set_value(update["value"])
        if "visible" in update:
            self.setVisible(bool(update["visible"]))
        if "interactive" in update:
            self._combo.setEnabled(bool(update["interactive"]))
        if "label" in update:
            self.set_label(str(update["label"]))


class LabeledSlider(LabeledWidget):
    def __init__(
        self,
        label: str,
        *,
        minimum: int = 0,
        maximum: int = 100,
        value: int = 0,
        step: int = 1,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(label, parent=parent)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        self._slider = QSlider(Qt.Horizontal, self)
        self._slider.setMinimum(int(minimum))
        self._slider.setMaximum(int(maximum))
        self._slider.setSingleStep(int(step) or 1)
        self._slider.setPageStep(max(int(step) or 1, 1))
        self._slider.setValue(int(value))
        self._readout = QLabel(str(int(value)), self)
        self._readout.setMinimumWidth(40)
        self._readout.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._readout.setStyleSheet(
            "color: #54e2ff; font-size: 13px; font-weight: 600;"
            " background: transparent;"
        )
        self._slider.valueChanged.connect(
            lambda v: self._readout.setText(str(int(v)))
        )
        row.addWidget(self._slider, 1)
        row.addWidget(self._readout, 0)
        self._outer.addLayout(row)

    def value(self) -> int:
        return int(self._slider.value())

    def set_value(self, value: Any) -> None:
        try:
            self._slider.setValue(int(float(value)))
        except (TypeError, ValueError):
            pass

    def apply_update(self, update: dict[str, Any]) -> None:
        if "value" in update:
            self.set_value(update["value"])
        if "visible" in update:
            self.setVisible(bool(update["visible"]))
        if "interactive" in update:
            self._slider.setEnabled(bool(update["interactive"]))
        if "label" in update:
            self.set_label(str(update["label"]))


class LabeledFloatSlider(LabeledWidget):
    def __init__(
        self,
        label: str,
        *,
        minimum: float = 0.0,
        maximum: float = 1.0,
        value: float = 0.0,
        step: float = 0.01,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(label, parent=parent)
        self._step = float(step) if step else 0.01
        if self._step <= 0:
            self._step = 0.01
        self._minimum = float(minimum)
        self._maximum = float(maximum)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        self._slider = QSlider(Qt.Horizontal, self)
        scale = self._scale_factor()
        self._slider.setMinimum(int(round(self._minimum * scale)))
        self._slider.setMaximum(int(round(self._maximum * scale)))
        self._slider.setSingleStep(1)
        self._slider.setValue(int(round(float(value) * scale)))
        self._readout = QLabel(self._format(float(value)), self)
        self._readout.setMinimumWidth(60)
        self._readout.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._readout.setStyleSheet(
            "color: #54e2ff; font-size: 13px; font-weight: 600;"
            " background: transparent;"
        )
        self._slider.valueChanged.connect(self._on_changed)
        row.addWidget(self._slider, 1)
        row.addWidget(self._readout, 0)
        self._outer.addLayout(row)

    def _scale_factor(self) -> float:
        return 1.0 / self._step

    def _on_changed(self, raw: int) -> None:
        self._readout.setText(self._format(raw / self._scale_factor()))

    def _format(self, value: float) -> str:
        return f"{value:.3f}".rstrip("0").rstrip(".") or "0"

    def value(self) -> float:
        return float(self._slider.value()) / self._scale_factor()

    def set_value(self, value: Any) -> None:
        try:
            self._slider.setValue(
                int(round(float(value) * self._scale_factor()))
            )
        except (TypeError, ValueError):
            pass

    def apply_update(self, update: dict[str, Any]) -> None:
        if "value" in update:
            self.set_value(update["value"])
        if "visible" in update:
            self.setVisible(bool(update["visible"]))
        if "interactive" in update:
            self._slider.setEnabled(bool(update["interactive"]))
        if "label" in update:
            self.set_label(str(update["label"]))


class LabeledSpin(LabeledWidget):
    def __init__(
        self,
        label: str,
        *,
        minimum: int = -1_000_000_000,
        maximum: int = 1_000_000_000,
        value: int = 0,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(label, parent=parent)
        self._spin = QSpinBox(self)
        self._spin.setMinimum(int(minimum))
        self._spin.setMaximum(int(maximum))
        self._spin.setValue(int(value))
        self._outer.addWidget(self._spin)

    def value(self) -> int:
        return int(self._spin.value())

    def set_value(self, value: Any) -> None:
        try:
            self._spin.setValue(int(float(value)))
        except (TypeError, ValueError):
            pass

    def apply_update(self, update: dict[str, Any]) -> None:
        if "value" in update:
            self.set_value(update["value"])
        if "visible" in update:
            self.setVisible(bool(update["visible"]))
        if "interactive" in update:
            self._spin.setEnabled(bool(update["interactive"]))
        if "label" in update:
            self.set_label(str(update["label"]))


class LabeledFloatSpin(LabeledWidget):
    def __init__(
        self,
        label: str,
        *,
        minimum: float = -1.0e9,
        maximum: float = 1.0e9,
        value: float = 0.0,
        decimals: int = 3,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(label, parent=parent)
        self._spin = QDoubleSpinBox(self)
        self._spin.setRange(float(minimum), float(maximum))
        self._spin.setDecimals(int(decimals))
        self._spin.setValue(float(value))
        self._outer.addWidget(self._spin)

    def value(self) -> float:
        return float(self._spin.value())

    def set_value(self, value: Any) -> None:
        try:
            self._spin.setValue(float(value))
        except (TypeError, ValueError):
            pass

    def apply_update(self, update: dict[str, Any]) -> None:
        if "value" in update:
            self.set_value(update["value"])
        if "visible" in update:
            self.setVisible(bool(update["visible"]))
        if "interactive" in update:
            self._spin.setEnabled(bool(update["interactive"]))
        if "label" in update:
            self.set_label(str(update["label"]))


class LabeledCheckbox(QCheckBox):
    def __init__(
        self,
        label: str,
        *,
        value: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(label, parent)
        self.setChecked(bool(value))

    def value(self) -> bool:
        return bool(self.isChecked())

    def set_value(self, value: Any) -> None:
        self.setChecked(bool(value))

    def apply_update(self, update: dict[str, Any]) -> None:
        if "value" in update:
            self.set_value(update["value"])
        if "visible" in update:
            self.setVisible(bool(update["visible"]))
        if "interactive" in update:
            self.setEnabled(bool(update["interactive"]))
        if "label" in update:
            self.setText(str(update["label"]))


class LabeledCheckboxGroup(LabeledWidget):
    def __init__(
        self,
        label: str,
        *,
        choices: Iterable[str] = (),
        value: Sequence[str] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(label, parent=parent)
        self._boxes: list[QCheckBox] = []
        selected = set(value or ())
        for choice in choices:
            box = QCheckBox(str(choice), self)
            if str(choice) in selected:
                box.setChecked(True)
            self._outer.addWidget(box)
            self._boxes.append(box)

    def value(self) -> list[str]:
        return [box.text() for box in self._boxes if box.isChecked()]

    def set_value(self, value: Any) -> None:
        if value is None:
            value_set: set[str] = set()
        else:
            value_set = {str(v) for v in value}
        for box in self._boxes:
            box.setChecked(box.text() in value_set)

    def apply_update(self, update: dict[str, Any]) -> None:
        if "value" in update:
            self.set_value(update["value"])
        if "visible" in update:
            self.setVisible(bool(update["visible"]))
        if "interactive" in update:
            for box in self._boxes:
                box.setEnabled(bool(update["interactive"]))
        if "label" in update:
            self.set_label(str(update["label"]))


class FilePicker(LabeledWidget):
    selected = Signal(list)

    def __init__(
        self,
        label: str,
        *,
        button_text: str = "Choose File...",
        file_filter: str = "All Files (*)",
        max_files: int | None = 1,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(label, parent=parent)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)

        self._button = QPushButton(button_text, self)
        self._button.clicked.connect(self._open_dialog)

        self._path_label = QLabel("No file selected.", self)
        self._path_label.setWordWrap(True)
        self._path_label.setStyleSheet(
            "color: #4e6278; font-size: 13px; background: transparent;"
        )

        row.addWidget(self._button, 0)
        row.addWidget(self._path_label, 1)
        self._outer.addLayout(row)

        self._file_filter = file_filter
        self._max_files = max_files

        self._paths: str | list[str] = []

    def _open_dialog(self) -> None:
        if self._max_files == 1:
            path, _ = QFileDialog.getOpenFileName(
                self,
                self._label_text or "Select File",
                "",
                self._file_filter,
            )
            paths = path
        else:
            paths, _ = QFileDialog.getOpenFileNames(
                self,
                self._label_text or "Select Files",
                "",
                self._file_filter,
            )

            if self._max_files is not None and len(paths) > self._max_files:
                paths = paths[: self._max_files]

        if paths:
            self._set_paths(paths)
            self.selected.emit(paths)

    def _set_paths(self, paths: str | list[str]) -> None:
        self._paths = paths
        if paths:
            if self._max_files == 1:
                self._path_label.setText(Path(paths).name)
                self._path_label.setToolTip(paths)
            elif len(paths) == 1:
                self._path_label.setText(Path(paths[0]).name)
                self._path_label.setToolTip(paths[0])
            else:
                self._path_label.setText(f"{len(paths)} files selected")
                self._path_label.setToolTip("\n".join(paths))
        else:
            self._path_label.setText(
                "No file selected."
                if self._max_files == 1
                else "No files selected."
            )
            self._path_label.setToolTip("")

    def value(self) -> str | list[str]:
        return self._paths

    def set_value(self, value: Any) -> None:
        if value is None:
            self._set_paths([])
        elif isinstance(value, list):
            self._set_paths([str(v) for v in value])
        else:
            self._set_paths([str(value)])

    def apply_update(self, update: dict[str, Any]) -> None:
        if "value" in update:
            self.set_value(update["value"])
        if "visible" in update:
            self.setVisible(bool(update["visible"]))
        if "interactive" in update:
            self._button.setEnabled(bool(update["interactive"]))
        if "label" in update:
            self.set_label(str(update["label"]))
        if "max_files" in update:
            self._max_files = int(update["max_files"])


class ImageInput(LabeledWidget):
    selected = Signal(str)

    def __init__(
        self,
        label: str,
        *,
        height: int = 220,
        file_filter: str = "Images (*.png *.jpg *.jpeg *.webp *.gif)",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(label, parent=parent)
        self._preview = QLabel(self)
        self._preview.setAlignment(Qt.AlignCenter)
        self._preview.setMinimumHeight(int(height))
        self._preview.setFrameShape(QFrame.Box)
        self._preview.setStyleSheet(
            "background-color: #17223a;"
            " border: 1px solid rgba(84,226,255,0.14);"
            " border-radius: 8px;"
            " color: #6a80a0;"
            " font-size: 13px;"
        )
        self._preview.setText("(no image)")
        self._button = QPushButton("Choose Image...", self)
        self._button.clicked.connect(self._open_dialog)
        self._outer.addWidget(self._preview)
        self._outer.addWidget(self._button)
        self._file_filter = file_filter
        self._path: str | None = None

    def _open_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            self._label_text or "Select Image",
            "",
            self._file_filter,
        )
        if path:
            self.set_value(path)
            self.selected.emit(path)

    def value(self) -> str | None:
        return self._path

    def set_value(self, value: Any) -> None:
        if value is None:
            self._path = None
            self._preview.setPixmap(QPixmap())
            self._preview.setText("(no image)")
            return
        path = str(value)
        self._path = path
        pixmap = QPixmap(path)
        if pixmap.isNull():
            self._preview.setPixmap(QPixmap())
            self._preview.setText(Path(path).name)
        else:
            self._preview.setPixmap(
                pixmap.scaled(
                    self._preview.width() or 320,
                    self._preview.height() or 220,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )
            self._preview.setText("")

    def apply_update(self, update: dict[str, Any]) -> None:
        if "value" in update:
            self.set_value(update["value"])
        if "visible" in update:
            self.setVisible(bool(update["visible"]))
        if "interactive" in update:
            self._button.setEnabled(bool(update["interactive"]))
        if "label" in update:
            self.set_label(str(update["label"]))


class ImageOutput(LabeledWidget):
    def __init__(
        self,
        label: str,
        *,
        height: int = 220,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(label, parent=parent)
        self._preview = QLabel(self)
        self._preview.setAlignment(Qt.AlignCenter)
        self._preview.setMinimumHeight(int(height))
        self._preview.setFrameShape(QFrame.Box)
        self._preview.setStyleSheet(
            "background-color: #17223a;"
            " border: 1px solid rgba(84,226,255,0.14);"
            " border-radius: 8px;"
            " color: #6a80a0;"
            " font-size: 13px;"
        )
        self._preview.setText("(no image)")
        self._outer.addWidget(self._preview)
        self._path: str | None = None

    def value(self) -> str | None:
        return self._path

    def set_value(self, value: Any) -> None:
        if value is None:
            self._path = None
            self._preview.setPixmap(QPixmap())
            self._preview.setText("(no image)")
            return
        path = str(value)
        self._path = path
        pixmap = QPixmap(path)
        if pixmap.isNull():
            self._preview.setPixmap(QPixmap())
            self._preview.setText(Path(path).name)
        else:
            self._preview.setPixmap(
                pixmap.scaled(
                    self._preview.width() or 320,
                    self._preview.height() or 220,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )
            self._preview.setText("")

    def apply_update(self, update: dict[str, Any]) -> None:
        if "value" in update:
            self.set_value(update["value"])
        if "visible" in update:
            self.setVisible(bool(update["visible"]))
        if "label" in update:
            self.set_label(str(update["label"]))


class MediaOutput(LabeledWidget):
    def __init__(
        self,
        label: str,
        *,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(label, parent=parent)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        self._path_label = QLabel("(no output)", self)
        self._path_label.setWordWrap(True)
        self._path_label.setStyleSheet(
            "color: #4e6278; font-size: 13px; background: transparent;"
        )
        self._open_button = QPushButton("Open", self)
        self._open_button.setEnabled(False)
        self._open_button.clicked.connect(self._open_path)
        row.addWidget(self._path_label, 1)
        row.addWidget(self._open_button, 0)
        self._outer.addLayout(row)
        self._path: str | None = None

    def _open_path(self) -> None:
        if not self._path:
            return
        from definers.ui.gui_shared import _open_directory

        target = Path(self._path)
        directory = str(target.parent) if target.is_file() else str(target)
        _open_directory(directory)

    def value(self) -> str | None:
        return self._path

    def set_value(self, value: Any) -> None:
        if value is None:
            self._path = None
            self._path_label.setText("(no output)")
            self._path_label.setToolTip("")
            self._open_button.setEnabled(False)
            return
        path = str(value)
        self._path = path
        self._path_label.setText(Path(path).name)
        self._path_label.setToolTip(path)
        self._open_button.setEnabled(True)

    def apply_update(self, update: dict[str, Any]) -> None:
        if "value" in update:
            self.set_value(update["value"])
        if "visible" in update:
            self.setVisible(bool(update["visible"]))
        if "label" in update:
            self.set_label(str(update["label"]))


VideoOutput = MediaOutput
AudioOutput = MediaOutput


class FileOutput(MediaOutput):
    pass


def build_main_window(
    title: str,
    body: QWidget,
    *,
    minimum_width: int = 1100,
    minimum_height: int = 720,
) -> QMainWindow:

    window = QMainWindow()
    window.setWindowTitle(title)
    window.resize(int(minimum_width), int(minimum_height))
    window.setWindowOpacity(0.94)
    scroll = QScrollArea(window)
    scroll.setWidget(body)
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.NoFrame)
    if isinstance(body, Section):
        body.layout_target().setContentsMargins(36, 30, 36, 36)
        body.layout_target().setSpacing(22)
    body.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
    window.setCentralWidget(scroll)
    return window


class _AdaptiveTabWidget(QTabWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.currentChanged.connect(self._on_tab_changed)

    def _on_tab_changed(self) -> None:
        self._propagate_geometry_change()

    def _propagate_geometry_change(self) -> None:
        self.updateGeometry()
        parent = self.parentWidget()
        while parent is not None:
            parent.updateGeometry()
            if isinstance(parent, QScrollArea):
                host = parent.widget()
                if host is not None:
                    host.adjustSize()
                break
            parent = parent.parentWidget()

    def event(self, e: QEvent) -> bool:
        result = super().event(e)

        if e.type() == QEvent.Type.LayoutRequest:
            self._propagate_geometry_change()
        return result

    def _bar_extent(self) -> int:
        if self.tabPosition() in (QTabWidget.West, QTabWidget.East):
            return self.tabBar().sizeHint().width()
        return self.tabBar().sizeHint().height()

    def _content_width(self, width: int) -> int:
        margins = self.contentsMargins()
        usable_width = max(int(width) - margins.left() - margins.right(), 0)
        if self.tabPosition() in (QTabWidget.West, QTabWidget.East):
            usable_width = max(usable_width - self._bar_extent(), 0)
        return usable_width

    def _page_size(self, hint_fn: str) -> QSize:
        page = self.currentWidget()
        if page is None:
            return getattr(super(), hint_fn)()
        page_layout = page.layout()
        if page_layout is not None:
            page_layout.activate()
        page_hint: QSize = getattr(page, hint_fn)()
        margins = self.contentsMargins()
        extra_h = margins.top() + margins.bottom()
        if self.tabPosition() in (QTabWidget.North, QTabWidget.South):
            extra_h += self._bar_extent()
        min_w = self.tabBar().minimumSizeHint().width()
        width = max(page_hint.width(), min_w)
        if self.tabPosition() in (QTabWidget.West, QTabWidget.East):
            width += self._bar_extent()
        return QSize(width, page_hint.height() + extra_h)

    def hasHeightForWidth(self) -> bool:
        page = self.currentWidget()
        if page is not None and page.hasHeightForWidth():
            return True
        return super().hasHeightForWidth()

    def heightForWidth(self, width: int) -> int:
        page = self.currentWidget()
        if page is None:
            return super().heightForWidth(width)
        page_layout = page.layout()
        if page_layout is not None:
            page_layout.activate()
        content_width = self._content_width(width)
        if page.hasHeightForWidth():
            page_height = page.heightForWidth(content_width)
        else:
            page_height = page.sizeHint().height()
        margins = self.contentsMargins()
        total_height = page_height + margins.top() + margins.bottom()
        if self.tabPosition() in (QTabWidget.North, QTabWidget.South):
            total_height += self._bar_extent()
        return max(total_height, 0)

    def sizeHint(self) -> QSize:
        return self._page_size("sizeHint")

    def minimumSizeHint(self) -> QSize:
        return self._page_size("minimumSizeHint")


def build_tab_widget(parent: QWidget | None = None) -> QTabWidget:

    widget = _AdaptiveTabWidget(parent)
    widget.setDocumentMode(True)
    return widget


__all__ = [
    "Accordion",
    "AudioOutput",
    "CardFrame",
    "Column",
    "DividerLabel",
    "FileOutput",
    "FilePicker",
    "HeroCard",
    "HeroSection",
    "HtmlLabel",
    "ImageInput",
    "ImageOutput",
    "LabeledCheckbox",
    "LabeledCheckboxGroup",
    "LabeledCombo",
    "LabeledFloatSlider",
    "LabeledFloatSpin",
    "LabeledLineEdit",
    "LabeledSlider",
    "LabeledSpin",
    "LabeledTextbox",
    "MarkdownLabel",
    "MediaOutput",
    "Row",
    "Section",
    "VideoOutput",
    "build_main_window",
    "build_tab_widget",
]
