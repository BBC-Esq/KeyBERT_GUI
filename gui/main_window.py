from pathlib import Path
import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QDoubleSpinBox, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton,
    QProgressBar, QSpinBox, QTextEdit, QVBoxLayout, QWidget, QMainWindow
)

from processing.keybert import KeyBERTProcessor, KeywordExtractorWorker
from processing.extraction import TextExtractorWorker
from resources import TOOLTIPS
from utils import (
    is_nvidia_gpu_available,
    set_cuda_paths,
    validate_keyword_params,
)

# Helper – configure CUDA PATH early if an NVIDIA GPU is detected
if is_nvidia_gpu_available():
    set_cuda_paths()

class MainWindow(QMainWindow):
    """Qt main window wrapping all widgets and event logic."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("KeyBERT GUI — Multi-Format Keyword Extractor")

        self.processor = KeyBERTProcessor()

        self.text_worker: TextExtractorWorker | None = None
        self.keyword_worker: KeywordExtractorWorker | None = None

        self._build_ui()
        self._refresh_model_status()

    # UI Construction
    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        top_bar = QHBoxLayout()
        self.lbl_cuda = QLabel(
            f"CUDA: {'Available' if is_nvidia_gpu_available() else 'Not available'}"
        )
        self.lbl_cuda.setStyleSheet("color: blue;")
        self.lbl_cuda.setToolTip(TOOLTIPS["cuda_status"])
        self.lbl_model = QLabel()
        self.lbl_model.setStyleSheet("color: green;")
        self.lbl_model.setToolTip(TOOLTIPS["model_status"])

        top_bar.addWidget(self.lbl_cuda, alignment=Qt.AlignLeft)
        top_bar.addWidget(self.lbl_model, alignment=Qt.AlignRight)
        root.addLayout(top_bar)

        input_box = QVBoxLayout()
        button_row = QHBoxLayout()
        self.btn_load = QPushButton("Load File (TXT·PDF·DOCX)")
        self.btn_load.setToolTip(TOOLTIPS["load_file_button"])
        self.btn_load.clicked.connect(self._on_load_clicked)
        self.btn_select_model = QPushButton("Select custom model dir")
        self.btn_select_model.setToolTip(TOOLTIPS["custom_model_button"])
        self.btn_select_model.clicked.connect(self._on_model_select)

        button_row.addWidget(self.btn_load)
        button_row.addWidget(self.btn_select_model)
        input_box.addLayout(button_row)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        input_box.addWidget(self.progress)

        self.txt_input = QTextEdit()
        self.txt_input.setToolTip(TOOLTIPS["text_area"])
        input_box.addWidget(self.txt_input)

        root.addLayout(input_box)

        self._build_param_group(root)

        self.btn_extract = QPushButton("Extract keywords")
        self.btn_extract.setStyleSheet(
            "background-color: lightblue; font-weight: bold;"
        )
        self.btn_extract.setToolTip(TOOLTIPS["extract_button"])
        self.btn_extract.clicked.connect(self._on_extract_clicked)
        root.addWidget(self.btn_extract)

        results_box = QGroupBox("Extracted keywords / keyphrases")
        self.txt_results = QTextEdit(readOnly=True)
        self.txt_results.setToolTip(TOOLTIPS["results_area"])
        v_res = QVBoxLayout(results_box)
        v_res.addWidget(self.txt_results)
        root.addWidget(results_box)

    def _build_param_group(self, parent_layout: QVBoxLayout) -> None:
        """Create the parameter controls (ngram, stop-words, diversification, …)."""
        grp = QGroupBox("Extraction parameters")
        grid = QGridLayout(grp)

        # N-gram range
        grid.addWidget(
            QLabel("Keyphrase n-gram range:"), 0, 0, alignment=Qt.AlignRight
        )

        self.spin_min = QSpinBox()
        self.spin_min.setRange(1, 10)
        self.spin_min.setValue(1)
        self.spin_min.setToolTip(TOOLTIPS["ngram_min"])
        grid.addWidget(self.spin_min, 0, 1)

        grid.addWidget(QLabel("to"), 0, 2, alignment=Qt.AlignCenter)

        self.spin_max = QSpinBox()
        self.spin_max.setRange(1, 10)
        self.spin_max.setValue(2)
        self.spin_max.setToolTip(TOOLTIPS["ngram_max"])
        grid.addWidget(self.spin_max, 0, 3)

        # Stop-words
        grid.addWidget(QLabel("Stop words:"), 1, 0, alignment=Qt.AlignRight)

        self.line_stop = QLineEdit()
        self.line_stop.setToolTip(TOOLTIPS["stop_words_entry"])
        grid.addWidget(self.line_stop, 1, 1, 1, 3)

        # Diversification strategy
        grid.addWidget(QLabel("Diversification:"), 2, 0, alignment=Qt.AlignRight)

        self.combo_div = QComboBox()
        self.combo_div.addItems(["None", "Max Sum Similarity", "MMR"])
        self.combo_div.setToolTip(TOOLTIPS["diversification_dropdown"])
        self.combo_div.currentTextChanged.connect(self._on_div_change)
        grid.addWidget(self.combo_div, 2, 1)

        # --- diversity spinbox (only for MMR) ---
        grid.addWidget(QLabel("Diversity (0-1):"), 2, 2, alignment=Qt.AlignRight)

        self.spin_diversity = QDoubleSpinBox()
        self.spin_diversity.setRange(0.0, 1.0)
        self.spin_diversity.setSingleStep(0.1)
        self.spin_diversity.setValue(0.5)
        self.spin_diversity.setToolTip(TOOLTIPS["diversity_label"])
        grid.addWidget(self.spin_diversity, 2, 3)

        # nr_candidates  &  top_n
        grid.addWidget(QLabel("Candidates:"), 3, 0, alignment=Qt.AlignRight)

        self.spin_candidates = QSpinBox()
        self.spin_candidates.setRange(1, 1000)
        self.spin_candidates.setValue(20)
        self.spin_candidates.setToolTip(TOOLTIPS["candidates_spinbox"])
        grid.addWidget(self.spin_candidates, 3, 1)

        grid.addWidget(QLabel("Top N:"), 3, 2, alignment=Qt.AlignRight)

        self.spin_top_n = QSpinBox()
        self.spin_top_n.setRange(1, 1000)
        self.spin_top_n.setValue(5)
        self.spin_top_n.setToolTip(TOOLTIPS["top_n_spinbox"])
        grid.addWidget(self.spin_top_n, 3, 3)

        # Default-KeyBERT checkbox
        self.chk_default_kb = QCheckBox("Use default KeyBERT (no embeddings)")
        self.chk_default_kb.setToolTip(TOOLTIPS["default_keybert_checkbox"])
        self.chk_default_kb.stateChanged.connect(self._on_default_kb_change)
        grid.addWidget(self.chk_default_kb, 4, 0, 1, 4)

        parent_layout.addWidget(grp)
        self._on_div_change(self.combo_div.currentText())

    # Event handlers
    def _on_load_clicked(self) -> None:
        if self.text_worker and self.text_worker.isRunning():
            self.text_worker.quit()
            self.text_worker.wait()

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open file",
            "",
            "Text/Docs (*.txt *.pdf *.docx);;All files (*)",
        )
        if not path:
            return
        self._set_loading(True, "Extracting text…")
        self.text_worker = TextExtractorWorker(path)
        self.text_worker.text_extracted.connect(self.txt_input.setPlainText)
        self.text_worker.extraction_error.connect(
            lambda e: QMessageBox.critical(self, "Extraction error", e)
        )
        self.text_worker.finished.connect(lambda: self._set_loading(False))
        self.text_worker.start()

    def _on_model_select(self) -> None:
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Sentence-Transformers model directory"
        )
        if not dir_path:
            return
        try:
            self.processor.load_custom_model(dir_path)
            self._refresh_model_status()
            QMessageBox.information(self, "Model loaded", f"Loaded {dir_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Load failed", str(exc))

    def _on_extract_clicked(self) -> None:
        doc = self.txt_input.toPlainText().strip()
        if not doc:
            QMessageBox.warning(self, "No input", "Please load or paste a document.")
            return

        params = {
            "keyphrase_ngram_range": (self.spin_min.value(), self.spin_max.value()),
            "stop_words": (
                [w.strip() for w in self.line_stop.text().split(",") if w.strip()]
                if self.line_stop.text().strip()
                else None
            ),
            "top_n": self.spin_top_n.value(),
        }

        div_method = self.combo_div.currentText()
        if div_method == "Max Sum Similarity":
            params.update(
                use_maxsum=True,
                use_mmr=False,
                nr_candidates=self.spin_candidates.value(),
                diversity=0.5,
            )
        elif div_method == "MMR":
            params.update(
                use_maxsum=False,
                use_mmr=True,
                nr_candidates=20,
                diversity=self.spin_diversity.value(),
            )
        else:
            params.update(
                use_maxsum=False,
                use_mmr=False,
                nr_candidates=20,
                diversity=0.5,
            )

        errors = validate_keyword_params(params)
        if errors:
            QMessageBox.critical(self, "Invalid parameters", "\n".join(errors))
            return

        if self.keyword_worker and self.keyword_worker.isRunning():
            self.keyword_worker.quit()
            self.keyword_worker.wait()

        self.btn_extract.setEnabled(False)
        self.btn_extract.setText("Extracting…")

        self.keyword_worker = KeywordExtractorWorker(
            doc, self.processor.get_keybert_model(), params
        )
        self.keyword_worker.result.connect(self._show_results)
        self.keyword_worker.error.connect(
            lambda e: QMessageBox.critical(self, "Extraction error", e)
        )
        self.keyword_worker.finished.connect(
            lambda: (
                self.btn_extract.setEnabled(True),
                self.btn_extract.setText("Extract keywords"),
            )
        )
        self.keyword_worker.start()

    def _on_div_change(self, text: str) -> None:
        if text == "MMR":
            self.spin_diversity.setEnabled(True)
            self.spin_candidates.setEnabled(False)
        elif text == "Max Sum Similarity":
            self.spin_diversity.setEnabled(False)
            self.spin_candidates.setEnabled(True)
        else:
            self.spin_diversity.setEnabled(False)
            self.spin_candidates.setEnabled(False)

    def _on_default_kb_change(self) -> None:
        self.processor.update_keybert_model(self.chk_default_kb.isChecked())
        self._refresh_model_status()

    # Helpers
    def _refresh_model_status(self) -> None:
        if self.chk_default_kb.isChecked():
            self.lbl_model.setText("Using default KeyBERT")
        elif self.processor.has_custom_model():
            self.lbl_model.setText(f"Custom model: {self.processor.get_custom_model_name()}")
        else:
            self.lbl_model.setText(f"Default model: {self.processor.get_default_model_name()}")

    def _set_loading(self, is_loading: bool, text: str = "") -> None:
        self.progress.setVisible(is_loading)
        self.progress.setRange(0, 0 if is_loading else 1)
        self.btn_load.setEnabled(not is_loading)
        self.btn_load.setText(text or "Load File (TXT·PDF·DOCX)")

    def _show_results(self, keywords: list[tuple[str, float]]) -> None:
        self.txt_results.setPlainText(
            "\n".join(f"{kw} ({score:.4f})" for kw, score in keywords)
        )

    def closeEvent(self, event):
        """Gracefully stop background threads when the window closes."""
        if self.text_worker and self.text_worker.isRunning():
            self.text_worker.quit()
            self.text_worker.wait()
        if self.keyword_worker and self.keyword_worker.isRunning():
            self.keyword_worker.quit()
            self.keyword_worker.wait()
        event.accept()


# Convenience wrapper for `python -m keybert_gui`
def run_app() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.resize(900, 700)
    win.show()
    sys.exit(app.exec())
