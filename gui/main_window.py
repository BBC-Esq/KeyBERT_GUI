from pathlib import Path
import sys
import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QDoubleSpinBox, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton,
    QProgressBar, QSpinBox, QTextEdit, QVBoxLayout, QWidget, QMainWindow
)

from processing.keybert import KeyBERTProcessor, KeywordExtractorWorker
from processing.extraction import TextExtractorWorker
from processing.batch_processor import BatchProcessorWorker
from utils import (
    scan_directory_for_documents, 
    get_file_counts_by_type, 
    estimate_processing_time,
    is_nvidia_gpu_available,
    set_cuda_paths,
    validate_keyword_params,
    validate_batch_params,
)
from resources import TOOLTIPS

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
        self.batch_worker: BatchProcessorWorker | None = None

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

        # Single file processing section
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

        # Add batch processing section
        self._build_batch_section(root)

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

    def _build_batch_section(self, parent_layout: QVBoxLayout) -> None:
        """Create the batch processing section."""
        # Collapsible batch processing group
        self.batch_group = QGroupBox("Batch Processing")
        self.batch_group.setCheckable(True)
        self.batch_group.setChecked(False)
        self.batch_group.setToolTip(TOOLTIPS["batch_toggle_button"])
        
        batch_layout = QVBoxLayout(self.batch_group)
        
        # Directory selection
        dir_row = QHBoxLayout()
        dir_row.addWidget(QLabel("Directory:"))
        
        self.line_batch_dir = QLineEdit()
        self.line_batch_dir.setPlaceholderText("Select directory containing documents...")  # Fixed: setPlaceholder -> setPlaceholderText
        self.line_batch_dir.setToolTip(TOOLTIPS["batch_directory_label"])
        dir_row.addWidget(self.line_batch_dir)
        
        self.btn_batch_dir = QPushButton("Browse")
        self.btn_batch_dir.setToolTip(TOOLTIPS["batch_directory_button"])
        self.btn_batch_dir.clicked.connect(self._on_batch_dir_clicked)
        dir_row.addWidget(self.btn_batch_dir)
        
        batch_layout.addLayout(dir_row)
        
        # File count display
        self.lbl_file_count = QLabel("No directory selected")
        self.lbl_file_count.setStyleSheet("color: gray; font-style: italic;")
        self.lbl_file_count.setToolTip(TOOLTIPS["batch_file_count_label"])
        batch_layout.addWidget(self.lbl_file_count)
        
        # Output file selection
        output_row = QHBoxLayout()
        output_row.addWidget(QLabel("Output JSON:"))
        
        self.line_batch_output = QLineEdit()
        self.line_batch_output.setPlaceholderText("Select output file location...")  # Fixed: setPlaceholder -> setPlaceholderText
        self.line_batch_output.setToolTip(TOOLTIPS["batch_output_label"])
        output_row.addWidget(self.line_batch_output)
        
        self.btn_batch_output = QPushButton("Browse")
        self.btn_batch_output.setToolTip(TOOLTIPS["batch_output_button"])
        self.btn_batch_output.clicked.connect(self._on_batch_output_clicked)
        output_row.addWidget(self.btn_batch_output)
        
        batch_layout.addLayout(output_row)
        
        # Batch processing controls
        batch_controls = QHBoxLayout()
        
        self.btn_batch_process = QPushButton("Process Directory")
        self.btn_batch_process.setStyleSheet(
            "background-color: lightgreen; font-weight: bold;"
        )
        self.btn_batch_process.setToolTip(TOOLTIPS["batch_process_button"])
        self.btn_batch_process.clicked.connect(self._on_batch_process_clicked)
        batch_controls.addWidget(self.btn_batch_process)
        
        self.btn_batch_cancel = QPushButton("Cancel")
        self.btn_batch_cancel.setToolTip(TOOLTIPS["batch_cancel_button"])
        self.btn_batch_cancel.clicked.connect(self._on_batch_cancel_clicked)
        self.btn_batch_cancel.setVisible(False)
        batch_controls.addWidget(self.btn_batch_cancel)
        
        batch_controls.addStretch()
        batch_layout.addLayout(batch_controls)
        
        # Batch progress bar
        self.batch_progress = QProgressBar()
        self.batch_progress.setVisible(False)
        self.batch_progress.setToolTip(TOOLTIPS["batch_progress_bar"])
        batch_layout.addWidget(self.batch_progress)
        
        # Batch status label
        self.lbl_batch_status = QLabel()
        self.lbl_batch_status.setVisible(False)
        self.lbl_batch_status.setStyleSheet("color: blue;")
        batch_layout.addWidget(self.lbl_batch_status)
        
        parent_layout.addWidget(self.batch_group)

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

        params = self._get_extraction_params()
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

    def _on_batch_dir_clicked(self) -> None:
        """Handle batch directory selection."""
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select directory containing documents"
        )
        if not dir_path:
            return
        
        self.line_batch_dir.setText(dir_path)
        self._update_file_count_display()

    def _on_batch_output_clicked(self) -> None:
        """Handle batch output file selection."""
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save batch results as JSON",
            "",
            "JSON files (*.json);;All files (*)"
        )
        if not output_path:
            return
        
        # Ensure .json extension
        if not output_path.lower().endswith('.json'):
            output_path += '.json'
        
        self.line_batch_output.setText(output_path)

    def _on_batch_process_clicked(self) -> None:
        """Handle batch processing start."""
        directory = self.line_batch_dir.text().strip()
        output_path = self.line_batch_output.text().strip()
        
        # Validate inputs
        errors = validate_batch_params(directory, output_path)
        if errors:
            QMessageBox.critical(self, "Invalid batch parameters", "\n".join(errors))
            return
        
        # Scan for files
        try:
            files = scan_directory_for_documents(directory)
        except Exception as e:
            QMessageBox.critical(self, "Directory scan error", str(e))
            return
        
        if not files:
            QMessageBox.warning(
                self, 
                "No files found", 
                "No supported files (.txt, .pdf, .docx) found in the selected directory."
            )
            return
        
        # Confirm processing
        file_counts = get_file_counts_by_type(files)
        count_text = ", ".join(f"{count} {ext}" for ext, count in file_counts.items() if count > 0)
        estimate = estimate_processing_time(len(files))
        
        reply = QMessageBox.question(
            self,
            "Confirm batch processing",
            f"Process {len(files)} files ({count_text})?\n\nEstimated time: {estimate}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        # Start batch processing
        self._start_batch_processing(files, output_path)

    def _on_batch_cancel_clicked(self) -> None:
        """Handle batch processing cancellation."""
        if self.batch_worker and self.batch_worker.isRunning():
            self.batch_worker.cancel()
            self.batch_worker.quit()
            self.batch_worker.wait()
        
        self._set_batch_loading(False)
        self.lbl_batch_status.setText("Batch processing cancelled")
        self.lbl_batch_status.setStyleSheet("color: orange;")

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

    # Batch processing helpers
    def _update_file_count_display(self) -> None:
        """Update the file count display based on selected directory."""
        directory = self.line_batch_dir.text().strip()
        if not directory:
            self.lbl_file_count.setText("No directory selected")
            self.lbl_file_count.setStyleSheet("color: gray; font-style: italic;")
            return
        
        try:
            files = scan_directory_for_documents(directory)
            if not files:
                self.lbl_file_count.setText("No supported files found")
                self.lbl_file_count.setStyleSheet("color: orange; font-style: italic;")
            else:
                file_counts = get_file_counts_by_type(files)
                count_text = ", ".join(f"{count} {ext}" for ext, count in file_counts.items() if count > 0)
                estimate = estimate_processing_time(len(files))
                self.lbl_file_count.setText(f"{len(files)} files found ({count_text}) - {estimate}")
                self.lbl_file_count.setStyleSheet("color: green;")
        except Exception as e:
            self.lbl_file_count.setText(f"Error scanning directory: {e}")
            self.lbl_file_count.setStyleSheet("color: red; font-style: italic;")

    def _get_extraction_params(self) -> dict:
        """Get current extraction parameters from UI."""
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
        
        return params

    def _start_batch_processing(self, files: list[Path], output_path: str) -> None:
        """Start the batch processing worker."""
        params = self._get_extraction_params()
        
        self.batch_worker = BatchProcessorWorker(
            files, Path(output_path), self.processor.get_keybert_model(), params
        )
        
        # Connect signals
        self.batch_worker.progress_update.connect(self._on_batch_progress_update)
        self.batch_worker.file_completed.connect(self._on_batch_file_completed)
        self.batch_worker.batch_completed.connect(self._on_batch_completed)
        self.batch_worker.error_occurred.connect(self._on_batch_error)
        
        # Update UI
        self._set_batch_loading(True)
        self.batch_progress.setMaximum(len(files))
        self.batch_progress.setValue(0)
        
        # Start processing
        self.batch_worker.start()

    def _set_batch_loading(self, is_loading: bool) -> None:
        """Update UI for batch processing state."""
        self.btn_batch_process.setEnabled(not is_loading)
        self.btn_batch_cancel.setVisible(is_loading)
        self.batch_progress.setVisible(is_loading)
        self.lbl_batch_status.setVisible(is_loading or bool(self.lbl_batch_status.text()))
        
        # Disable single-file processing during batch
        self.btn_load.setEnabled(not is_loading)
        self.btn_extract.setEnabled(not is_loading)

    def _on_batch_progress_update(self, current: int, total: int, filename: str) -> None:
        """Handle batch processing progress updates."""
        self.batch_progress.setValue(current)
        self.lbl_batch_status.setText(f"Processing {filename} ({current}/{total})")
        self.lbl_batch_status.setStyleSheet("color: blue;")

    def _on_batch_file_completed(self, file_path: str, success: bool) -> None:
        """Handle individual file completion."""
        # Could add more detailed logging here if needed
        pass

    def _on_batch_completed(self, output_path: str, summary_stats: dict) -> None:
        """Handle batch processing completion."""
        self._set_batch_loading(False)
        
        # Show completion message
        stats = summary_stats
        message = (
            f"Batch processing completed!\n\n"
            f"Total files: {stats['total_files']}\n"
            f"Successful: {stats['successful']}\n"
            f"Failed: {stats['failed']}\n"
            f"Success rate: {stats['success_rate']:.1f}%\n\n"
            f"Results saved to: {output_path}"
        )
        
        self.lbl_batch_status.setText(f"Completed: {stats['successful']}/{stats['total_files']} files")
        self.lbl_batch_status.setStyleSheet("color: green;")
        
        # Ask if user wants to open output location
        reply = QMessageBox.question(
            self,
            "Batch processing completed",
            message + "\n\nOpen output file location?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # Open file location in system file manager
            output_dir = Path(output_path).parent
            if sys.platform == "win32":
                os.startfile(output_dir)
            elif sys.platform == "darwin":
                os.system(f"open '{output_dir}'")
            else:
                os.system(f"xdg-open '{output_dir}'")

    def _on_batch_error(self, error_message: str) -> None:
        """Handle batch processing errors."""
        self._set_batch_loading(False)
        self.lbl_batch_status.setText("Batch processing failed")
        self.lbl_batch_status.setStyleSheet("color: red;")
        QMessageBox.critical(self, "Batch processing error", error_message)

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
        if self.batch_worker and self.batch_worker.isRunning():
            self.batch_worker.cancel()
            self.batch_worker.quit()
            self.batch_worker.wait()
        event.accept()


# Convenience wrapper for `python -m keybert_gui`
def run_app() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.resize(900, 800)  # Slightly taller to accommodate batch section
    win.show()
    sys.exit(app.exec())