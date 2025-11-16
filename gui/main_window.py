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
from processing.output_formatter import SingleResultFormatter
from utils import (
    scan_directory_for_documents, 
    get_file_counts_by_type, 
    estimate_processing_time,
    is_nvidia_gpu_available,
    set_cuda_paths,
    validate_keyword_params,
    validate_batch_params,
    SettingsManager,
    DirectoryScanWorker,
)
from resources import TOOLTIPS

if is_nvidia_gpu_available():
    set_cuda_paths()

class MainWindow(QMainWindow):

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("KeyBERT GUI — Multi-Format Keyword Extractor")

        self.processor = KeyBERTProcessor()
        self.settings_manager = SettingsManager()
        self.settings = self.settings_manager.load_settings()

        self.text_worker: TextExtractorWorker | None = None
        self.keyword_worker: KeywordExtractorWorker | None = None
        self.batch_worker: BatchProcessorWorker | None = None
        self.scan_worker: DirectoryScanWorker | None = None
        
        self.current_source_file: str | None = None
        self.current_source_text: str | None = None
        self.current_keywords: list[tuple[str, float]] | None = None

        self._build_ui()
        self._load_settings()
        self._refresh_model_status()

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
        
        self.btn_cancel_load = QPushButton("Cancel")
        self.btn_cancel_load.setToolTip(TOOLTIPS["cancel_load_button"])
        self.btn_cancel_load.clicked.connect(self._on_cancel_load_clicked)
        self.btn_cancel_load.setVisible(False)
        
        self.btn_select_model = QPushButton("Select custom model dir")
        self.btn_select_model.setToolTip(TOOLTIPS["custom_model_button"])
        self.btn_select_model.clicked.connect(self._on_model_select)

        button_row.addWidget(self.btn_load)
        button_row.addWidget(self.btn_cancel_load)
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

        extract_row = QHBoxLayout()
        self.btn_extract = QPushButton("Extract keywords")
        self.btn_extract.setStyleSheet(
            "background-color: lightblue; font-weight: bold;"
        )
        self.btn_extract.setToolTip(TOOLTIPS["extract_button"])
        self.btn_extract.clicked.connect(self._on_extract_clicked)
        
        self.btn_cancel_extract = QPushButton("Cancel")
        self.btn_cancel_extract.setToolTip(TOOLTIPS["cancel_extract_button"])
        self.btn_cancel_extract.clicked.connect(self._on_cancel_extract_clicked)
        self.btn_cancel_extract.setVisible(False)
        
        extract_row.addWidget(self.btn_extract)
        extract_row.addWidget(self.btn_cancel_extract)
        extract_row.addStretch()
        root.addLayout(extract_row)

        self._build_batch_section(root)

        results_box = QGroupBox("Extracted keywords / keyphrases")
        results_layout = QVBoxLayout(results_box)
        
        self.txt_results = QTextEdit(readOnly=True)
        self.txt_results.setToolTip(TOOLTIPS["results_area"])
        results_layout.addWidget(self.txt_results)
        
        export_row = QHBoxLayout()
        self.btn_export = QPushButton("Export Results")
        self.btn_export.setToolTip(TOOLTIPS["export_button"])
        self.btn_export.clicked.connect(self._on_export_clicked)
        self.btn_export.setEnabled(False)
        export_row.addWidget(self.btn_export)
        export_row.addStretch()
        results_layout.addLayout(export_row)
        
        root.addWidget(results_box)

    def _build_param_group(self, parent_layout: QVBoxLayout) -> None:
        grp = QGroupBox("Extraction parameters")
        grid = QGridLayout(grp)

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

        grid.addWidget(QLabel("Stop words:"), 1, 0, alignment=Qt.AlignRight)

        self.line_stop = QLineEdit()
        self.line_stop.setToolTip(TOOLTIPS["stop_words_entry"])
        grid.addWidget(self.line_stop, 1, 1, 1, 3)

        grid.addWidget(QLabel("Diversification:"), 2, 0, alignment=Qt.AlignRight)

        self.combo_div = QComboBox()
        self.combo_div.addItems(["None", "Max Sum Similarity", "MMR"])
        self.combo_div.setToolTip(TOOLTIPS["diversification_dropdown"])
        self.combo_div.currentTextChanged.connect(self._on_div_change)
        grid.addWidget(self.combo_div, 2, 1)

        grid.addWidget(QLabel("Diversity (0-1):"), 2, 2, alignment=Qt.AlignRight)

        self.spin_diversity = QDoubleSpinBox()
        self.spin_diversity.setRange(0.0, 1.0)
        self.spin_diversity.setSingleStep(0.1)
        self.spin_diversity.setValue(0.5)
        self.spin_diversity.setToolTip(TOOLTIPS["diversity_label"])
        grid.addWidget(self.spin_diversity, 2, 3)

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

        self.chk_default_kb = QCheckBox("Use default KeyBERT (no embeddings)")
        self.chk_default_kb.setToolTip(TOOLTIPS["default_keybert_checkbox"])
        self.chk_default_kb.stateChanged.connect(self._on_default_kb_change)
        grid.addWidget(self.chk_default_kb, 4, 0, 1, 4)

        parent_layout.addWidget(grp)
        self._on_div_change(self.combo_div.currentText())

    def _build_batch_section(self, parent_layout: QVBoxLayout) -> None:
        self.batch_group = QGroupBox("Batch Processing")
        self.batch_group.setCheckable(True)
        self.batch_group.setChecked(False)
        self.batch_group.setToolTip(TOOLTIPS["batch_toggle_button"])
        
        batch_layout = QVBoxLayout(self.batch_group)
        
        dir_row = QHBoxLayout()
        dir_row.addWidget(QLabel("Directory:"))
        
        self.line_batch_dir = QLineEdit()
        self.line_batch_dir.setPlaceholderText("Select directory containing documents...")
        self.line_batch_dir.setToolTip(TOOLTIPS["batch_directory_label"])
        dir_row.addWidget(self.line_batch_dir)
        
        self.btn_batch_dir = QPushButton("Browse")
        self.btn_batch_dir.setToolTip(TOOLTIPS["batch_directory_button"])
        self.btn_batch_dir.clicked.connect(self._on_batch_dir_clicked)
        dir_row.addWidget(self.btn_batch_dir)
        
        batch_layout.addLayout(dir_row)
        
        self.lbl_file_count = QLabel("No directory selected")
        self.lbl_file_count.setStyleSheet("color: gray; font-style: italic;")
        self.lbl_file_count.setToolTip(TOOLTIPS["batch_file_count_label"])
        batch_layout.addWidget(self.lbl_file_count)
        
        output_row = QHBoxLayout()
        output_row.addWidget(QLabel("Output JSON:"))
        
        self.line_batch_output = QLineEdit()
        self.line_batch_output.setPlaceholderText("Select output file location...")
        self.line_batch_output.setToolTip(TOOLTIPS["batch_output_label"])
        output_row.addWidget(self.line_batch_output)
        
        self.btn_batch_output = QPushButton("Browse")
        self.btn_batch_output.setToolTip(TOOLTIPS["batch_output_button"])
        self.btn_batch_output.clicked.connect(self._on_batch_output_clicked)
        output_row.addWidget(self.btn_batch_output)
        
        batch_layout.addLayout(output_row)
        
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
        
        self.batch_progress = QProgressBar()
        self.batch_progress.setVisible(False)
        self.batch_progress.setToolTip(TOOLTIPS["batch_progress_bar"])
        batch_layout.addWidget(self.batch_progress)
        
        self.lbl_batch_status = QLabel()
        self.lbl_batch_status.setVisible(False)
        self.lbl_batch_status.setStyleSheet("color: blue;")
        batch_layout.addWidget(self.lbl_batch_status)
        
        parent_layout.addWidget(self.batch_group)

    def _load_settings(self) -> None:
        geom = self.settings["window_geometry"]
        self.resize(geom["width"], geom["height"])
        if geom["x"] is not None and geom["y"] is not None:
            self.move(geom["x"], geom["y"])
        
        params = self.settings["extraction_params"]
        self.spin_min.setValue(params["ngram_min"])
        self.spin_max.setValue(params["ngram_max"])
        self.line_stop.setText(params["stop_words"])
        self.combo_div.setCurrentText(params["diversification"])
        self.spin_diversity.setValue(params["diversity"])
        self.spin_candidates.setValue(params["candidates"])
        self.spin_top_n.setValue(params["top_n"])
        self.chk_default_kb.setChecked(params["use_default_keybert"])
        
        paths = self.settings["paths"]
        if paths["last_batch_dir"]:
            self.line_batch_dir.setText(paths["last_batch_dir"])
            self._update_file_count_display()
        if paths["last_batch_output"]:
            self.line_batch_output.setText(paths["last_batch_output"])
        if paths["custom_model_path"]:
            try:
                self.processor.load_custom_model(paths["custom_model_path"])
            except Exception:
                pass

    def _save_settings(self) -> None:
        self.settings["window_geometry"] = {
            "width": self.width(),
            "height": self.height(),
            "x": self.x(),
            "y": self.y()
        }
        
        self.settings["extraction_params"] = {
            "ngram_min": self.spin_min.value(),
            "ngram_max": self.spin_max.value(),
            "stop_words": self.line_stop.text(),
            "diversification": self.combo_div.currentText(),
            "diversity": self.spin_diversity.value(),
            "candidates": self.spin_candidates.value(),
            "top_n": self.spin_top_n.value(),
            "use_default_keybert": self.chk_default_kb.isChecked()
        }
        
        self.settings["paths"]["last_batch_dir"] = self.line_batch_dir.text()
        self.settings["paths"]["last_batch_output"] = self.line_batch_output.text()
        if self.processor.has_custom_model():
            self.settings["paths"]["custom_model_path"] = self.processor.get_custom_model_name()
        
        self.settings_manager.save_settings(self.settings)

    def _on_load_clicked(self) -> None:
        if self.text_worker and self.text_worker.isRunning():
            self._terminate_worker(self.text_worker, 3000)

        start_dir = self.settings["paths"].get("last_file_dir", "")
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open file",
            start_dir,
            "Text/Docs (*.txt *.pdf *.docx);;All files (*)",
        )
        if not path:
            return
        
        self.settings["paths"]["last_file_dir"] = str(Path(path).parent)
        self.current_source_file = path
        
        self._set_loading(True, "Extracting text…")
        self.text_worker = TextExtractorWorker(path)
        self.text_worker.text_extracted.connect(self._on_text_extracted)
        self.text_worker.extraction_error.connect(
            lambda e: QMessageBox.critical(self, "Extraction error", e)
        )
        self.text_worker.finished.connect(lambda: self._set_loading(False))
        self.text_worker.start()

    def _on_cancel_load_clicked(self) -> None:
        if self.text_worker and self.text_worker.isRunning():
            self._terminate_worker(self.text_worker, 3000)
        self._set_loading(False)

    def _on_text_extracted(self, text: str) -> None:
        self.txt_input.setPlainText(text)
        self.current_source_text = text

    def _on_model_select(self) -> None:
        start_dir = self.settings["paths"].get("custom_model_path", "")
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Sentence-Transformers model directory", start_dir
        )
        if not dir_path:
            return
        try:
            self.processor.load_custom_model(dir_path)
            self.settings["paths"]["custom_model_path"] = dir_path
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
            self._terminate_worker(self.keyword_worker, 3000)

        self.current_source_text = doc
        self._set_extracting(True)

        self.keyword_worker = KeywordExtractorWorker(
            doc, self.processor.get_keybert_model(), params
        )
        self.keyword_worker.result.connect(self._show_results)
        self.keyword_worker.error.connect(
            lambda e: QMessageBox.critical(self, "Extraction error", e)
        )
        self.keyword_worker.finished.connect(lambda: self._set_extracting(False))
        self.keyword_worker.start()

    def _on_cancel_extract_clicked(self) -> None:
        if self.keyword_worker and self.keyword_worker.isRunning():
            self._terminate_worker(self.keyword_worker, 3000)
        self._set_extracting(False)

    def _on_export_clicked(self) -> None:
        if not self.current_keywords:
            QMessageBox.warning(self, "No results", "No keywords to export.")
            return
        
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Results",
            "",
            "JSON files (*.json);;CSV files (*.csv);;All files (*)"
        )
        
        if not file_path:
            return
        
        if not Path(file_path).suffix:
            if "JSON" in selected_filter:
                file_path += ".json"
            elif "CSV" in selected_filter:
                file_path += ".csv"
            else:
                file_path += ".json"
        
        try:
            params = self._get_extraction_params()
            SingleResultFormatter.save_to_file(
                file_path,
                self.current_keywords,
                source_file=self.current_source_file,
                source_text=self.current_source_text,
                parameters=params
            )
            QMessageBox.information(
                self, 
                "Export successful", 
                f"Results exported to:\n{file_path}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Export failed", str(e))

    def _on_batch_dir_clicked(self) -> None:
        start_dir = self.settings["paths"].get("last_batch_dir", "")
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select directory containing documents", start_dir
        )
        if not dir_path:
            return
        
        self.line_batch_dir.setText(dir_path)
        self.settings["paths"]["last_batch_dir"] = dir_path
        self._update_file_count_display()

    def _on_batch_output_clicked(self) -> None:
        start_path = self.settings["paths"].get("last_batch_output", "")
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save batch results as JSON",
            start_path,
            "JSON files (*.json);;All files (*)"
        )
        if not output_path:
            return
        
        if not output_path.lower().endswith('.json'):
            output_path += '.json'
        
        self.line_batch_output.setText(output_path)
        self.settings["paths"]["last_batch_output"] = output_path

    def _on_batch_process_clicked(self) -> None:
        directory = self.line_batch_dir.text().strip()
        output_path = self.line_batch_output.text().strip()
        
        errors = validate_batch_params(directory, output_path)
        if errors:
            QMessageBox.critical(self, "Invalid batch parameters", "\n".join(errors))
            return
        
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
        
        self._start_batch_processing(files, output_path)

    def _on_batch_cancel_clicked(self) -> None:
        if self.batch_worker and self.batch_worker.isRunning():
            self.batch_worker.cancel()
            self._terminate_worker(self.batch_worker, 5000)
        
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

    def _update_file_count_display(self) -> None:
        directory = self.line_batch_dir.text().strip()
        if not directory:
            self.lbl_file_count.setText("No directory selected")
            self.lbl_file_count.setStyleSheet("color: gray; font-style: italic;")
            return
        
        if self.scan_worker and self.scan_worker.isRunning():
            self._terminate_worker(self.scan_worker, 2000)
        
        self.lbl_file_count.setText("Scanning directory...")
        self.lbl_file_count.setStyleSheet("color: blue; font-style: italic;")
        
        self.scan_worker = DirectoryScanWorker(directory)
        self.scan_worker.scan_completed.connect(self._on_scan_completed)
        self.scan_worker.scan_error.connect(self._on_scan_error)
        self.scan_worker.start()

    def _on_scan_completed(self, files: list[Path]) -> None:
        if not files:
            self.lbl_file_count.setText("No supported files found")
            self.lbl_file_count.setStyleSheet("color: orange; font-style: italic;")
        else:
            file_counts = get_file_counts_by_type(files)
            count_text = ", ".join(f"{count} {ext}" for ext, count in file_counts.items() if count > 0)
            estimate = estimate_processing_time(len(files))
            self.lbl_file_count.setText(f"{len(files)} files found ({count_text}) - {estimate}")
            self.lbl_file_count.setStyleSheet("color: green;")

    def _on_scan_error(self, error: str) -> None:
        self.lbl_file_count.setText(f"Error scanning directory: {error}")
        self.lbl_file_count.setStyleSheet("color: red; font-style: italic;")

    def _get_extraction_params(self) -> dict:
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
        params = self._get_extraction_params()
        
        self.batch_worker = BatchProcessorWorker(
            files, Path(output_path), self.processor.get_keybert_model(), params
        )
        
        self.batch_worker.progress_update.connect(self._on_batch_progress_update)
        self.batch_worker.file_completed.connect(self._on_batch_file_completed)
        self.batch_worker.batch_completed.connect(self._on_batch_completed)
        self.batch_worker.error_occurred.connect(self._on_batch_error)
        
        self._set_batch_loading(True)
        self.batch_progress.setMaximum(len(files))
        self.batch_progress.setValue(0)
        
        self.batch_worker.start()

    def _set_batch_loading(self, is_loading: bool) -> None:
        self.btn_batch_process.setEnabled(not is_loading)
        self.btn_batch_cancel.setVisible(is_loading)
        self.batch_progress.setVisible(is_loading)
        self.lbl_batch_status.setVisible(is_loading or bool(self.lbl_batch_status.text()))
        
        self.btn_load.setEnabled(not is_loading)
        self.btn_extract.setEnabled(not is_loading)

    def _on_batch_progress_update(self, current: int, total: int, filename: str) -> None:
        self.batch_progress.setValue(current)
        self.lbl_batch_status.setText(f"Processing {filename} ({current}/{total})")
        self.lbl_batch_status.setStyleSheet("color: blue;")

    def _on_batch_file_completed(self, file_path: str, success: bool) -> None:
        pass

    def _on_batch_completed(self, output_path: str, summary_stats: dict) -> None:
        self._set_batch_loading(False)
        
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
        
        reply = QMessageBox.question(
            self,
            "Batch processing completed",
            message + "\n\nOpen output file location?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            output_dir = Path(output_path).parent
            if sys.platform == "win32":
                os.startfile(output_dir)
            elif sys.platform == "darwin":
                os.system(f"open '{output_dir}'")
            else:
                os.system(f"xdg-open '{output_dir}'")

    def _on_batch_error(self, error_message: str) -> None:
        self._set_batch_loading(False)
        self.lbl_batch_status.setText("Batch processing failed")
        self.lbl_batch_status.setStyleSheet("color: red;")
        QMessageBox.critical(self, "Batch processing error", error_message)

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
        self.btn_cancel_load.setVisible(is_loading)

    def _set_extracting(self, is_extracting: bool) -> None:
        self.btn_extract.setEnabled(not is_extracting)
        self.btn_extract.setText("Extracting…" if is_extracting else "Extract keywords")
        self.btn_cancel_extract.setVisible(is_extracting)

    def _show_results(self, keywords: list[tuple[str, float]]) -> None:
        self.current_keywords = keywords
        self.txt_results.setPlainText(
            "\n".join(f"{kw} ({score:.4f})" for kw, score in keywords)
        )
        self.btn_export.setEnabled(True)

    def _terminate_worker(self, worker: QThread, timeout_ms: int) -> None:
        worker.quit()
        if not worker.wait(timeout_ms):
            worker.terminate()
            worker.wait()

    def closeEvent(self, event):
        workers = [
            (self.text_worker, 3000),
            (self.keyword_worker, 3000),
            (self.batch_worker, 5000),
            (self.scan_worker, 2000)
        ]
        
        for worker, timeout in workers:
            if worker and worker.isRunning():
                if hasattr(worker, 'cancel'):
                    worker.cancel()
                self._terminate_worker(worker, timeout)
        
        self._save_settings()
        event.accept()


def run_app() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.resize(900, 800)
    win.show()
    sys.exit(app.exec())