import sys
import os
import subprocess
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication, QWidget, QMainWindow, QLabel, QPushButton, QTextEdit, QFileDialog,
    QMessageBox, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QCheckBox,
    QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox, QProgressBar
)
from PySide6.QtCore import Qt, Signal, QObject, QThread
from PySide6.QtGui import QFont, QIcon
import torch

from text_extractor import TextExtractorWorker
from keybert_processor import KeyBERTProcessor, KeywordExtractorWorker
from constants import TOOLTIPS

def is_nvidia_gpu_available():
    try:
        subprocess.run(["nvidia-smi"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

if is_nvidia_gpu_available():
    def set_cuda_paths():
        venv_base = Path(sys.executable).parent.parent
        nvidia_base_path = venv_base / 'Lib' / 'site-packages' / 'nvidia'
        cuda_path_runtime = nvidia_base_path / 'cuda_runtime' / 'bin'
        cuda_path_runtime_lib = nvidia_base_path / 'cuda_runtime' / 'lib' / 'x64'
        cuda_path_runtime_include = nvidia_base_path / 'cuda_runtime' / 'include'
        cublas_path = nvidia_base_path / 'cublas' / 'bin'
        cudnn_path = nvidia_base_path / 'cudnn' / 'bin'
        nvrtc_path = nvidia_base_path / 'cuda_nvrtc' / 'bin'
        nvcc_path = nvidia_base_path / 'cuda_nvcc' / 'bin'
        paths_to_add = [
            str(cuda_path_runtime), str(cuda_path_runtime_lib), str(cuda_path_runtime_include),
            str(cublas_path), str(cudnn_path), str(nvrtc_path), str(nvcc_path),
        ]
        current_value = os.environ.get('PATH', '')
        new_value = os.pathsep.join(paths_to_add + ([current_value] if current_value else []))
        os.environ['PATH'] = new_value

        triton_cuda_path = nvidia_base_path / 'cuda_runtime'
        current_cuda_path = os.environ.get('CUDA_PATH', '')
        new_cuda_path = os.pathsep.join([str(triton_cuda_path)] + ([current_cuda_path] if current_cuda_path else []))
        os.environ['CUDA_PATH'] = new_cuda_path

    set_cuda_paths()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("KeyBERT GUI with Multi-Format Text Extraction")
        self.setGeometry(100, 100, 900, 700)

        self.keybert_processor = KeyBERTProcessor()
        self.use_cuda = torch.cuda.is_available()

        self.text_worker = None
        self.keyword_worker = None

        self.setup_ui()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        frame_top = QHBoxLayout()
        main_layout.addLayout(frame_top)

        self.lbl_cuda = QLabel(f"CUDA: {'Available' if self.use_cuda else 'Not Available'}")
        self.lbl_cuda.setStyleSheet("color: blue;")
        self.lbl_cuda.setToolTip(TOOLTIPS["cuda_status"])
        frame_top.addWidget(self.lbl_cuda, alignment=Qt.AlignLeft)

        self.lbl_model_status = QLabel("")
        self.lbl_model_status.setStyleSheet("color: green;")
        self.lbl_model_status.setToolTip(TOOLTIPS["model_status"])
        frame_top.addWidget(self.lbl_model_status, alignment=Qt.AlignRight)

        frame_input = QVBoxLayout()
        main_layout.addLayout(frame_input)

        frame_input_buttons = QHBoxLayout()
        frame_input.addLayout(frame_input_buttons)

        self.btn_load = QPushButton("Load File (TXT, PDF, DOC, DOCX)")
        self.btn_load.setToolTip(TOOLTIPS["load_file_button"])
        self.btn_load.clicked.connect(self.load_file)
        frame_input_buttons.addWidget(self.btn_load)

        self.btn_select_model = QPushButton("Select Custom Model Directory")
        self.btn_select_model.setToolTip(TOOLTIPS["custom_model_button"])
        self.btn_select_model.clicked.connect(self.select_model_directory)
        frame_input_buttons.addWidget(self.btn_select_model)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        frame_input.addWidget(self.progress_bar)

        self.text_area = QTextEdit()
        self.text_area.setToolTip(TOOLTIPS["text_area"])
        frame_input.addWidget(self.text_area)

        self.setup_parameters_frame(main_layout)

        self.btn_extract = QPushButton("Extract Keywords")
        self.btn_extract.setStyleSheet("background-color: lightblue; color: black; font-weight: bold;")
        self.btn_extract.setToolTip(TOOLTIPS["extract_button"])
        self.btn_extract.clicked.connect(self.extract_keywords)
        main_layout.addWidget(self.btn_extract)

        frame_results = QGroupBox("Extracted Keywords/Keyphrases")
        main_layout.addWidget(frame_results)
        results_layout = QVBoxLayout()
        frame_results.setLayout(results_layout)

        self.results_area = QTextEdit()
        self.results_area.setReadOnly(True)
        self.results_area.setToolTip(TOOLTIPS["results_area"])
        results_layout.addWidget(self.results_area)

        frame_footer = QHBoxLayout()
        main_layout.addLayout(frame_footer)
        lbl_footer = QLabel("KeyBERT GUI with Multi-Format Text Extraction")
        lbl_footer.setStyleSheet("color: grey;")
        frame_footer.addWidget(lbl_footer, alignment=Qt.AlignRight)

        self.update_diversification()
        self.update_model_status()

    def setup_parameters_frame(self, main_layout):
        frame_params = QGroupBox("Extraction Parameters")
        main_layout.addWidget(frame_params)
        params_layout = QGridLayout()
        frame_params.setLayout(params_layout)

        lbl_ngram = QLabel("Keyphrase Ngram Range:")
        lbl_ngram.setToolTip(TOOLTIPS["ngram_range_label"])
        params_layout.addWidget(lbl_ngram, 0, 0, 1, 1, Qt.AlignRight)

        self.min_ngram = QSpinBox()
        self.min_ngram.setRange(1, 10)
        self.min_ngram.setValue(1)
        self.min_ngram.setToolTip(TOOLTIPS["ngram_min"])
        params_layout.addWidget(self.min_ngram, 0, 1, 1, 1)

        lbl_to = QLabel("to")
        params_layout.addWidget(lbl_to, 0, 2, 1, 1, Qt.AlignCenter)

        self.max_ngram = QSpinBox()
        self.max_ngram.setRange(1, 10)
        self.max_ngram.setValue(2)
        self.max_ngram.setToolTip(TOOLTIPS["ngram_max"])
        params_layout.addWidget(self.max_ngram, 0, 3, 1, 1)

        lbl_stop_words = QLabel("Stop Words:")
        lbl_stop_words.setToolTip(TOOLTIPS["stop_words_label"])
        params_layout.addWidget(lbl_stop_words, 1, 0, 1, 1, Qt.AlignRight)

        self.stop_words_entry = QLineEdit()
        self.stop_words_entry.setToolTip(TOOLTIPS["stop_words_entry"])
        params_layout.addWidget(self.stop_words_entry, 1, 1, 1, 3)

        lbl_stop_words_note = QLabel("(Leave blank for None)")
        params_layout.addWidget(lbl_stop_words_note, 1, 4, 1, 1)

        lbl_diversification = QLabel("Diversification:")
        lbl_diversification.setToolTip(TOOLTIPS["diversification_dropdown"])
        params_layout.addWidget(lbl_diversification, 2, 0, 1, 1, Qt.AlignRight)

        self.dropdown_diversification = QComboBox()
        self.dropdown_diversification.addItems(["None", "Max Sum Similarity", "MMR"])
        self.dropdown_diversification.currentTextChanged.connect(self.update_diversification)
        self.dropdown_diversification.setToolTip(TOOLTIPS["diversification_dropdown"])
        params_layout.addWidget(self.dropdown_diversification, 2, 1, 1, 1)

        lbl_diversity = QLabel("Diversity (0-1):")
        lbl_diversity.setToolTip(TOOLTIPS["diversity_label"])
        params_layout.addWidget(lbl_diversity, 2, 2, 1, 1, Qt.AlignRight)

        self.diversity_scale = QDoubleSpinBox()
        self.diversity_scale.setRange(0.0, 1.0)
        self.diversity_scale.setSingleStep(0.1)
        self.diversity_scale.setValue(0.5)
        self.diversity_scale.setToolTip(TOOLTIPS["diversity_scale"])
        params_layout.addWidget(self.diversity_scale, 2, 3, 1, 1)

        lbl_nr_candidates = QLabel("Number of Candidates:")
        lbl_nr_candidates.setToolTip(TOOLTIPS["candidates_label"])
        params_layout.addWidget(lbl_nr_candidates, 3, 0, 1, 1, Qt.AlignRight)

        self.nr_candidates_entry = QSpinBox()
        self.nr_candidates_entry.setRange(1, 1000)
        self.nr_candidates_entry.setValue(20)
        self.nr_candidates_entry.setToolTip(TOOLTIPS["candidates_spinbox"])
        params_layout.addWidget(self.nr_candidates_entry, 3, 1, 1, 1)

        lbl_top_n = QLabel("Top N Keywords:")
        lbl_top_n.setToolTip(TOOLTIPS["top_n_label"])
        params_layout.addWidget(lbl_top_n, 3, 2, 1, 1, Qt.AlignRight)

        self.top_n_entry = QSpinBox()
        self.top_n_entry.setRange(1, 1000)
        self.top_n_entry.setValue(5)
        self.top_n_entry.setToolTip(TOOLTIPS["top_n_spinbox"])
        params_layout.addWidget(self.top_n_entry, 3, 3, 1, 1)

        self.chk_use_default_keybert = QCheckBox("Use default KeyBERT model (no embedding model)")
        self.chk_use_default_keybert.stateChanged.connect(self.update_keybert_model)
        self.chk_use_default_keybert.setToolTip(TOOLTIPS["default_keybert_checkbox"])
        params_layout.addWidget(self.chk_use_default_keybert, 4, 0, 1, 4)

    def load_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open File", "", 
            "All Supported (*.txt *.pdf *.doc *.docx);;Text Files (*.txt);;PDF Files (*.pdf);;Word Documents (*.doc *.docx);;All Files (*)"
        )
        if file_path:
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)  # Indeterminate progress
            self.btn_load.setEnabled(False)
            self.btn_load.setText("Extracting Text...")

            # Start text extraction worker
            self.text_worker = TextExtractorWorker(file_path)
            self.text_worker.text_extracted.connect(self.on_text_extracted)
            self.text_worker.extraction_error.connect(self.on_text_extraction_error)
            self.text_worker.finished.connect(self.on_text_extraction_finished)
            self.text_worker.start()

    def on_text_extracted(self, text):
        self.text_area.setPlainText(text)

    def on_text_extraction_error(self, error_message):
        QMessageBox.critical(self, "Text Extraction Error", f"Failed to extract text:\n{error_message}")

    def on_text_extraction_finished(self):
        self.progress_bar.setVisible(False)
        self.btn_load.setEnabled(True)
        self.btn_load.setText("Load File (TXT, PDF, DOC, DOCX)")

    def select_model_directory(self):
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Sentence-Transformers Model Directory", ""
        )
        if dir_path:
            try:
                self.keybert_processor.load_custom_model(dir_path)
                self.update_model_status()
                QMessageBox.information(self, "Model Loaded", f"Successfully loaded model from:\n{dir_path}")
            except Exception as e:
                QMessageBox.critical(self, "Model Loading Error", f"Failed to load the model:\n{str(e)}")

    def extract_keywords(self):
        doc = self.text_area.toPlainText().strip()
        if not doc:
            QMessageBox.warning(self, "Input Required", "Please enter or load a document.")
            return

        # Validate parameters
        if not self.validate_parameters():
            return

        # Prepare parameters
        params = self.get_extraction_parameters()

        # Disable extract button and show progress
        self.btn_extract.setEnabled(False)
        self.btn_extract.setText("Extracting Keywords...")

        # Start keyword extraction worker
        self.keyword_worker = KeywordExtractorWorker(
            doc, self.keybert_processor.get_keybert_model(), params
        )
        self.keyword_worker.result.connect(self.display_results)
        self.keyword_worker.error.connect(self.show_extraction_error)
        self.keyword_worker.finished.connect(self.on_keyword_extraction_finished)
        self.keyword_worker.start()

    def validate_parameters(self):
        min_n = self.min_ngram.value()
        max_n = self.max_ngram.value()
        if min_n > max_n or min_n < 1:
            QMessageBox.critical(self, "Invalid Input", "Please enter valid integers for n-gram range (min <= max and min >= 1).")
            return False

        top_n = self.top_n_entry.value()
        if top_n < 1:
            QMessageBox.critical(self, "Invalid Input", "Please enter a valid integer for Top N Keywords (>=1).")
            return False

        diversification = self.dropdown_diversification.currentText()
        if diversification == "MMR":
            diversity = self.diversity_scale.value()
            if not (0 <= diversity <= 1):
                QMessageBox.critical(self, "Invalid Input", "Diversity must be between 0 and 1.")
                return False
        elif diversification == "Max Sum Similarity":
            nr_candidates = self.nr_candidates_entry.value()
            if nr_candidates < top_n or nr_candidates < 1:
                QMessageBox.critical(self, "Invalid Input", "Number of Candidates must be >= Top N Keywords and >= 1.")
                return False

        return True

    def get_extraction_parameters(self):
        min_n = self.min_ngram.value()
        max_n = self.max_ngram.value()
        keyphrase_ngram_range = (min_n, max_n)

        stop_words_str = self.stop_words_entry.text().strip()
        stop_words = [word.strip() for word in stop_words_str.split(",")] if stop_words_str else None

        diversification = self.dropdown_diversification.currentText()
        use_maxsum = diversification == "Max Sum Similarity"
        use_mmr = diversification == "MMR"

        return {
            'keyphrase_ngram_range': keyphrase_ngram_range,
            'stop_words': stop_words,
            'use_maxsum': use_maxsum,
            'use_mmr': use_mmr,
            'diversity': self.diversity_scale.value(),
            'top_n': self.top_n_entry.value(),
            'nr_candidates': self.nr_candidates_entry.value()
        }

    def display_results(self, keywords):
        results_text = ""
        for kw, score in keywords:
            results_text += f"{kw} ({score:.4f})\n"
        self.results_area.setPlainText(results_text)

    def show_extraction_error(self, error_message):
        QMessageBox.critical(self, "Extraction Error", f"An error occurred during extraction:\n{error_message}")

    def on_keyword_extraction_finished(self):
        self.btn_extract.setEnabled(True)
        self.btn_extract.setText("Extract Keywords")

    def update_diversification(self):
        diversification = self.dropdown_diversification.currentText()
        if diversification == "MMR":
            self.diversity_scale.setEnabled(True)
            self.nr_candidates_entry.setEnabled(False)
        elif diversification == "Max Sum Similarity":
            self.diversity_scale.setEnabled(False)
            self.nr_candidates_entry.setEnabled(True)
        else:  # "None"
            self.diversity_scale.setEnabled(False)
            self.nr_candidates_entry.setEnabled(False)

    def update_model_status(self):
        if self.chk_use_default_keybert.isChecked():
            self.lbl_model_status.setText("Using default KeyBERT model")
        elif self.keybert_processor.has_custom_model():
            model_name = self.keybert_processor.get_custom_model_name()
            self.lbl_model_status.setText(f"Custom Model: {model_name}")
        else:
            default_name = self.keybert_processor.get_default_model_name()
            self.lbl_model_status.setText(f"Default Model: {default_name}")

    def update_keybert_model(self):
        use_default = self.chk_use_default_keybert.isChecked()
        self.keybert_processor.update_keybert_model(use_default)
        self.update_model_status()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()