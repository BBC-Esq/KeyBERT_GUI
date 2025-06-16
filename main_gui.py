import sys
import os
import subprocess

def is_nvidia_gpu_available():
    try:
        subprocess.run(["nvidia-smi"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

if is_nvidia_gpu_available():
    from pathlib import Path

    def set_cuda_paths():
        from pathlib import Path
        
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
            str(cuda_path_runtime),
            str(cuda_path_runtime_lib),
            str(cuda_path_runtime_include),
            str(cublas_path),
            str(cudnn_path),
            str(nvrtc_path),
            str(nvcc_path),
        ]
        current_value = os.environ.get('PATH', '')
        new_value = os.pathsep.join(paths_to_add + ([current_value] if current_value else []))
        os.environ['PATH'] = new_value

        triton_cuda_path = nvidia_base_path / 'cuda_runtime'
        current_cuda_path = os.environ.get('CUDA_PATH', '')
        new_cuda_path = os.pathsep.join([str(triton_cuda_path)] + ([current_cuda_path] if current_cuda_path else []))
        os.environ['CUDA_PATH'] = new_cuda_path

        set_cuda_paths()


from PySide6.QtWidgets import (
    QApplication, QWidget, QMainWindow, QLabel, QPushButton, QTextEdit, QFileDialog,
    QMessageBox, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QCheckBox,
    QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox, QScrollArea
)
from PySide6.QtCore import Qt, Signal, QObject, QThread
from PySide6.QtGui import QFont, QIcon
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
import torch


class KeywordExtractorWorker(QThread):
    result = Signal(list)
    error = Signal(str)
    finished = Signal()

    def __init__(self, doc, kw_model, params):
        super().__init__()
        self.doc = doc
        self.kw_model = kw_model
        self.params = params

    def run(self):
        try:
            if self.params['use_maxsum']:
                keywords = self.kw_model.extract_keywords(
                    self.doc,
                    keyphrase_ngram_range=self.params['keyphrase_ngram_range'],
                    stop_words=self.params['stop_words'],
                    use_maxsum=True,
                    nr_candidates=self.params['nr_candidates'],
                    top_n=self.params['top_n']
                )
            elif self.params['use_mmr']:
                keywords = self.kw_model.extract_keywords(
                    self.doc,
                    keyphrase_ngram_range=self.params['keyphrase_ngram_range'],
                    stop_words=self.params['stop_words'],
                    use_mmr=True,
                    diversity=self.params['diversity'],
                    top_n=self.params['top_n']
                )
            else:
                keywords = self.kw_model.extract_keywords(
                    self.doc,
                    keyphrase_ngram_range=self.params['keyphrase_ngram_range'],
                    stop_words=self.params['stop_words'],
                    top_n=self.params['top_n']
                )
            self.result.emit(keywords)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()

# Main Window Class
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("KeyBERT GUI with CUDA and Custom Model Support")
        self.setGeometry(100, 100, 800, 600)

        # Initialize default Sentence-Transformer model
        self.default_model_name = 'all-MiniLM-L12-v2'
        self.use_cuda = torch.cuda.is_available()
        self.device = 'cuda' if self.use_cuda else 'cpu'

        try:
            self.default_sentence_model = SentenceTransformer(
                self.default_model_name, 
                device=self.device,
                trust_remote_code=True,
            )
        except Exception as e:
            QMessageBox.critical(self, "Model Loading Error", f"Failed to load the default model:\n{str(e)}")
            sys.exit(1)

        # Global variables to hold custom model and kw_model
        self.kw_model = None
        self.custom_model = None
        self.custom_model_name = None

        # Setup UI first to initialize all widgets
        self.setup_ui()

        # Initialize kw_model after UI is set up
        self.update_kw_model()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # Top Frame for CUDA and Model Info
        frame_top = QHBoxLayout()
        main_layout.addLayout(frame_top)

        # CUDA Status
        self.lbl_cuda = QLabel(f"CUDA: {'Available' if self.use_cuda else 'Not Available'}")
        self.lbl_cuda.setStyleSheet("color: blue;")
        self.lbl_cuda.setToolTip("Indicates whether CUDA (GPU acceleration) is available.")
        frame_top.addWidget(self.lbl_cuda, alignment=Qt.AlignLeft)

        # Model Status
        self.lbl_model_status = QLabel("")
        self.lbl_model_status.setStyleSheet("color: green;")
        self.lbl_model_status.setToolTip("Displays the current Sentence-Transformers model in use.")
        frame_top.addWidget(self.lbl_model_status, alignment=Qt.AlignRight)

        # Main Content Frame (Input and Model Selection)
        frame_input = QVBoxLayout()
        main_layout.addLayout(frame_input)

        # Buttons for loading text and selecting model
        frame_input_buttons = QHBoxLayout()
        frame_input.addLayout(frame_input_buttons)

        btn_load = QPushButton("Load Text File")
        btn_load.setToolTip("Load a text file (.txt) to extract keywords from.")
        btn_load.clicked.connect(self.load_text)
        frame_input_buttons.addWidget(btn_load)

        btn_select_model = QPushButton("Select Custom Model Directory")
        btn_select_model.setToolTip("Select a directory containing a custom Sentence-Transformers model to use for keyword extraction.")
        btn_select_model.clicked.connect(self.select_model_directory)
        frame_input_buttons.addWidget(btn_select_model)

        # Text area for document input
        self.text_area = QTextEdit()
        self.text_area.setToolTip("Enter or view the text document from which to extract keywords/keyphrases.")
        frame_input.addWidget(self.text_area)

        # Parameters Frame
        frame_params = QGroupBox("Extraction Parameters")
        main_layout.addWidget(frame_params)
        params_layout = QGridLayout()
        frame_params.setLayout(params_layout)

        # Ngram range
        lbl_ngram = QLabel("Keyphrase Ngram Range:")
        lbl_ngram.setToolTip("Minimum and maximum number of words in keyphrases (e.g., 1 for single words to 2 for bi-grams).")
        params_layout.addWidget(lbl_ngram, 0, 0, 1, 1, Qt.AlignRight)

        self.min_ngram = QSpinBox()
        self.min_ngram.setRange(1, 10)
        self.min_ngram.setValue(1)
        self.min_ngram.setToolTip("Minimum number of words in keyphrases (e.g., 1 for single words).")
        params_layout.addWidget(self.min_ngram, 0, 1, 1, 1, Qt.AlignLeft)

        lbl_to = QLabel("to")
        params_layout.addWidget(lbl_to, 0, 2, 1, 1, Qt.AlignCenter)

        self.max_ngram = QSpinBox()
        self.max_ngram.setRange(1, 10)
        self.max_ngram.setValue(2)
        self.max_ngram.setToolTip("Maximum number of words in keyphrases (e.g., 2 for bi-grams).")
        params_layout.addWidget(self.max_ngram, 0, 3, 1, 1, Qt.AlignLeft)

        # Stop words
        lbl_stop_words = QLabel("Stop Words:")
        lbl_stop_words.setToolTip("Enter stop words separated by commas to exclude from extraction. Leave blank for none.")
        params_layout.addWidget(lbl_stop_words, 1, 0, 1, 1, Qt.AlignRight)

        self.stop_words_entry = QLineEdit()
        self.stop_words_entry.setToolTip("Enter stop words separated by commas to exclude from extraction. Leave blank for none.")
        params_layout.addWidget(self.stop_words_entry, 1, 1, 1, 3, Qt.AlignLeft)

        lbl_stop_words_note = QLabel("(Leave blank for None)")
        lbl_stop_words_note.setToolTip("Leave blank for none.")
        params_layout.addWidget(lbl_stop_words_note, 1, 4, 1, 1, Qt.AlignLeft)

        # Diversification method
        lbl_diversification = QLabel("Diversification:")
        lbl_diversification.setToolTip("Select a diversification method to reduce keyword/keyphrase similarity:\n- None: Basic extraction.\n- Max Sum Similarity: Diverse keywords based on maximum sum similarity.\n- MMR: Maximal Marginal Relevance for high diversity.")
        params_layout.addWidget(lbl_diversification, 2, 0, 1, 1, Qt.AlignRight)

        self.dropdown_diversification = QComboBox()
        self.dropdown_diversification.addItems(["None", "Max Sum Similarity", "MMR"])
        self.dropdown_diversification.setToolTip("Select a diversification method to reduce keyword/keyphrase similarity:\n- None: Basic extraction.\n- Max Sum Similarity: Diverse keywords based on maximum sum similarity.\n- MMR: Maximal Marginal Relevance for high diversity.")
        self.dropdown_diversification.currentTextChanged.connect(self.update_diversification)  # Add this line
        params_layout.addWidget(self.dropdown_diversification, 2, 1, 1, 1, Qt.AlignLeft)

        # Diversity scale (only for MMR)
        lbl_diversity = QLabel("Diversity (0-1):")
        lbl_diversity.setToolTip("Sets the diversity level for MMR (Maximal Marginal Relevance).\n- 0: Low diversity (more similar keywords).\n- 1: High diversity (more diverse keywords).")
        params_layout.addWidget(lbl_diversity, 2, 2, 1, 1, Qt.AlignRight)

        self.diversity_scale = QDoubleSpinBox()
        self.diversity_scale.setRange(0.0, 1.0)
        self.diversity_scale.setSingleStep(0.1)
        self.diversity_scale.setValue(0.5)
        self.diversity_scale.setToolTip("Sets the diversity level for MMR (Maximal Marginal Relevance).\n- 0: Low diversity (more similar keywords).\n- 1: High diversity (more diverse keywords).")
        params_layout.addWidget(self.diversity_scale, 2, 3, 1, 1, Qt.AlignLeft)

        # Number of Candidates (for Max Sum Similarity)
        lbl_nr_candidates = QLabel("Number of Candidates:")
        lbl_nr_candidates.setToolTip("Number of candidate keywords/keyphrases to consider for extraction.\nApplicable when using Max Sum Similarity.")
        params_layout.addWidget(lbl_nr_candidates, 3, 0, 1, 1, Qt.AlignRight)

        self.nr_candidates_entry = QSpinBox()
        self.nr_candidates_entry.setRange(1, 1000)
        self.nr_candidates_entry.setValue(20)
        self.nr_candidates_entry.setToolTip("Number of candidate keywords/keyphrases to consider for extraction.\nApplicable when using Max Sum Similarity.")
        params_layout.addWidget(self.nr_candidates_entry, 3, 1, 1, 1, Qt.AlignLeft)

        # Top N Keywords
        lbl_top_n = QLabel("Top N Keywords:")
        lbl_top_n.setToolTip("Number of top keywords/keyphrases to extract.")
        params_layout.addWidget(lbl_top_n, 3, 2, 1, 1, Qt.AlignRight)

        self.top_n_entry = QSpinBox()
        self.top_n_entry.setRange(1, 1000)
        self.top_n_entry.setValue(5)
        self.top_n_entry.setToolTip("Number of top keywords/keyphrases to extract.")
        params_layout.addWidget(self.top_n_entry, 3, 3, 1, 1, Qt.AlignLeft)

        # Checkbox for using default KeyBERT model (no embedding model)
        self.chk_use_default_keybert = QCheckBox("Use default KeyBERT model (no embedding model)")
        self.chk_use_default_keybert.setToolTip("Check this to use the default KeyBERT model without specifying an embedding model.")
        self.chk_use_default_keybert.stateChanged.connect(self.update_kw_model)
        params_layout.addWidget(self.chk_use_default_keybert, 4, 0, 1, 4, Qt.AlignLeft)

        # Extract Button
        self.btn_extract = QPushButton("Extract Keywords")
        self.btn_extract.setStyleSheet("background-color: lightblue; color: black; font-weight: bold;")
        self.btn_extract.setToolTip("Click to extract keywords/keyphrases based on the provided parameters.")
        self.btn_extract.clicked.connect(self.extract_keywords)
        main_layout.addWidget(self.btn_extract)

        # Results Frame
        frame_results = QGroupBox("Extracted Keywords/Keyphrases")
        main_layout.addWidget(frame_results)
        results_layout = QVBoxLayout()
        frame_results.setLayout(results_layout)

        self.results_area = QTextEdit()
        self.results_area.setReadOnly(True)
        self.results_area.setToolTip("Displays the extracted keywords/keyphrases along with their scores.")
        results_layout.addWidget(self.results_area)

        # Footer Frame for Additional Info
        frame_footer = QHBoxLayout()
        main_layout.addLayout(frame_footer)

        lbl_footer = QLabel("KeyBERT GUI with CUDA and Custom Model Support")
        lbl_footer.setStyleSheet("color: grey;")
        lbl_footer.setToolTip("This application uses KeyBERT for keyword extraction with support for CUDA and custom models.")
        frame_footer.addWidget(lbl_footer, alignment=Qt.AlignRight)

        # Initialize diversification state
        self.update_diversification()

        # Update model status after all UI elements are created
        self.update_model_status()

    def update_model_status(self):
        if self.chk_use_default_keybert.isChecked():
            self.lbl_model_status.setText("Using default KeyBERT model (no embedding model)")
        elif self.custom_model:
            self.lbl_model_status.setText(f"Using Custom Model: {self.custom_model_name}")
        else:
            self.lbl_model_status.setText(f"Using Default Model: {self.default_model_name}")

    def update_kw_model(self):
        if self.chk_use_default_keybert.isChecked():
            self.kw_model = KeyBERT()
        else:
            if self.custom_model:
                self.kw_model = KeyBERT(model=self.custom_model)
            else:
                self.kw_model = KeyBERT(model=self.default_sentence_model)
        self.update_model_status()

    def load_text(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Text File", "", "Text Files (*.txt);;All Files (*)"
        )
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    self.text_area.setPlainText(content)
            except Exception as e:
                QMessageBox.critical(self, "File Error", f"Failed to load the file:\n{str(e)}")

    def select_model_directory(self):
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Sentence-Transformers Model Directory", ""
        )
        if dir_path:
            if not os.path.isdir(dir_path):
                QMessageBox.critical(self, "Invalid Directory", "Selected path is not a directory.")
                return
            try:
                loaded_model = SentenceTransformer(
                    dir_path,
                    device=self.device,
                    trust_remote_code=True,
                )
                self.custom_model = loaded_model
                self.custom_model_name = os.path.basename(dir_path)
                self.update_kw_model()
            except Exception as e:
                QMessageBox.critical(self, "Model Loading Error", f"Failed to load the model:\n{str(e)}")

    def extract_keywords(self):
        doc = self.text_area.toPlainText().strip()
        if not doc:
            QMessageBox.warning(self, "Input Required", "Please enter or load a document.")
            return

        # Get ngram range
        min_n = self.min_ngram.value()
        max_n = self.max_ngram.value()
        if min_n > max_n or min_n < 1:
            QMessageBox.critical(self, "Invalid Input", "Please enter valid integers for n-gram range (min <= max and min >= 1).")
            return
        keyphrase_ngram_range = (min_n, max_n)

        # Get stop words
        stop_words_str = self.stop_words_entry.text().strip()
        if stop_words_str:
            stop_words = [word.strip() for word in stop_words_str.split(",")]
        else:
            stop_words = None

        # Get diversification options
        diversification = self.dropdown_diversification.currentText()
        use_maxsum = False
        use_mmr = False
        diversity = 0.5  # Default value

        if diversification == "Max Sum Similarity":
            use_maxsum = True
        elif diversification == "MMR":
            use_mmr = True
            diversity = self.diversity_scale.value()
            if not (0 <= diversity <= 1):
                QMessageBox.critical(self, "Invalid Input", "Diversity must be a float between 0 and 1.")
                return

        # Get top_n
        top_n = self.top_n_entry.value()
        if top_n < 1:
            QMessageBox.critical(self, "Invalid Input", "Please enter a valid integer for Top N Keywords (>=1).")
            return

        # Get nr_candidates if needed
        nr_candidates = None
        if use_maxsum:
            nr_candidates = self.nr_candidates_entry.value()
            if nr_candidates < top_n or nr_candidates < 1:
                QMessageBox.critical(self, "Invalid Input", "Please enter a valid integer for Number of Candidates (>= Top N Keywords and >=1).")
                return

        # Prepare parameters
        params = {
            'keyphrase_ngram_range': keyphrase_ngram_range,
            'stop_words': stop_words,
            'use_maxsum': use_maxsum,
            'use_mmr': use_mmr,
            'diversity': diversity,
            'top_n': top_n,
            'nr_candidates': nr_candidates
        }

        # Disable the extract button and change its text to indicate processing
        self.btn_extract.setEnabled(False)
        self.btn_extract.setText("Extracting...")

        # Start the worker thread
        self.worker = KeywordExtractorWorker(doc, self.kw_model, params)
        self.worker.result.connect(self.display_results)
        self.worker.error.connect(self.show_error)
        self.worker.finished.connect(self.extracting_finished)
        self.worker.start()

    def display_results(self, keywords):
        results_text = ""
        for kw, score in keywords:
            results_text += f"{kw} ({score:.4f})\n"
        self.results_area.setPlainText(results_text)

    def show_error(self, error_message):
        QMessageBox.critical(self, "Extraction Error", f"An error occurred during extraction:\n{error_message}")

    def extracting_finished(self):
        # Re-enable the extract button
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

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Set Fusion style

    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
