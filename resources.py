"""
Central place for user-visible strings, tool-tips, etc.
You can expand this dict later or load it from JSON/YAML if localisation
becomes necessary.
"""

TOOLTIPS: dict[str, str] = {
    # Status labels
    "cuda_status": "CUDA / GPU availability detected at start-up",
    "model_status": "Name of the embedding model currently in use",

    # Buttons & fields
    "load_file_button": "Open a .txt, .pdf or .docx file and extract its text",
    "custom_model_button": "Choose a custom Sentence-Transformers model directory",
    "text_area": "Paste or edit the document here",
    "extract_button": "Run KeyBERT keyword extraction",
    "results_area": "Keywords extracted from the document will appear here",

    # Parameter widgets
    "ngram_range_label": "Minimum and maximum n-gram sizes to consider",
    "ngram_min": "Lower n-gram bound",
    "ngram_max": "Upper n-gram bound",
    "stop_words_label": "Comma-separated custom stop-words",
    "stop_words_entry": "Leave blank to disable stop-word removal",
    "diversification_dropdown": "Choose a diversification strategy",
    "diversity_label": "Only for MMR: how diverse results should be (0-1)",
    "candidates_label": "Only for Max-Sum: how many candidate phrases to consider",
    "candidates_spinbox": "nr_candidates (must be ≥ Top N)",
    "top_n_label": "Number of keywords to return",
    "top_n_spinbox": "top_n (must be ≥ 1)",
    "default_keybert_checkbox": "Skip embeddings and use KeyBERT in zero-shot mode",
    
    # Batch processing tooltips
    "batch_toggle_button": "Show/hide batch processing options",
    "batch_directory_button": "Select a directory containing documents to process",
    "batch_directory_label": "Directory containing .txt, .pdf, and .docx files",
    "batch_output_button": "Choose where to save the batch processing results (JSON file)",
    "batch_output_label": "JSON file where results will be saved",
    "batch_process_button": "Process all supported files in the selected directory",
    "batch_progress_bar": "Shows progress of batch processing operation",
    "batch_file_count_label": "Number and types of files found in directory",
    "batch_cancel_button": "Cancel the current batch processing operation",
}