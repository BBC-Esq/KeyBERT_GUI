TOOLTIPS = {
    # N-gram Range Controls
    "ngram_min": "N-grams are sequences of consecutive words. Range (1,1) extracts single words like 'learning', while (1,2) captures both single words and phrases like 'machine learning'. Longer ranges create more specific phrases but increase computation time.",
    
    "ngram_max": "N-grams are sequences of consecutive words. Range (1,1) extracts single words like 'learning', while (1,2) captures both single words and phrases like 'machine learning'. Longer ranges create more specific phrases but increase computation time.",
    
    "ngram_range_label": "Controls the length of keyword phrases extracted. (1,2) is optimal for most documents, providing both specific phrases and individual keywords. Longer ranges like (1,3) capture technical terminology but may miss important shorter terms.",
    
    # Stop Words
    "stop_words_entry": "High-frequency words like 'the', 'and', 'is' that carry minimal meaning. Filtering these prevents meaningless keywords like 'the algorithm'. Leave blank to use default English stop words, or add custom terms separated by commas.",
    
    "stop_words_label": "Common words filtered out during extraction to improve keyword quality. Default English filtering removes ~318 words. Add domain-specific terms like 'patient, study' for medical texts or 'system, method' for technical documents.",
    
    # Diversification Controls
    "diversification_dropdown": "Addresses semantic redundancy in results. 'None' ranks by relevance only. 'MMR' balances relevance with diversity efficiently. 'Max Sum Similarity' finds optimal diverse combinations but requires more computation.",
    
    "mmr_option": "Maximal Marginal Relevance iteratively selects keywords that balance document relevance with diversity from already-selected terms. Computationally efficient with quadratic scaling, suitable for production use.",
    
    "maxsum_option": "Finds the optimal combination of keywords with maximum document relevance and minimum inter-similarity. Examines all possible combinations for best diversity but has exponential computation time.",
    
    # Diversity Parameter
    "diversity_scale": "Controls relevance vs. variety trade-off in MMR. Values near 0 (like 0.2) emphasize document relevance. Values near 1 (like 0.8) prioritize diversity. Optimal range 0.4-0.6 provides balanced results for most applications.",
    
    "diversity_label": "Low values (0.2) produce highly relevant but similar keywords. High values (0.7) create varied keywords with potentially lower individual relevance. 0.5 typically provides the best balance of relevance and diversity.",
    
    # Candidates Parameter
    "candidates_spinbox": "Defines the intermediate pool size for Max Sum Similarity optimization. The algorithm finds the best top-N combination from these candidates. Keep below 20 for optimal performance. Larger pools increase diversity potential but also computation time.",
    
    "candidates_label": "For Max Sum Distance: pool size for combination optimization. Larger pools enable better diversity but exponentially increase computation. Optimal performance typically achieved with 15-25 candidates for documents under 1000 words.",
    
    # Top N Keywords
    "top_n_spinbox": "Final number of keywords returned after similarity calculations and diversification. Values 5-15 work best for most applications. Must be less than or equal to number of candidates when using diversification algorithms.",
    
    "top_n_label": "Target size for final keyword set. With diversification, represents optimized selection balancing relevance and variety. Without diversification, simply returns the N highest-scoring keywords by document similarity.",
    
    # Model Selection
    "default_keybert_checkbox": "Uses KeyBERT's built-in model without sentence transformers. Faster but less semantic sophistication. Uncheck to use sentence transformer models for better semantic understanding and contextual keyword extraction.",
    
    "custom_model_button": "Load custom sentence transformer models for domain-specific optimization. Medical, legal, or technical documents often benefit from specialized models trained on domain vocabulary and concepts.",
    
    # General Concepts
    "extract_button": "Processes text through KeyBERT's pipeline: generates n-gram candidates, applies stop word filtering, creates semantic embeddings, calculates document similarity, and applies diversification for final keyword selection.",
    
    "keybert_overview": "KeyBERT combines BERT's semantic understanding with classical keyword extraction. It embeds document and candidates in semantic space, measures similarity using cosine distance, then applies diversification to prevent redundant results.",
    
    # Performance Indicators
    "cuda_status": "GPU acceleration available for faster processing. CUDA enables parallel embedding computation, significantly reducing extraction time for large documents or batch processing multiple texts.",
    
    "model_status": "Shows current embedding model in use. Sentence transformer models (like all-MiniLM-L12-v2) provide superior semantic understanding compared to default statistical approaches. Custom models enable domain-specific optimization.",
    
    # Text Areas
    "text_area": "Enter text manually or view extracted text from loaded files. KeyBERT analyzes this content to identify semantically relevant keywords and keyphrases using BERT embeddings and similarity calculations.",
    
    "results_area": "Displays extracted keywords with similarity scores in parentheses. Higher scores indicate stronger semantic relevance to the document. Diversification algorithms ensure varied, non-redundant keyword selection.",
    
    # File Operations
    "load_file_button": "Load text files (TXT, PDF, DOC, DOCX) for keyword extraction. The system automatically extracts text content and displays it in the text area for processing with current parameter settings."
}