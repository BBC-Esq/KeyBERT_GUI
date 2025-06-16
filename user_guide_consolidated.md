# KeyBERT Keyword Extraction: Complete Technical Guide

KeyBERT represents a sophisticated approach to keyword extraction that combines the semantic understanding of BERT embeddings with classical information retrieval techniques. This comprehensive guide examines the technical foundations, parameter mechanics, and practical applications of KeyBERT's keyword extraction system.

## Understanding n-gram ranges in KeyBERT

**N-grams form the foundation of keyword candidate generation in KeyBERT.** An n-gram is a sequence of N consecutive words from text - unigrams (1-gram) are single words like "learning", bigrams (2-gram) are two-word phrases like "machine learning", and trigrams (3-gram) are three-word phrases like "supervised learning algorithm".

The `keyphrase_ngram_range` parameter controls this extraction process through a tuple `(min_n, max_n)`. Setting `(1,1)` extracts only single keywords, while `(1,2)` captures both individual words and two-word phrases. The impact on keyword types is dramatic:

- **Range (1,1)**: Produces broad keywords like "learning", "algorithm", "training" - good for general topics but lacks specificity
- **Range (1,2)**: Generates balanced results combining specific phrases like "machine learning" with individual keywords - optimal for most applications  
- **Range (1,3)**: Creates highly specific phrases like "supervised learning algorithm" but may miss important shorter terms

KeyBERT uses scikit-learn's CountVectorizer for n-gram extraction, applying the range during the candidate generation phase. **Longer n-gram ranges exponentially increase the candidate pool size**, affecting both computational performance and keyword quality. For documents under 500 words, `(1,2)` provides optimal results, while longer documents benefit from `(2,3)` or `(1,3)` ranges to capture complex concepts.

The algorithm processes each n-gram range systematically: it tokenizes the document, generates all possible n-grams within the specified range, applies stop word filtering, then embeds each candidate for similarity comparison with the document embedding.

## Stop words in natural language processing

**Stop words are high-frequency words that carry minimal semantic meaning** and are typically filtered out during text processing. In KeyBERT, stop words serve a crucial role in improving keyword quality by preventing common words like "the", "and", "is" from appearing as extracted keywords.

KeyBERT implements stop word filtering during the tokenization phase using scikit-learn's built-in lists. The default setting `stop_words='english'` removes approximately 318 common English words. **This filtering occurs before candidate embedding, not after**, making it computationally efficient.

The impact on keyword extraction is significant. Without stop word filtering, results might include meaningless terms like "the algorithm" or "and learning". With proper filtering, KeyBERT extracts semantically meaningful keywords that better represent document content.

**Custom stop word configuration enables domain-specific optimization.** Medical documents might add terms like "patient", "study", "clinical" to the stop word list, while technical documentation could include "system", "method", "approach". Multi-language support exists through language-specific stop word lists: `stop_words='german'` for German text, though this requires corresponding multilingual embedding models.

Common stop word lists include function words (prepositions, conjunctions, articles), high-frequency verbs ("is", "has", "will"), and domain-general terms. The key consideration is balancing semantic filtering with preserving meaningful content - overly aggressive stop word filtering can remove important domain-specific terms.

## Diversification methods in keyword extraction

**Diversification addresses the core problem of semantic redundancy in keyword extraction.** Without diversification, KeyBERT might return highly similar keywords like "machine learning", "learning algorithm", and "learning machine", which provide limited additional information despite high individual relevance scores.

### Max Sum Similarity algorithm

The Max Sum Similarity algorithm solves diversification through global optimization. **It maximizes document similarity while minimizing inter-candidate similarity**, using a two-stage process:

1. **Candidate Selection**: Select top `nr_candidates` keywords by document similarity (typically 20)
2. **Combination Optimization**: Find the `top_n` combination with minimum pairwise similarity

The mathematical formulation involves:
```
objective = maximize(Σ doc_similarity) - minimize(Σ inter_candidate_similarity)
```

This approach examines all possible combinations of `top_n` keywords from the candidate pool, calculating the sum of pairwise similarities for each combination. **The combination with the lowest inter-similarity score becomes the final keyword set.**

Max Sum Distance provides optimal diversity but requires exponential computation time - iterating through all possible combinations of candidates. It's most effective when `nr_candidates` is kept below 20 and document vocabulary is moderate.

### Maximal Marginal Relevance (MMR) algorithm

**MMR provides an elegant balance between relevance and diversity through iterative selection.** Originally developed for information retrieval by Carbonell & Goldstein (1998), MMR has become foundational in reducing redundancy while maintaining quality.

The algorithm follows these steps:

1. **Initialize**: Select the keyword with highest document similarity
2. **Iterate**: For each remaining position, calculate MMR scores for all candidates
3. **Select**: Choose the keyword with the highest MMR score
4. **Update**: Add selected keyword to results, remove from candidates

The **mathematical formulation** captures this balance:
```
MMR = argmax[λ × Sim(keyword, document) - (1-λ) × max(Sim(keyword, selected))]
```

Where λ is the diversity parameter controlling the relevance-diversity trade-off. **Each iteration selects the keyword that maximizes document relevance while minimizing similarity to previously selected keywords.**

MMR's greedy approach makes it computationally efficient - O(k×m²) complexity for k keywords from m candidates, compared to Max Sum's exponential complexity. This efficiency makes MMR suitable for production applications while maintaining high-quality diverse results.

## Diversity parameter mechanics and effects

**The diversity parameter (0-1 range) controls the balance between relevance and variety in MMR selection.** This parameter, often denoted as λ in academic literature but implemented as `diversity` in KeyBERT, fundamentally changes keyword extraction behavior.

**Values closer to 0 (low diversity)** emphasize document relevance over variety. Setting `diversity=0.2` produces results like:
- "supervised learning algorithm" (0.7502)
- "learning algorithm analyzes" (0.7587)  
- "learning machine learning" (0.7577)

These keywords share high semantic similarity and strong document relevance but offer limited conceptual diversity.

**Values closer to 1 (high diversity)** prioritize variety over individual relevance. Setting `diversity=0.7` generates:
- "algorithm generalize training" (0.7727)
- "labels unseen instances" (0.1649)
- "new examples optimal" (0.4185)

Notice the significant variation in similarity scores - some keywords have lower document relevance but provide broader topic coverage.

**The mathematical effect** occurs in MMR's scoring function. Low diversity values weight the document similarity term heavily, while high diversity values emphasize the penalty for similarity to already-selected keywords. **The optimal range for most applications is 0.4-0.6**, providing balanced results with both relevance and variety.

Practical applications require diversity tuning based on use case: SEO keyword extraction benefits from moderate diversity (0.5-0.6) to cover related search terms, while academic research extraction might use lower diversity (0.3-0.4) to maintain conceptual coherence.

## Number of candidates parameter

**The candidates parameter defines the intermediate selection pool** from which final keywords are chosen. This parameter operates differently depending on the diversification method used.

For **Max Sum Similarity**, `nr_candidates` specifies how many top-ranked keywords by document similarity form the optimization pool. The algorithm then finds the optimal `top_n` combination from this pool. Setting `nr_candidates=20` with `top_n=5` means the algorithm examines the 20 most similar keywords and finds the 5-keyword combination with maximum internal diversity.

For **MMR selection**, candidates typically include all extracted n-grams, with the algorithm iteratively selecting from this full pool. However, pre-filtering can improve efficiency by limiting candidates to the top-scoring options by document similarity.

**The relationship between candidates and final results is crucial.** A larger candidate pool increases diversity potential but may introduce noise. Research shows optimal performance when `nr_candidates` remains below 20% of the document's unique words. For a 500-word document with 200 unique words, `nr_candidates=40` would be the theoretical maximum, though practical performance often peaks around 20-30 candidates.

**Computational impact scales significantly** with candidate pool size. Max Sum Similarity's exponential complexity makes large candidate pools impractical, while MMR's quadratic scaling remains manageable for moderate pool sizes.

## Top N keywords parameter and ranking

**The top_n parameter controls the final number of keywords returned**, representing the output size after all similarity calculations and diversification processes complete. This parameter interacts with the candidate pool and diversification methods to determine final results.

In **simple cosine similarity mode** (no diversification), KeyBERT ranks all candidates by document similarity and returns the top_n highest-scoring keywords. This provides maximum relevance but no diversity protection.

With **diversification enabled**, top_n represents the target size for the optimized keyword set. For Max Sum Distance, the algorithm finds the specific top_n combination with optimal diversity characteristics. For MMR, the algorithm iteratively builds a top_n-sized set balancing relevance and diversity.

**The ranking mechanism** depends on the selection method. Without diversification, ranking follows pure cosine similarity scores. With MMR, keywords are ranked by their selection order - the first keyword has highest document similarity, while subsequent keywords balance relevance with diversity relative to previously selected terms.

**Best practices suggest top_n values between 5-15** for most applications. Smaller values (3-5) work well for focused keyword extraction, while larger values (10-15) suit comprehensive topic analysis. The relationship to candidates should maintain `top_n ≤ nr_candidates` for diversification algorithms.

## Essential NLP concepts for understanding KeyBERT

### TF-IDF foundations and limitations

**TF-IDF (Term Frequency-Inverse Document Frequency) represents the statistical foundation** that KeyBERT transcends. The mathematical formulation combines local term importance with global rarity:

```
TF-IDF(term, document, corpus) = TF(term, document) × IDF(term, corpus)
```

TF-IDF assigns higher weights to terms that appear frequently in specific documents but rarely across the entire corpus. **While effective for basic keyword extraction, TF-IDF suffers from semantic blindness** - it treats "car" and "automobile" as completely different terms despite identical meaning.

KeyBERT addresses these limitations by replacing statistical counting with semantic similarity. Where TF-IDF might rank "algorithm" and "method" as unrelated terms, KeyBERT's BERT embeddings recognize their conceptual similarity and can prevent redundant selection through diversification.

### Semantic similarity and embeddings

**Semantic similarity measures how closely two pieces of text share meaning**, moving beyond lexical matching to conceptual understanding. Traditional approaches relied on word co-occurrence statistics or lexical resources like WordNet, but transformer-based embeddings provide superior semantic capture.

**BERT embeddings** create dense vector representations where semantically similar terms cluster in high-dimensional space. The key innovation is contextual awareness - the word "bank" receives different embeddings in "river bank" versus "savings bank", capturing meaning precisely based on surrounding context.

KeyBERT leverages this contextual understanding by comparing document embeddings with candidate keyword embeddings in the same semantic space. **Cosine similarity between these embeddings quantifies semantic relatedness**, enabling keyword selection based on meaning rather than just statistical frequency.

### Sentence transformers integration

**Sentence-BERT (SBERT) optimizes BERT for similarity tasks** through specialized training on sentence pairs. While standard BERT requires expensive inference for similarity comparison, SBERT generates meaningful sentence-level embeddings suitable for efficient similarity computation.

KeyBERT's default model `all-MiniLM-L6-v2` represents a sentence transformer trained specifically for semantic similarity tasks. **This model transforms both documents and keyword candidates into 384-dimensional vectors optimized for cosine similarity comparison.**

The integration enables KeyBERT's core workflow: encode the document once, encode all candidates once, then compute pairwise similarities efficiently. This architecture makes KeyBERT practical for real-world applications while maintaining semantic sophistication.

### Mathematical foundations of similarity

**Cosine similarity forms KeyBERT's core comparison mechanism**, measuring the cosine of the angle between two vectors in high-dimensional space:

```
cosine_similarity(A, B) = (A·B) / (||A|| × ||B||)
```

This measure ranges from 0 (orthogonal vectors, no similarity) to 1 (identical direction, maximum similarity). **Cosine similarity focuses on vector direction rather than magnitude**, making it ideal for normalized embeddings where semantic meaning matters more than raw activation strength.

The geometric interpretation reveals why cosine similarity works well for text: semantically similar texts point in similar directions in embedding space, regardless of length differences. A short phrase and long document about the same topic will have high cosine similarity despite vastly different vector magnitudes.

## Parameter interaction and optimization strategies

**KeyBERT's parameters interact in complex ways that affect both quality and performance.** Understanding these interactions enables effective optimization for specific use cases.

**N-gram range interacts critically with diversity settings.** Longer n-gram ranges like `(1,3)` generate more specific phrases but require higher diversity values to differentiate between similar multi-word expressions. A setting of `keyphrase_ngram_range=(3,3)` with `diversity=0.2` often produces repetitive results like "learning algorithm analyzes" and "learning algorithm generalize", while `diversity=0.7` yields more varied phrases.

**The relationship between candidates and computational efficiency** follows predictable patterns. Max Sum Similarity exhibits exponential complexity with candidate pool size, making `nr_candidates > 20` impractical. MMR scales quadratically, handling moderate candidate pools efficiently. **Pre-computing embeddings with `extract_embeddings()` enables parameter tuning without recalculation costs.**

**Model selection affects all other parameter decisions.** Smaller models like `all-MiniLM-L6-v2` process documents faster but may require different diversity settings than larger models like `all-mpnet-base-v2`. Multilingual models introduce additional complexity for stop word filtering and require language-appropriate parameter tuning.

**Optimization strategies depend on text characteristics and use case requirements.** Short documents (under 500 words) benefit from `keyphrase_ngram_range=(1,2)`, `diversity=0.5-0.7`, and efficient models. Long documents require splitting strategies, higher candidate pools, and lower diversity values to maintain relevance. Technical documents perform better with part-of-speech-based candidate extraction rather than simple n-grams.

## Practical implementation considerations

**Real-world KeyBERT deployment requires careful consideration of trade-offs** between accuracy, diversity, computational efficiency, and specific application requirements.

For **content creation and SEO applications**, moderate diversity (0.6) with balanced n-gram ranges `(1,2)` provides keyword variety suitable for content optimization. Academic research benefits from lower diversity (0.4) with extended ranges `(1,3)` to capture technical terminology while maintaining conceptual coherence.

**Performance optimization techniques** include pre-computing embeddings for parameter experimentation, using batch processing for multiple documents, and selecting embedding models based on speed versus quality requirements. Production environments typically favor MMR over Max Sum Distance due to superior computational efficiency.

The integration of KeyBERT with traditional keyword extraction methods creates powerful hybrid approaches. **Combining YAKE for candidate pre-filtering with KeyBERT for semantic ranking** leverages both statistical and semantic approaches, often producing superior results to either method alone.

Understanding these technical foundations enables practitioners to harness KeyBERT's full potential, creating keyword extraction systems that balance semantic sophistication with practical performance requirements across diverse applications and domains.