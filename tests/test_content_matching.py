#!/usr/bin/env python3
"""
Test content matching with a simulated answer that should show chunk content
"""

import sys
sys.path.append('.')

from src.vector_store.chroma_store import ChromaVectorStore
from src.vector_store.embedding_generator import EmbeddingConfig
from src.vector_store.base import SearchQuery
import re
from difflib import SequenceMatcher

def debug_content_percentage(answer, context_chunks):
    """Debug version of content percentage calculation"""
    print(f"\n=== DEBUG CONTENT PERCENTAGE CALCULATION ===")
    print(f"Answer length: {len(answer)} chars")
    print(f"Number of context chunks: {len(context_chunks)}")
    
    if not answer or not context_chunks:
        print("No answer or chunks - returning 0%, 100%")
        return 0.0, 100.0
    
    # Combine all chunk content  
    chunk_text = " ".join([chunk.content for chunk in context_chunks])
    print(f"Combined chunk text length: {len(chunk_text)} chars")
    
    # Clean and normalize text for comparison
    def clean_text(text):
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip().lower())
        # Remove common punctuation that might differ
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    answer_clean = clean_text(answer)
    chunk_text_clean = clean_text(chunk_text)
    
    print(f"Answer clean preview: {answer_clean[:200]}...")
    print(f"Chunk text clean preview: {chunk_text_clean[:200]}...")
    
    if not answer_clean:
        print("Empty clean answer - returning 0%, 100%")
        return 0.0, 100.0
    
    # Method 1: Find direct text overlaps using sliding window
    answer_words = answer_clean.split()
    chunk_words = chunk_text_clean.split()
    
    print(f"Answer words: {len(answer_words)}")
    print(f"Chunk words: {len(chunk_words)}")
    
    if not chunk_words:
        print("No chunk words - returning 0%, 100%")
        return 0.0, 100.0
    
    matched_words = set()
    window_sizes = [8, 6, 4, 3, 2]  # Different phrase lengths to check
    
    for window_size in window_sizes:
        matches_found = 0
        for i in range(len(answer_words) - window_size + 1):
            answer_phrase = ' '.join(answer_words[i:i + window_size])
            
            # Check if this phrase exists in chunk text
            if answer_phrase in chunk_text_clean:
                for j in range(i, i + window_size):
                    matched_words.add(j)
                matches_found += 1
        
        print(f"Window size {window_size}: {matches_found} matches found")
    
    # Method 2: Use sequence matching for additional coverage
    matcher = SequenceMatcher(None, answer_clean, chunk_text_clean)
    matching_blocks = matcher.get_matching_blocks()
    
    print(f"Sequence matcher found {len(matching_blocks)} matching blocks")
    substantial_blocks = [block for block in matching_blocks if block.size >= 20]
    print(f"Substantial blocks (>=20 chars): {len(substantial_blocks)}")
    
    # Count characters that match in substantial blocks (min 20 chars)
    char_matches = sum(block.size for block in matching_blocks if block.size >= 20)
    sequence_match_ratio = char_matches / len(answer_clean) if answer_clean else 0
    
    print(f"Character matches (>=20 chars): {char_matches}")
    print(f"Sequence match ratio: {sequence_match_ratio:.3f}")
    
    # Combine both methods
    word_match_ratio = len(matched_words) / len(answer_words) if answer_words else 0
    print(f"Word match ratio: {word_match_ratio:.3f} ({len(matched_words)}/{len(answer_words)})")
    
    # Weight the methods (word matching is more precise, sequence matching catches missed cases)
    chunk_percentage = min(100.0, (word_match_ratio * 70 + sequence_match_ratio * 30) * 100)
    llm_percentage = max(0.0, 100.0 - chunk_percentage)
    
    print(f"Final chunk percentage: {chunk_percentage:.1f}%")
    print(f"Final LLM percentage: {llm_percentage:.1f}%")
    
    return chunk_percentage, llm_percentage

def main():
    print("Testing content matching with simulated scenario...")
    
    # Get the relevant chunks
    embedding_config = EmbeddingConfig(model_name='sentence-transformers/multi-qa-MiniLM-L6-cos-v1', device='cpu')
    vector_store = ChromaVectorStore(collection_name='nvme_rag_chunks', persist_directory='data/vector_store', embedding_config=embedding_config)
    
    # Search for submission queue content
    query = SearchQuery(query_text='submission queue SQ circular buffer', top_k=3)
    results = vector_store.search(query)
    
    print(f"Retrieved {len(results)} chunks")
    
    # Convert to the format expected by the content percentage calculation
    context_chunks = []
    for result in results:
        context_chunks.append(result.chunk)
        print(f"\\nChunk content: {result.chunk.content[:300]}...")
    
    # Simulate three different types of answers
    test_cases = [
        {
            "name": "Answer that copies from chunks",
            "answer": "A Submission Queue (SQ) is a circular buffer with a fixed slot size that the host software uses to submit commands for execution by the controller."
        },
        {
            "name": "Answer that paraphrases chunks", 
            "answer": "The submission queue is a data structure used by the host to send commands to the NVMe controller. It operates as a circular buffer where commands are placed for processing."
        },
        {
            "name": "Answer with no chunk content",
            "answer": "Submission queues are important components of the NVMe protocol that facilitate efficient command processing and improve overall system performance through optimized data flow."
        }
    ]
    
    for test_case in test_cases:
        print(f"\\n{'='*80}")
        print(f"TEST CASE: {test_case['name']}")
        print(f"{'='*80}")
        print(f"Answer: {test_case['answer']}")
        
        chunk_pct, llm_pct = debug_content_percentage(test_case['answer'], context_chunks)
        
        print(f"\\nRESULT: {chunk_pct:.1f}% chunk content, {llm_pct:.1f}% LLM generated")

if __name__ == "__main__":
    main()