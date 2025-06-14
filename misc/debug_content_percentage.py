#!/usr/bin/env python3
"""
Debug script to test content percentage calculation
"""

import sys
sys.path.append('.')

from src.pipeline.integration import RAGPipelineIntegration
from src.vector_store.embedding_generator import EmbeddingConfig
from src.llm.ollama_client import OllamaConfig
from src.llm.answer_generator import AnswerGenerationConfig, AnswerStyle
from src.retrieval.retrieval_pipeline import RetrievalConfig, RetrievalStrategy
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
    print("Testing RAG pipeline with debug content percentage calculation...")
    
    # Initialize with simpler config for testing
    embedding_config = EmbeddingConfig(model_name='sentence-transformers/multi-qa-MiniLM-L6-cos-v1', device='cpu')
    ollama_config = OllamaConfig(base_url='http://localhost:11434', model='gemma3:12b-it-qat', temperature=0.1, max_tokens=512)
    retrieval_config = RetrievalConfig(strategy=RetrievalStrategy.SEMANTIC_ONLY, top_k=3, enable_query_enhancement=False, enable_context_filtering=False, max_context_length=1500)
    answer_config = AnswerGenerationConfig(style=AnswerStyle.CONCISE, max_answer_length=300, include_sources=True, include_confidence=True, cite_sections=True)

    rag_pipeline = RAGPipelineIntegration('data/vector_store', embedding_config, ollama_config, retrieval_config, answer_config)
    
    # Test question
    question = "What is the NVMe submission queue?"
    print(f"Question: {question}")
    
    # First get retrieval results to see what chunks are retrieved
    print("\n=== TESTING RETRIEVAL ===")
    search_result = rag_pipeline.search_and_retrieve(question)
    print(f"Retrieved {len(search_result['chunks'])} chunks")
    
    if search_result['chunks']:
        for i, chunk in enumerate(search_result['chunks'][:2]):
            print(f"\nChunk {i+1}:")
            print(f"  Section: {chunk['section_header']}")
            print(f"  Content: {chunk['content'][:150]}...")
    
    # Now test full QA
    print("\n=== TESTING QA PIPELINE ===")
    try:
        qa_result = rag_pipeline.ask_question(question)
        print(f"QA completed successfully!")
        print(f"Answer: {qa_result.generated_answer.answer}")
        print(f"Context chunks used: {qa_result.generated_answer.context_used}")
        
        # Debug the content percentage calculation
        debug_content_percentage(qa_result.generated_answer.answer, search_result['chunks'])
        
        print(f"\nOriginal percentage calculation:")
        print(f"  Chunk content: {qa_result.generated_answer.chunk_content_percentage:.1f}%")
        print(f"  LLM generated: {qa_result.generated_answer.llm_generated_percentage:.1f}%")
        
    except Exception as e:
        print(f"QA failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()