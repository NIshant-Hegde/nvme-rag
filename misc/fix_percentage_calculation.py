#!/usr/bin/env python3
"""
Test the exact calculation to find the bug
"""

import re
from difflib import SequenceMatcher

def debug_content_percentage_step_by_step(answer, context_chunks):
    """Debug version that shows each step clearly"""
    print(f"\n=== STEP BY STEP DEBUG ===")
    print(f"Answer: '{answer}'")
    print(f"Answer length: {len(answer)} chars")
    print(f"Number of context chunks: {len(context_chunks)}")
    
    if not answer or not context_chunks:
        print("Early return: No answer or chunks - returning 0%, 100%")
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
    
    print(f"\\nCLEANED TEXTS:")
    print(f"Answer clean: '{answer_clean}'")
    print(f"Chunk text clean (first 200 chars): '{chunk_text_clean[:200]}'")
    
    if not answer_clean:
        print("Early return: Empty clean answer - returning 0%, 100%")
        return 0.0, 100.0
    
    # Method 1: Find direct text overlaps using sliding window
    answer_words = answer_clean.split()
    chunk_words = chunk_text_clean.split()
    
    print(f"\\nWORD ANALYSIS:")
    print(f"Answer words: {len(answer_words)} - {answer_words}")
    print(f"Chunk words: {len(chunk_words)} (showing first 20: {chunk_words[:20]})")
    
    if not chunk_words:
        print("Early return: No chunk words - returning 0%, 100%")
        return 0.0, 100.0
    
    matched_words = set()
    window_sizes = [8, 6, 4, 3, 2]  # Different phrase lengths to check
    
    print(f"\\nSLIDING WINDOW MATCHING:")
    for window_size in window_sizes:
        matches_found = 0
        for i in range(len(answer_words) - window_size + 1):
            answer_phrase = ' '.join(answer_words[i:i + window_size])
            
            # Check if this phrase exists in chunk text
            if answer_phrase in chunk_text_clean:
                print(f"  MATCH found for window {window_size}: '{answer_phrase}'")
                for j in range(i, i + window_size):
                    matched_words.add(j)
                matches_found += 1
        
        print(f"Window size {window_size}: {matches_found} matches found")
    
    print(f"\\nTotal matched word indices: {sorted(matched_words)}")
    
    # Method 2: Use sequence matching for additional coverage
    print(f"\\nSEQUENCE MATCHING:")
    matcher = SequenceMatcher(None, answer_clean, chunk_text_clean)
    matching_blocks = matcher.get_matching_blocks()
    
    print(f"Total matching blocks: {len(matching_blocks)}")
    for i, block in enumerate(matching_blocks):
        if block.size >= 5:  # Show substantial blocks
            matched_text = answer_clean[block.a:block.a+block.size]
            print(f"  Block {i}: size={block.size}, text='{matched_text}'")
    
    # Count characters that match in substantial blocks (min 20 chars)
    char_matches = sum(block.size for block in matching_blocks if block.size >= 20)
    sequence_match_ratio = char_matches / len(answer_clean) if answer_clean else 0
    
    print(f"\\nSEQUENCE MATCH RESULTS:")
    print(f"Character matches (>=20 chars): {char_matches}")
    print(f"Sequence match ratio: {sequence_match_ratio:.6f}")
    
    # Combine both methods
    word_match_ratio = len(matched_words) / len(answer_words) if answer_words else 0
    print(f"\\nFINAL CALCULATION:")
    print(f"Word match ratio: {word_match_ratio:.6f} ({len(matched_words)}/{len(answer_words)})")
    print(f"Sequence match ratio: {sequence_match_ratio:.6f}")
    
    # Weight the methods (word matching is more precise, sequence matching catches missed cases)
    combined_score = word_match_ratio * 70 + sequence_match_ratio * 30
    print(f"Combined score (before *100): {combined_score:.6f}")
    print(f"Combined score * 100: {combined_score * 100:.6f}")
    
    chunk_percentage = min(100.0, combined_score)
    llm_percentage = max(0.0, 100.0 - chunk_percentage)
    
    print(f"\\nFINAL RESULTS:")
    print(f"Chunk percentage: {chunk_percentage:.6f}%")
    print(f"LLM percentage: {llm_percentage:.6f}%")
    
    return chunk_percentage, llm_percentage

# Test with a simple example
class MockChunk:
    def __init__(self, content):
        self.content = content

def main():
    # Create mock chunks
    chunks = [
        MockChunk("A Submission Queue (SQ) is a circular buffer with a fixed slot size that the host software uses to submit commands for execution by the controller."),
        MockChunk("This section applies only to Submission Queues that use SQ flow control."),
        MockChunk("SQ flow control is enabled and shall be used for a created queue pair.")
    ]
    
    # Test case that should show low chunk percentage
    answer = "Submission queues are important components of the NVMe protocol that facilitate efficient command processing and improve overall system performance through optimized data flow."
    
    chunk_pct, llm_pct = debug_content_percentage_step_by_step(answer, chunks)
    
    print(f"\\nFINAL RESULT: {chunk_pct:.1f}% chunk content, {llm_pct:.1f}% LLM generated")

if __name__ == "__main__":
    main()