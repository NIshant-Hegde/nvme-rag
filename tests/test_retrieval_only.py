#!/usr/bin/env python3
"""
Test retrieval only to see what chunks are retrieved
"""

import sys
sys.path.append('.')

from src.pipeline.integration import RAGPipelineIntegration
from src.vector_store.embedding_generator import EmbeddingConfig
from src.llm.ollama_client import OllamaConfig
from src.llm.answer_generator import AnswerGenerationConfig, AnswerStyle
from src.retrieval.retrieval_pipeline import RetrievalConfig, RetrievalStrategy

def main():
    print("Testing retrieval to see what chunks are being retrieved...")
    
    # Initialize with simpler config for testing
    embedding_config = EmbeddingConfig(model_name='sentence-transformers/multi-qa-MiniLM-L6-cos-v1', device='cpu')
    ollama_config = OllamaConfig(base_url='http://localhost:11434', model='gemma3:12b-it-qat', temperature=0.1, max_tokens=512)
    retrieval_config = RetrievalConfig(strategy=RetrievalStrategy.SEMANTIC_ONLY, top_k=5, enable_query_enhancement=False, enable_context_filtering=False, max_context_length=2000)
    answer_config = AnswerGenerationConfig(style=AnswerStyle.CONCISE, max_answer_length=300, include_sources=True, include_confidence=True, cite_sections=True)

    rag_pipeline = RAGPipelineIntegration('data/vector_store', embedding_config, ollama_config, retrieval_config, answer_config)
    
    # Test questions
    questions = [
        "What is the NVMe submission queue?",
        "How does NVMe command processing work?",
        "What are NVMe queues?"
    ]
    
    for question in questions:
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print(f"{'='*60}")
        
        # Get retrieval results
        search_result = rag_pipeline.search_and_retrieve(question)
        print(f"Retrieved {len(search_result['chunks'])} chunks")
        
        if search_result['chunks']:
            for i, chunk in enumerate(search_result['chunks']):
                print(f"\nChunk {i+1}:")
                print(f"  Section: {chunk['section_header']}")
                print(f"  Page: {chunk['page_number']}")
                print(f"  Content length: {len(chunk['content'])} chars")
                print(f"  Content preview: {chunk['content'][:200]}...")
                
                # Check if content contains question keywords
                question_words = question.lower().split()
                chunk_content_lower = chunk['content'].lower()
                matching_words = [word for word in question_words if word in chunk_content_lower]
                print(f"  Matching question words: {matching_words}")
        else:
            print("No chunks retrieved!")

if __name__ == "__main__":
    main()