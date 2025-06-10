# phase2_demo.py

from src.pipeline.integration import RAGPipelineIntegration
from src.vector_store.embedding_generator import EmbeddingConfig
from src.llm.ollama_client import OllamaConfig
from src.retrieval.retrieval_pipeline import RetrievalConfig, RetrievalStrategy


def main():
    # --- Configuration ---
    vector_store_path = "data/vector_store"  # path to the vector store

    # Embedding and LLM config
    embedding_config = EmbeddingConfig(
        model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1",  # TODO: experiment with different models
        device="cpu"
    )
    ollama_config = OllamaConfig(
        base_url="http://localhost:11434",  # Or your Ollama server
        model="gemma3:12b-it-qat"                  # TODO: experiment with different models
    )
    retrieval_config = RetrievalConfig(
        strategy=RetrievalStrategy.SEMANTIC_ONLY,  # TODO: experiment with different retrieval strategies
        max_context_length=3000             # TODO: experiment with different max context lengths
    )

    # --- Initialize the pipeline ---
    rag_pipeline = RAGPipelineIntegration(
        vector_store_path=vector_store_path,
        embedding_config=embedding_config,
        ollama_config=ollama_config,
        retrieval_config=retrieval_config
    )

    print("NVMe Spec RAG Demo")
    print("==================")
    print("Type your question about the NVMe spec (or press Enter to use a sample):")
    user_query = input("> ").strip()
    if not user_query:
        user_query = "What is the purpose of the NVMe submission queue?"   #TODO: experiment with different queries based on this query's performance

    print(f"\nQuerying: {user_query}\n")

    # --- Retrieve context ---
    result = rag_pipeline.search_and_retrieve(query=user_query)

    if result.get("success"):
        print("Top relevant context chunks:\n")
        for i, chunk in enumerate(result["chunks"], 1):
            print(f"[{i}] Section: {chunk.get('section_header', 'N/A')}")
            #print(chunk.get("content", "")[:500])  # Print first 500 chars
            print(chunk.get("content", ""))  # Print full content
            print("-" * 40)
        print("\nRetrieval stats:", result["stats"])
    else:
        print("Error during retrieval:", result.get("error"))

if __name__ == "__main__":
    main() 