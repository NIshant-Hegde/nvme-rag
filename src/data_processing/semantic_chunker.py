import torch
import logging
from typing import List, Dict, Tuple
import numpy as np
from transformers import AutoTokenizer, AutoModel
import re

logger = logging.getLogger(__name__)

class SemanticChunker:
    """
    Implements semantic chunking instead of arbitrary sizes
    Groups semantically similar paragraphs into coherent chunks
    """
    
    def __init__(self, device: str = "auto", similarity_threshold: float = 0.7, max_chunk_length: int = 1500):
        self.device = self._setup_device(device)
        self.similarity_threshold = similarity_threshold
        self.max_chunk_length = max_chunk_length
        self.tokenizer = None
        self.embedding_model = None
        self._setup_models()
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup optimal device for processing"""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device)
    
    def _setup_models(self):
        """Initialize models for semantic processing"""
        try:
            # Use a lightweight model for semantic similarity
            model_name = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.embedding_model = AutoModel.from_pretrained(model_name)
            self.embedding_model.to(self.device)
            logger.info(f"Semantic models loaded on device: {self.device}")
        except Exception as e:
            logger.error(f"Failed to load semantic models: {e}")
            raise
    
    def extract_section_headers(self, markdown_text: str) -> List[Tuple[str, int, int]]:
        """Extract markdown headers with their positions and levels"""
        headers = []
        lines = markdown_text.split('\n')
        
        for i, line in enumerate(lines):
            stripped_line = line.strip()
            if stripped_line.startswith('#'):
                level = len(line) - len(line.lstrip('#'))
                header_text = stripped_line.lstrip('#').strip()
                if header_text:  # Only add non-empty headers
                    headers.append((header_text, i, level))
        
        return headers
    
    def _get_paragraph_embeddings(self, paragraphs: List[str]) -> np.ndarray:
        """Generate embeddings for paragraphs using sentence transformer"""
        embeddings = []
        
        with torch.no_grad():
            for paragraph in paragraphs:
                # Tokenize and encode
                inputs = self.tokenizer(
                    paragraph[:512],  # Limit length to prevent overflow
                    return_tensors="pt",
                    truncation=True,
                    padding=True
                )
                
                # Move inputs to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.embedding_model(**inputs)
                # Use mean pooling
                embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                embeddings.append(embedding[0])
        
        return np.array(embeddings)
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def semantic_chunking(self, text: str) -> List[Dict]:
        """
        Implement semantic chunking instead of arbitrary sizes
        Groups semantically similar paragraphs into coherent chunks
        """
        # Split into paragraphs and clean
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip() and len(p.strip()) > 20]
        
        if not paragraphs:
            return []
        
        logger.info(f"Processing {len(paragraphs)} paragraphs for semantic chunking")
        
        # Get embeddings for semantic similarity
        embeddings = self._get_paragraph_embeddings(paragraphs)
        
        # Group semantically similar paragraphs
        chunks = []
        current_chunk = []
        current_embedding = None
        
        for i, (paragraph, embedding) in enumerate(zip(paragraphs, embeddings)):
            if current_embedding is None:
                # Start first chunk
                current_chunk = [paragraph]
                current_embedding = embedding
            else:
                similarity = self._cosine_similarity(current_embedding, embedding)
                current_length = len(' '.join(current_chunk))
                
                # Check if paragraph should be added to current chunk
                if (similarity >= self.similarity_threshold and 
                    current_length + len(paragraph) < self.max_chunk_length):
                    current_chunk.append(paragraph)
                    # Update embedding to running average
                    current_embedding = (current_embedding + embedding) / 2
                else:
                    # Finalize current chunk
                    if current_chunk:
                        chunk_content = '\n\n'.join(current_chunk)
                        chunks.append({
                            'content': chunk_content,
                            'paragraph_count': len(current_chunk),
                            'semantic_density': float(similarity) if similarity else 1.0,  # Convert to Python float
                            'character_length': len(chunk_content)  # Use actual joined content length
                        })
                    
                    # Start new chunk
                    current_chunk = [paragraph]
                    current_embedding = embedding
        
        # Add final chunk
        if current_chunk:
            chunk_content = '\n\n'.join(current_chunk)
            chunks.append({
                'content': chunk_content,
                'paragraph_count': len(current_chunk),
                'semantic_density': 1.0,
                'character_length': len(chunk_content)  # Use actual joined content length
            })
        
        logger.info(f"Created {len(chunks)} semantic chunks")
        return chunks
    
    def assign_section_context(self, chunks: List[Dict], headers: List[Tuple[str, int, int]], 
                              original_text: str) -> List[Dict]:
        """Assign section headers to chunks based on position"""
        
        for chunk in chunks:
            # Find chunk position in original text more reliably
            chunk_preview = chunk['content'][:min(50, len(chunk['content']))]
            chunk_start = original_text.find(chunk_preview)
            
            if chunk_start == -1:
                # Fallback: try to find by first sentence
                first_sentence = chunk['content'].split('.')[0] + '.'
                chunk_start = original_text.find(first_sentence)
                
            if chunk_start == -1:
                # Last fallback: try first few words
                first_words = ' '.join(chunk['content'].split()[:5])
                chunk_start = original_text.find(first_words)
            
            chunk_line = original_text[:chunk_start].count('\n') if chunk_start != -1 else 0
            
            # Find the most recent header that appears before this chunk
            relevant_header = "Document Root"
            header_level = 0
            
            # Sort headers by line number to ensure we process them in order
            sorted_headers = sorted(headers, key=lambda x: x[1])
            
            for header_text, header_line, level in sorted_headers:
                if header_line < chunk_line:  # Header must appear before the chunk
                    relevant_header = header_text
                    header_level = level
                else:
                    break  # Headers are sorted, so we can stop here
            
            chunk['section_header'] = relevant_header
            chunk['section_level'] = header_level
        
        return chunks