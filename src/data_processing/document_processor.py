import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from src.models.document import DocumentMetadata, ProcessedChunk, ProcessingResult
from src.data_processing.pdf_processor import PDFProcessor
from src.data_processing.semantic_chunker import SemanticChunker

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Main document processing pipeline that orchestrates:
    - PDF text extraction
    - Semantic chunking
    - Metadata generation
    - Structure preservation
    """
    
    def __init__(self, device: str = "auto", similarity_threshold: float = 0.7):
        self.pdf_processor = PDFProcessor()
        self.semantic_chunker = SemanticChunker(
            device=device, 
            similarity_threshold=similarity_threshold
        )
        logger.info("Document processor initialized")
    
    def process_document(self, pdf_path: Path) -> ProcessingResult:
        """
        Main method to process PDF following RAG best practices
        Returns structured chunks with proper metadata
        """
        logger.info(f"Starting document processing: {pdf_path}")
        start_time = datetime.now()
        
        # Step 1: Extract text and basic info
        extracted_text, extraction_info = self.pdf_processor.extract_text(pdf_path)
        
        # Step 2: Calculate document metadata
        file_hash = self.pdf_processor.calculate_file_hash(pdf_path)
        page_count = self.pdf_processor.get_page_count(pdf_path)
        
        doc_metadata = DocumentMetadata(
            source_path=str(pdf_path),
            document_type="pdf",
            page_count=page_count,
            processing_timestamp=datetime.now().isoformat(),
            file_hash=file_hash,
            extraction_method=extraction_info["extraction_method"],
            has_images=extraction_info.get("has_images", False),
            has_tables=extraction_info.get("has_tables", False)
        )
        
        # Step 3: Extract document structure
        headers = self.semantic_chunker.extract_section_headers(extracted_text)
        
        # Step 4: Perform semantic chunking
        semantic_chunks = self.semantic_chunker.semantic_chunking(extracted_text)
        
        # Step 5: Assign section context
        contextualized_chunks = self.semantic_chunker.assign_section_context(
            semantic_chunks, headers, extracted_text
        )
        
        # Step 6: Create ProcessedChunk objects
        processed_chunks = []
        doc_id = file_hash[:12]  # Use first 12 chars of hash as doc ID
        
        for i, chunk_data in enumerate(contextualized_chunks):
            chunk_id = f"{doc_id}_{i:04d}"
            
            processed_chunk = ProcessedChunk(
                content=chunk_data['content'],
                metadata={
                    "source": str(pdf_path),
                    "extraction_method": extraction_info["extraction_method"],
                    "paragraph_count": chunk_data['paragraph_count'],
                    "character_length": chunk_data['character_length'],
                    "processing_timestamp": doc_metadata.processing_timestamp,
                    "section_level": chunk_data.get('section_level', 0)
                },
                chunk_id=chunk_id,
                parent_doc_id=doc_id,
                section_header=chunk_data['section_header'],
                page_number=0,  # Would need more sophisticated page tracking
                chunk_type="text",
                semantic_density=chunk_data['semantic_density']
            )
            
            processed_chunks.append(processed_chunk)
        
        # Step 7: Calculate processing statistics
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        processing_stats = {
            "processing_time_seconds": float(processing_time),  # Ensure it's a Python float
            "total_chunks": int(len(processed_chunks)),
            "total_characters": int(len(extracted_text)),
            "average_chunk_size": float(sum(chunk.metadata["character_length"] for chunk in processed_chunks) / len(processed_chunks)) if processed_chunks else 0.0,
            "headers_found": int(len(headers)),
            "extraction_method": str(extraction_info["extraction_method"]),
            "device_used": str(self.semantic_chunker.device)
        }
        
        logger.info(f"Successfully processed {len(processed_chunks)} chunks in {processing_time:.2f}s")
        
        return ProcessingResult(
            chunks=processed_chunks,
            metadata=doc_metadata,
            processing_stats=processing_stats
        )
    
    def process_multiple_documents(self, pdf_paths: List[Path], 
                                 output_dir: Path = None) -> Dict[str, ProcessingResult]:
        """Process multiple documents and optionally save results"""
        results = {}
        
        for pdf_path in pdf_paths:
            try:
                result = self.process_document(pdf_path)
                doc_id = result.metadata.file_hash[:12]
                results[doc_id] = result
                
                if output_dir:
                    result.save_to_files(output_dir / "processed")
                    
            except Exception as e:
                logger.error(f"Failed to process {pdf_path}: {e}")
                continue
        
        return results
    
    def get_processing_summary(self, results: Dict[str, ProcessingResult]) -> Dict[str, Any]:
        """Generate summary statistics for processed documents"""
        if not results:
            return {}
        
        total_docs = len(results)
        total_chunks = sum(len(result.chunks) for result in results.values())
        total_processing_time = sum(result.processing_stats["processing_time_seconds"] for result in results.values())
        
        extraction_methods = {}
        for result in results.values():
            method = result.processing_stats["extraction_method"]
            extraction_methods[method] = extraction_methods.get(method, 0) + 1
        
        return {
            "total_documents": total_docs,
            "total_chunks": total_chunks,
            "total_processing_time_seconds": total_processing_time,
            "average_chunks_per_doc": total_chunks / total_docs,
            "extraction_methods_used": extraction_methods,
            "documents_processed": list(results.keys())
        }