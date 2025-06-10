#!/usr/bin/env python3
"""
Processes the NVME base specification following RAG best practices
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_processing.document_processor import DocumentProcessor
from src.models.document import ProcessingResult

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/processing.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Main processing function"""
    
    # Configuration
    PDF_PATH = Path("data/raw/nvme-base-spec.pdf")  #input path
    OUTPUT_DIR = Path("data/processed")  #path where the metadata, chunks and stats JSON files will be saved

    # Ensure directories exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    # Check if PDF exists
    if not PDF_PATH.exists():
        logger.error(f"PDF file not found: {PDF_PATH}")
        logger.info("Please ensure the NVME spec PDF is placed in data/raw/")
        return 1
    
    try:
        # Initialize processor
        logger.info("Initializing document processor...")
        processor = DocumentProcessor(
            device="auto",  # Will auto-detect best device
            similarity_threshold=0.7  # Adjust based on your needs, TODO - Experiment with the similarity threshold values
        )
        
        # Process the document
        logger.info(f"Processing NVME specification: {PDF_PATH}")
        result = processor.process_document(PDF_PATH)
        
        # Save results
        logger.info("Saving processed data...")
        result.save_to_files(OUTPUT_DIR)
        
        # Print summary
        print("\n" + "="*60)
        print("PROCESSING SUMMARY")
        print("="*60)
        print(f"Document: {PDF_PATH.name}")
        print(f"Total chunks created: {len(result.chunks)}")
        print(f"Extraction method: {result.processing_stats['extraction_method']}")
        print(f"Processing time: {result.processing_stats['processing_time_seconds']:.2f}s")
        print(f"Average chunk size: {result.processing_stats['average_chunk_size']:.0f} characters")
        print(f"Headers found: {result.processing_stats['headers_found']}")
        print(f"Device used: {result.processing_stats['device_used']}")
        print(f"Has images: {result.metadata.has_images}")
        print(f"Has tables: {result.metadata.has_tables}")
        
        # Show sample chunks
        print("\n" + "-"*40)
        print("SAMPLE CHUNKS")
        print("-"*40)
        for i, chunk in enumerate(result.chunks[:3]):
            print(f"\nChunk {i+1} ({chunk.section_header}):")
            print(f"Content preview: {chunk.content[:200]}...")
            print(f"Semantic density: {chunk.semantic_density:.3f}")
        
        print(f"\nData saved to: {OUTPUT_DIR}")
        print("Processing completed successfully!")
        
        return 0
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)