import logging
from pathlib import Path
from typing import Tuple, Dict, List
import hashlib
from datetime import datetime

# PDF Processing imports
import pymupdf4llm
try:
    from marker import convert_single_pdf
    MARKER_AVAILABLE = True
except ImportError:
    MARKER_AVAILABLE = False
    logging.warning("Marker not available, falling back to PyMuPDF only")

logger = logging.getLogger(__name__)

class PDFProcessor:
    """
    Handles PDF text extraction using multiple methods
    Priority: Marker -> PyMuPDF4LLM -> OCR fallback
    """
    
    def __init__(self):
        self.extraction_methods = ["marker", "pymupdf4llm"]
        if not MARKER_AVAILABLE:
            self.extraction_methods = ["pymupdf4llm"]
    
    def calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file for versioning"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def extract_with_marker(self, pdf_path: Path) -> Tuple[str, Dict]:
        """
        Extract PDF using Marker for clean markdown with metadata
        Marker preserves structure, handles tables, and maintains formatting
        """
        if not MARKER_AVAILABLE:
            return "", {"success": False, "error": "Marker not available"}
        
        try:
            logger.info(f"Attempting Marker extraction for {pdf_path}")
            full_text, images, metadata = convert_single_pdf(
                str(pdf_path),
                max_pages=None,
                langs=["en"],
                batch_multiplier=2
            )
            
            extraction_info = {
                "extraction_method": "marker",
                "has_images": len(images) > 0,
                "image_count": len(images),
                "has_tables": self._detect_tables(full_text),
                "success": True,
                "metadata": metadata,
                "text_length": len(full_text)
            }
            
            logger.info("Marker extraction successful")
            return full_text, extraction_info
            
        except Exception as e:
            logger.warning(f"Marker extraction failed: {e}")
            return "", {"success": False, "error": str(e)}
    
    def extract_with_pymupdf(self, pdf_path: Path) -> Tuple[str, Dict]:
        """Fallback extraction using PyMuPDF4LLM"""
        try:
            logger.info(f"Attempting PyMuPDF extraction for {pdf_path}")
            md_text = pymupdf4llm.to_markdown(str(pdf_path))
            
            extraction_info = {
                "extraction_method": "pymupdf4llm",
                "has_images": "![" in md_text,
                "has_tables": self._detect_tables(md_text),
                "success": True,
                "text_length": len(md_text)
            }
            
            logger.info("PyMuPDF extraction successful")
            return md_text, extraction_info
            
        except Exception as e:
            logger.warning(f"PyMuPDF extraction failed: {e}")
            return "", {"success": False, "error": str(e)}
    
    def extract_text(self, pdf_path: Path) -> Tuple[str, Dict]:
        """
        Main extraction method that tries multiple approaches
        Returns the best available extraction
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Try Marker first (best quality) if available
        if MARKER_AVAILABLE:
            text, info = self.extract_with_marker(pdf_path)
            if info.get("success", False) and text.strip():
                return text, info
        
        # Fallback to PyMuPDF
        text, info = self.extract_with_pymupdf(pdf_path)
        if info.get("success", False) and text.strip():
            return text, info
        
        raise RuntimeError(f"All extraction methods failed for {pdf_path}")
    
    def _detect_tables(self, text: str) -> bool:
        """Simple table detection in markdown text"""
        lines = text.strip().split('\n')
        table_lines = 0
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('|') and stripped.endswith('|') and '|' in stripped[1:-1]:
                table_lines += 1
        return table_lines >= 2  # At least header + one data row
    
    def get_page_count(self, pdf_path: Path) -> int:
        """Get page count from PDF"""
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(pdf_path)
            page_count = doc.page_count
            doc.close()
            return page_count
        except Exception as e:
            logger.warning(f"Could not get page count: {e}")
            return 0