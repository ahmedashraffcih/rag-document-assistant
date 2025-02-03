import arabic_reshaper
import unicodedata
import re
import fitz  # PyMuPDF


def extract_arabic_text(pdf_path):
    """Extract Arabic text from a PDF using PyMuPDF and fix encoding issues."""
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text("text") for page in doc])

    # Apply Arabic text fixing
    fixed_text = fix_arabic_text(text)

    return fixed_text


def fix_arabic_text(text):
    """Fix and normalize Arabic text for correct processing."""
    text = unicodedata.normalize("NFKC", text)  # Fix Unicode inconsistencies
    text = re.sub(r"\s+", " ", text)  # Remove excessive spaces
    return arabic_reshaper.reshape(text)  # Ensure proper letter formation
