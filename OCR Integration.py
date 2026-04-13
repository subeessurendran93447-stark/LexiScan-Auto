import fitz  # PyMuPDF
import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path
import re
import os

class LexiScanProcessor:
    def __init__(self, tesseract_path=None):
        # Update this path if Tesseract is not in your System Environment Variables
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path

    def is_scanned_pdf(self, pdf_path):
        """Checks if PDF needs OCR."""
        try:
            with fitz.open(pdf_path) as doc:
                text_content = "".join([page.get_text() for page in doc])
            return len(text_content.strip()) < 100
        except Exception:
            return True 

    def preprocess_image(self, pil_img):
        """Optimizes 2-page scans for OCR accuracy."""
        img = np.array(pil_img.convert('RGB'))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # 1. Upscale for small legal fonts
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
        
        # 2. Denoise and sharpen
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        
        # 3. Adaptive thresholding for uneven scans/shadows
        binary = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        return binary

    def extract_text(self, pdf_path):
        """The main extraction engine."""
        if not self.is_scanned_pdf(pdf_path):
            print(f"-> Extracting digital text from {os.path.basename(pdf_path)}...")
            with fitz.open(pdf_path) as doc:
                return "\n".join([page.get_text() for page in doc])

        print(f"-> Running OCR Pipeline for {os.path.basename(pdf_path)}...")
        pages = convert_from_path(pdf_path, dpi=300, first_page=1, last_page=2)
        full_text = []
        for page_img in pages:
            processed_img = self.preprocess_image(page_img)
            text = pytesseract.image_to_string(processed_img, config='--oem 3 --psm 3')
            full_text.append(text)

        return self.clean_text("\n".join(full_text))

    def clean_text(self, text):
        """Post-processing for clean NER results."""
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
        text = re.sub(r'[|○_•*]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

# ==========================================
# MAIN EXECUTION BLOCK (Auto-Detect Files)
# ==========================================
if __name__ == "__main__":
    # Path where your contracts are stored
    TARGET_DIR = r"D:\LexiScan Auto"
    
    # Initialize the processor
    # If you get a Tesseract error, put your full path here:
    # processor = LexiScanProcessor(tesseract_path=r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    processor = LexiScanProcessor()

    print(f"--- LexiScan Auto: Scanning Directory {TARGET_DIR} ---")
    
    try:
        # Get all files and filter for PDFs
        all_files = os.listdir(TARGET_DIR)
        pdf_files = [f for f in all_files if f.lower().endswith(".pdf")]

        if not pdf_files:
            print("[ALERT] No PDF files found to process.")
        else:
            print(f"[FOUND] {len(pdf_files)} PDF(s). Starting first document...")
            
            # Select the first PDF (handles contract_sample.pdf.pdf automatically)
            target_file = pdf_files[0]
            full_path = os.path.join(TARGET_DIR, target_file)
            
            # RUN EXTRACTION
            result_text = processor.extract_text(full_path)
            
            # DISPLAY OUTPUT
            print("\n" + "="*50)
            print(f"SUCCESS: TEXT EXTRACTED FROM {target_file}")
            print("="*50)
            print(result_text[:1200]) # Preview
            print("="*50)
            
    except Exception as e:
        print(f"[CRITICAL ERROR]: {e}")