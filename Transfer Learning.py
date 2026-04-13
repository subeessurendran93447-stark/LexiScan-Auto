import fitz  # PyMuPDF
import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path
import re
import os
import spacy

class LexiScanHighFidelity:
    def __init__(self, model_path=None, tesseract_path=None):
        """
        Initializes OCR and the 3-Tier NER Fallback system.
        """
        # 1. Setup Tesseract
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            
        # 2. Setup NLP Model
        self.nlp = None
        
        # Tier 1: Custom Model
        if model_path and os.path.exists(model_path):
            try:
                self.nlp = spacy.load(model_path)
                print(f"[LOADED] Custom Model: {model_path}")
            except: pass

        # Tier 2: Transformer (High F1)
        if not self.nlp:
            try:
                self.nlp = spacy.load("en_core_web_trf")
                print("[LOADED] Transformer Fallback (en_core_web_trf)")
            except:
                print("[SKIP] Transformer model not available.")

        # Tier 3: Small Model (Reliable Safety Net)
        if not self.nlp:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                print("[LOADED] Small Model Fallback (en_core_web_sm)")
            except:
                print("[CRITICAL] Run: python -m spacy download en_core_web_sm")

    def preprocess_image(self, pil_img):
        """Pre-OCR image enhancement."""
        img = np.array(pil_img.convert('RGB'))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        return cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)

    def clean_text(self, text):
        """Post-OCR noise reduction."""
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
        text = re.sub(r'[|○_•*]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def extract_text(self, pdf_path):
        """Main triage for Digital vs Scanned PDF."""
        # Check if PDF has embedded text
        with fitz.open(pdf_path) as doc:
            embedded_text = "".join([p.get_text() for p in doc]).strip()
            is_scanned = len(embedded_text) < 100

        if not is_scanned:
            print(f"-> Extracting digital text...")
            return embedded_text

        # If scanned, run OCR
        print(f"-> Running OCR Pipeline...")
        pages = convert_from_path(pdf_path, dpi=300, first_page=1, last_page=2)
        full_text = []
        for page_img in pages:
            processed = self.preprocess_image(page_img)
            full_text.append(pytesseract.image_to_string(processed, config='--oem 3 --psm 3'))
        
        return self.clean_text("\n".join(full_text))

    def run_ner_analysis(self, text):
        """Entity extraction with contextual clause detection."""
        if not self.nlp: return None
        doc = self.nlp(text)
        results = {"PARTIES": [], "DATES": [], "MONEY": [], "CLAUSES": []}
        
        for ent in doc.ents:
            if ent.label_ in ["ORG", "PERSON"]: results["PARTIES"].append(ent.text)
            elif ent.label_ == "DATE": results["DATES"].append(ent.text)
            elif ent.label_ == "MONEY": results["MONEY"].append(ent.text)

        for sent in doc.sents:
            if any(k in sent.text.lower() for k in ["terminate", "breach", "cancellation", "liability"]):
                results["CLAUSES"].append(sent.text.strip())
        return results

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    BASE_DIR = r"D:\LexiScan Auto"
    
    # Update this if Tesseract is not in your PATH
    TESS_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe" 
    
    processor = LexiScanHighFidelity(
        model_path=os.path.join(BASE_DIR, "lexiscan_bert_legal"),
        tesseract_path=TESS_PATH if os.path.exists(TESS_PATH) else None
    )
    
    pdf_files = [f for f in os.listdir(BASE_DIR) if f.lower().endswith(".pdf")]
    
    if pdf_files:
        target_path = os.path.join(BASE_DIR, pdf_files[0])
        print(f"--- Processing: {pdf_files[0]} ---")
        try:
            # Step 1: Extract
            raw_text = processor.extract_text(target_path)
            
            # Step 2: Analyze
            analysis = processor.run_ner_analysis(raw_text)
            
            if analysis:
                print("\n" + "="*50)
                print("LEXISCAN AUTO: NER ANALYSIS")
                print("="*50)
                print(f"Parties Found: {list(set(analysis['PARTIES']))[:5]}")
                print(f"Dates Found:   {list(set(analysis['DATES']))[:3]}")
                print(f"Legal Clauses: {len(analysis['CLAUSES'])}")
                print("="*50)
                if analysis['CLAUSES']:
                    print(f"Snippet: {analysis['CLAUSES'][0][:150]}...")
                
        except Exception as e:
            print(f"[PROCESS ERROR]: {e}")
    else:
        print(f"No PDF found in {BASE_DIR}")