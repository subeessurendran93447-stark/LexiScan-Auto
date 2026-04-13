import os
import re
import uuid
import shutil
import fitz  # PyMuPDF
import cv2
import numpy as np
import pytesseract
import spacy
from datetime import datetime
from typing import List, Dict
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from pdf2image import convert_from_path

# ==========================================
# 1. CORE VALIDATION LOGIC
# ==========================================
class LexiValidator:
    @staticmethod
    def validate_date(date_str: str) -> str:
        """Standardizes date formats to YYYY-MM-DD."""
        clean_date = re.sub(r'[|©_•*]', '', date_str).strip()
        # Common legal date patterns
        formats = ["%d/%m/%Y", "%Y-%m-%d", "%B %d, %Y", "%m/%d/%Y", "%d-%b-%Y"]
        for fmt in formats:
            try:
                dt = datetime.strptime(clean_date, fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue
        return None

    @staticmethod
    def validate_money(money_str: str) -> str:
        """Filters and cleans currency strings from OCR noise."""
        # Fix common OCR errors (S instead of $, l instead of 1)
        money_str = money_str.replace('S', '$').replace('l', '1').strip()
        pattern = r'^[\$\£\€\₹]?\s?\d+(?:,\d{3})*(?:\.\d{2})?$'
        return money_str if re.match(pattern, money_str) else None

# ==========================================
# 2. OCR & NLP PROCESSING ENGINE
# ==========================================
class LexiScanProcessor:
    def __init__(self):
        # 3-Tier Fallback Loading for spaCy
        try:
            self.nlp = spacy.load("en_core_web_trf")  # High accuracy Transformer
            print("[NLP] Transformer Model Loaded.")
        except:
            print("[NLP] Transformer failed, loading Small Model...")
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except:
                raise ImportError("Please run 'python -m spacy download en_core_web_sm'")

    def preprocess_image(self, pil_img):
        """Enhances scanned images for better Tesseract accuracy."""
        img = np.array(pil_img.convert('RGB'))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Upscale and Denoise
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        return cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )

    def extract_text(self, pdf_path: str) -> str:
        """Determines if PDF is digital or scanned, then extracts text."""
        with fitz.open(pdf_path) as doc:
            text = "".join([p.get_text() for p in doc]).strip()
        
        # If text is present, return it. If empty or too short, it's likely a scan.
        if len(text) > 100:
            return text

        # Scanned PDF Workflow (OCR)
        pages = convert_from_path(pdf_path, dpi=300, first_page=1, last_page=3)
        full_text = []
        for page in pages:
            processed = self.preprocess_image(page)
            full_text.append(pytesseract.image_to_string(processed))
        
        return "\n".join(full_text)

    def run_analysis(self, text: str) -> Dict:
        """Uses NER to find parties, dates, money, and key clauses."""
        doc = self.nlp(text)
        results = {"PARTIES": [], "DATES": [], "MONEY": [], "CLAUSES": []}
        validator = LexiValidator()

        for ent in doc.ents:
            if ent.label_ in ["ORG", "PERSON"] and len(ent.text) > 3:
                results["PARTIES"].append(ent.text.strip())
            elif ent.label_ == "DATE":
                v_date = validator.validate_date(ent.text)
                if v_date: results["DATES"].append(v_date)
            elif ent.label_ == "MONEY":
                v_money = validator.validate_money(ent.text)
                if v_money: results["MONEY"].append(v_money)

        # Keyword-based Clause Extraction
        keywords = ["terminate", "breach", "liability", "indemnity", "governing law"]
        for sent in doc.sents:
            s_text = sent.text.strip()
            if any(k in s_text.lower() for k in keywords):
                if len(s_text.split()) > 6: # Filter out short headers
                    results["CLAUSES"].append(s_text)
        
        # De-duplicate results
        return {k: list(set(v)) for k, v in results.items()}

# ==========================================
# 3. FASTAPI REST ENDPOINT
# ==========================================
app = FastAPI(title="LexiScan Contract API")
engine = LexiScanProcessor()

class ScanResponse(BaseModel):
    status: str
    filename: str
    data: Dict

@app.post("/process-contract", response_model=ScanResponse)
async def process_contract(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid file type. PDF required.")

    job_id = str(uuid.uuid4())
    temp_file = f"temp_{job_id}.pdf"

    try:
        # Save uploaded file temporarily
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Execute Engine Pipeline
        raw_text = engine.extract_text(temp_file)
        analysis = engine.run_analysis(raw_text)

        return {
            "status": "success",
            "filename": file.filename,
            "data": analysis
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup
        if os.path.exists(temp_file):
            os.remove(temp_file)

# ==========================================
# 4. EXECUTION
# ==========================================
if __name__ == "__main__":
    import uvicorn
    # Run the server on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)