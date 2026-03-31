import platform
import io
import numpy as np
import cv2
import pytesseract
from PIL import Image
from langchain_core.documents import Document


class OCRExtractor:
    if platform.system() == "Windows":
            pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

        
    @staticmethod
    def preprocess_image(image_bytes: bytes) -> bytes:
        """
        Clean and enhance a scanned page image before passing to Tesseract.

        Steps:
            1. Grayscale conversion
            2. Deskew
            3. Denoise
            4. Adaptive thresholding
            5. Morphological cleaning

        Args:
            image_bytes: Raw image bytes of the scanned page.

        Returns:
            Cleaned image bytes ready for OCR.
        """
        nparr  = np.frombuffer(image_bytes, np.uint8)
        img    = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray   = OCRExtractor._deskew(gray)

        denoised = cv2.fastNlMeansDenoising(gray, h=10)

        binary = cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            10,
        )

        kernel  = np.ones((1, 1), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        _, buffer = cv2.imencode(".png", cleaned)
        return buffer.tobytes()

    @staticmethod
    def _deskew(image: np.ndarray) -> np.ndarray:
        """
        Detect and correct skew angle of a scanned page.

        Args:
            image: Grayscale numpy image array.

        Returns:
            Deskewed image array.
        """
        coords = np.column_stack(np.where(image > 0))

        if len(coords) == 0:
            return image

        angle = cv2.minAreaRect(coords)[-1]
        angle = -(90 + angle) if angle < -45 else -angle

        if abs(angle) < 0.5:
            return image

        (h, w)          = image.shape[:2]
        center          = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        return cv2.warpAffine(
            image,
            rotation_matrix,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )


    @staticmethod
    def extract_text_OCR(file_bytes: bytes, lang: str = "ara+eng") -> list[Document]:
        """
        Takes PDF file bytes, rasterizes each page and extracts text using Tesseract.

        Args:
            file_bytes: Raw PDF file bytes.
            lang:       Tesseract language string. Defaults to Arabic + English.

        Returns:
            List of LangChain Documents, one per page.
        """
        import fitz
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        documents = []

        try:
            doc = fitz.open(tmp_path)

            for page_num, page in enumerate(doc):
                pix           = page.get_pixmap(dpi=300)
                image_bytes   = pix.tobytes("png")
                cleaned_bytes = OCRExtractor.preprocess_image(image_bytes)
                img           = Image.open(io.BytesIO(cleaned_bytes))
                text          = pytesseract.image_to_string(img, lang=lang)

                if text.strip():
                    documents.append(Document(
                        page_content=text.strip(),
                        metadata={
                            "page":   page_num + 1,
                            "source": "ocr",
                            "lang":   lang,
                        }
                    ))

            doc.close()

        finally:
            os.remove(tmp_path)

        return documents
    
