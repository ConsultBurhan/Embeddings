from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    parsers
)
from ocr import OCRExtractor
from langchain_core.documents import Document
from dotenv import load_dotenv
import tempfile
import os

load_dotenv()

class Extraction:
    def __init__(
        self,
        vlm_model: str = "gpt-4o-mini",
    ):
        self.vlm = ChatOpenAI(
            model=vlm_model,
        )


    # Helper method to handle loading with a temp file for all loaders
    def _load_with_temp_file(self, file_bytes: bytes, file_extension: str, loader_cls, **loader_kwargs) -> list[Document]:
        """
        Write bytes to a temp file, run the loader, then delete the temp file.
        Required because LangChain loaders only accept file paths.
        """
        with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            loader = loader_cls(tmp_path, **loader_kwargs)
            return loader.load()
        finally:
            os.remove(tmp_path)


    def _load_pdf(self, file_bytes: bytes, file_extension: str) -> list[Document]:
        docs = self._load_with_temp_file(
            file_bytes, 
            file_extension, 
            PyMuPDFLoader,
            mode='page',
            images_parser=parsers.TesseractBlobParser(langs=["ara", "eng"]),

        )
        if self._validate_ocr_needed(docs):
            return OCRExtractor.extract_text_OCR(file_bytes)
        
        return docs

    def _load_docx(self, file_bytes: bytes, file_extension: str) -> list[Document]:
        return self._load_with_temp_file(file_bytes, file_extension, Docx2txtLoader)

    def _load_txt(self, file_bytes: bytes, file_extension: str) -> list[Document]:
        return self._load_with_temp_file(file_bytes, file_extension, TextLoader, encoding="utf-8")

    def _load_md(self, file_bytes: bytes, file_extension: str) -> list[Document]:
        return self._load_with_temp_file(file_bytes, file_extension, UnstructuredMarkdownLoader)

    
    def _load_full_ocr(self, file_bytes: bytes) -> list[Document]:
        pass


    # Helper method to determine if the file need a full OCR or not 
    def _validate_ocr_needed(self, documents: list[Document], min_chars: int = 200) -> bool:
        """
        Check if the PDF needs a full OCR pass.
        Returns True if the total extracted text is too small to be meaningful.

        Args:
            documents: List of Document objects returned from PyMuPDFLoader.
            min_chars: Minimum number of non-whitespace characters expected across all pages.

        Returns:
            True if OCR is needed, False otherwise.
        """
        total_text = "".join(doc.page_content for doc in documents).strip()
        return len(total_text) < min_chars



    # Heloper methods to extract images from PDFs
    def _extract_images_pdf(self, file_bytes: bytes) -> list[dict]:
        import fitz

        doc = fitz.open(stream=file_bytes, filetype="pdf")
        images = []

        try:
            for page_num, page in enumerate(doc):
                for img in page.get_images(full=True):
                    xref = img[0]
                    base_image = doc.extract_image(xref)

                    if base_image["width"] < 100 or base_image["height"] < 100:
                        continue

                    images.append({
                        "bytes": base_image["image"],
                        "page":  page_num + 1,
                        "ext":   base_image["ext"],  # "png", "jpeg", "webp" etc.
                    })
        finally:
            doc.close()

        return images

    # Helper method to extract images from DOCX files by reading the ZIP structure and filtering for media files
    def _extract_images_docx(self, file_bytes: bytes) -> list[dict]:
        import zipfile
        import io

        images = []

        with zipfile.ZipFile(io.BytesIO(file_bytes), "r") as z:
            image_files = [f for f in z.namelist() if f.startswith("word/media/")]

            for img_path in image_files:
                ext = os.path.splitext(img_path)[1].lower().strip(".")

                if ext not in ("png", "jpg", "jpeg"):
                    continue

                images.append({
                    "bytes": z.read(img_path),
                    "page":  None,
                    "ext":   ext,
                })

        return images
    
    # Method to use the VLM to generate a detailed description of the image
    def understand_image(self, image: dict) -> dict:
        """
        Use the VLM to generate a detailed description of the image.

        Args:
            image: dict with keys: bytes, page, ext.

        Returns:
            Same dict with an additional "text" field containing the description.
        """
        import base64
        from langchain_core.messages import HumanMessage

        b64 = base64.b64encode(image["bytes"]).decode("utf-8")

        prompt = """Analyze this image and follow these rules in order:

1. If the image is primarily text (notes, screenshot, paragraph, chat, invoice, document):
   - Extract ALL the text exactly as written, preserving the original language and formatting.
   - Do not summarize, do not describe. Just extract the raw text faithfully.

2. If the image is visual content (chart, diagram, photo, illustration):
   - Describe it in detail including type, key elements, any text labels, and what it communicates.

Do not mix these two modes. Pick one based on what the image primarily contains. If in doubt, default to the first mode and extract text.
    """

        message = HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{b64}"
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        )

        response = self.vlm.invoke([message])

        return {
            **image,
            "text": response.content,
        }
 

    # helper method to extract images from the file
    def _extract_images_from_file(self, file_bytes: bytes, file_extension: str) -> list[dict]:
        """
        Extract images from a PDF or DOCX file.

        Args:
            file_bytes: Raw file bytes.
            file_extension: The extension of the file e.g. ".pdf", ".docx".

        Returns:
            List of dicts with keys: bytes, page.
        """
        mapping = {
            ".pdf":  self._extract_images_pdf,
            ".docx": self._extract_images_docx,
        }

        if file_extension not in mapping:
            raise ValueError(f"Image extraction not supported for: {file_extension}")

        return mapping[file_extension](file_bytes)

    def extract_and_describe_images(self, file_bytes: bytes, file_extension: str) -> list[dict]:
        """
        Extract images from a file and generate a description for each one.

        Args:
            file_bytes: Raw file bytes.
            file_extension: The extension of the file e.g. ".pdf", ".docx".

        Returns:
            List of dicts with keys: bytes, page, text.
        """
        images = self._extract_images_from_file(file_bytes, file_extension)
        if not images:
            return None
        return [self.understand_image(image) for image in images]   
    
       
    # Method to extract the text from the image
    def extract_text_from_file(self, file_bytes: bytes, file_extension: str) -> list[Document]:
        """
        Extract text from a file based on its extension.

        Args:
            file_bytes (bytes): File content in bytes.
            file_extension (str): File type (e.g., ".pdf", ".docx").

        Returns:
            list[Document]: Extracted documents.

        Raises:
            ValueError: If the file type is not supported.
        """
        mapping = {
            ".pdf":  self._load_pdf,
            ".docx": self._load_docx,
            ".txt":  self._load_txt,
            ".md":   self._load_md,
        }

        if file_extension not in mapping:
            raise ValueError(f"Unsupported file type: {file_extension}")

        return mapping[file_extension](file_bytes, file_extension)
    

    def extract_text_and_images(self, file_bytes: bytes, file_extension: str) -> tuple[list[Document], list[dict]]:
        """
        Extract both text and images from the file.

        Args:
            file_bytes: Raw file bytes.
            file_extension: The extension of the file e.g. ".pdf", ".docx".

        Returns:
            A tuple containing a list of Document objects for text and a list of dicts for images.
        """
        text_from_file = None
        text_from_images = None
        if file_extension not in (".pdf", ".docx", ".txt", ".md", ".jpeg", ".jpg", ".png"):
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        if file_extension in (".pdf", ".docx"):
            text_from_file = self.extract_text_from_file(file_bytes, file_extension)
            text_from_images = self.extract_and_describe_images(file_bytes, file_extension)

        if file_extension in (".txt", ".md"):
            text_from_file = self.extract_text_from_file(file_bytes, file_extension) 
        
        if file_extension in (".jpeg", ".jpg", ".png"):
            text_from_images = self.understand_image({
                "bytes": file_bytes,    
                "page":  None,
            })
    

        return {
            "text_from_file": text_from_file,
            "text_from_images": text_from_images,
        }
    


# singleton class 
ExtractionInstance : Extraction | None = None
def get_extraction_instance():
    global ExtractionInstance
    if ExtractionInstance is None:
        ExtractionInstance = Extraction()
    return ExtractionInstance

