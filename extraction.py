from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.documents import Document
import tempfile
import os


# Global variables
OPENAI_API_KEY="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"


class Extraction:
    def __init__(
        self,
        openai_api_key: str,
        vlm_model: str = "gpt-4o-mini",
    ):
        self.vlm = ChatOpenAI(
            model=vlm_model,
            api_key=openai_api_key,
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
        return self._load_with_temp_file(file_bytes, file_extension, PyMuPDFLoader)

    def _load_docx(self, file_bytes: bytes, file_extension: str) -> list[Document]:
        return self._load_with_temp_file(file_bytes, file_extension, Docx2txtLoader)

    def _load_txt(self, file_bytes: bytes, file_extension: str) -> list[Document]:
        return self._load_with_temp_file(file_bytes, file_extension, TextLoader, encoding="utf-8")

    def _load_md(self, file_bytes: bytes, file_extension: str) -> list[Document]:
        return self._load_with_temp_file(file_bytes, file_extension, UnstructuredMarkdownLoader)

   
    # Helper method to determine if OCR is needed based on the amount of text extracted
    def _validate_ocr_need(self, documents: list[Document], min_non_whitespace_chars: int = 200) -> bool:
        """
        Check if the loaded documents have enough text or if OCR is needed.
        Combines all page content and checks against the minimum threshold.

        Args:
            documents: List of Document objects returned from any loader.
            min_non_whitespace_chars: Minimum non-whitespace characters expected.

        Returns:
            True if OCR is recommended, False otherwise.
        """
        combined = " ".join(doc.page_content for doc in documents).strip()
        return (not combined) or (len(combined) < min_non_whitespace_chars)
    
    

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
                })

        return images
    
    # Method to use the VLM to generate a detailed description of the image
    def understand_image(self, image: dict) -> dict:
        """
        Use the VLM to generate a detailed description of the image.

        Args:
            image: dict with keys: bytes, page.

        Returns:
            Same dict with an additional "text" field containing the description.
        """
        import base64
        from langchain_core.messages import HumanMessage

        b64 = base64.b64encode(image["bytes"]).decode("utf-8")

        prompt = """You are analyzing an image extracted from a document.
    Your task is to produce a thorough, factual description that will be used for semantic search and retrieval.

    Follow these rules:
    - If the image contains text, extract it exactly as written, preserving the original language (including Arabic or any other language).
    - If the image is a chart or graph, describe the type, axes, trends, and key data points.
    - If the image is a table, extract all rows and columns with their values.
    - If the image is a diagram or flowchart, describe the components and the relationships between them.
    - If the image is a photograph or illustration, describe the key visual elements, objects, and context.
    - Be specific and factual. Do not make assumptions or infer information not visible in the image.
    - Do not include phrases like "the image shows" or "I can see". Just describe directly.
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
    


if __name__ == "__main__":
    # Example usage
    extractor = Extraction(openai_api_key=OPENAI_API_KEY)
    with open("resume.docx", "rb") as f:
        file_bytes = f.read()
        results = extractor.extract_text_and_images(file_bytes, ".docx")
        print("Text from file:", results["text_from_file"])
        print("-----------------------------------------------------------------#-------------------------------------------------------------")
        print("Text from images:", results["text_from_images"])

    

