import pdfplumber
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.tools import Tool

class PDFExtractionArgs(BaseModel):
    file_path: str = Field(..., description="Path to the PDF file")

class PDFResponse(BaseModel):
    """Structured response for PDF data extraction"""
    text: str = Field(description="Extracted text from the PDF")

def extract_pdf(args: str | dict) -> PDFResponse:
    # Handle both string and dictionary input
    if isinstance(args, str):
        file_path = args
    elif isinstance(args, dict):
        file_path = args.get("file_path")
    else:
        raise ValueError("Invalid input type. Expected string or dictionary.")

    with pdfplumber.open(file_path) as pdf:
        text = "\n".join(page.extract_text() for page in pdf.pages)
    return PDFResponse(text=text)

pdf_extraction_tool = Tool(
    name="pdf_extraction",
    description="Extract text from PDF files",
    func=extract_pdf,
    args_schema=PDFExtractionArgs,
)

# Sample usage
if __name__ == "__main__":
    response = extract_pdf("/Users/kudipudi.bharat/Desktop/doc-json/2370172.PDF")
    print(response.json())