import pytesseract
from PIL import Image
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.tools import Tool

class ImageExtractionArgs(BaseModel):
    file_path: str = Field(..., description="Path to the image file")

class ImageResponse(BaseModel):
    """Structured response for image data extraction"""
    text: str = Field(description="Extracted text from the image")

def extract_image(args: str | dict) -> ImageResponse:
    # Handle both string and dictionary input
    if isinstance(args, str):
        file_path = args
    elif isinstance(args, dict):
        file_path = args.get("file_path")
    else:
        raise ValueError("Invalid input type. Expected string or dictionary.")

    text = pytesseract.image_to_string(Image.open(file_path))
    return ImageResponse(text=text)

image_extraction_tool = Tool(
    name="image_extraction",
    description="Extract text from image files",
    func=extract_image,
    args_schema=ImageExtractionArgs,
)

# Sample usage
if __name__ == "__main__":
    response = extract_image("/Users/kudipudi.bharat/Desktop/doc-json/img.png")
    print(response.json())