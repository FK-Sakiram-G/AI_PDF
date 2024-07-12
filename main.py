import os
import json
import argparse
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage

from pdf_extraction import pdf_extraction_tool
from image_extraction import image_extraction_tool

def process_with_llm(text: str) -> str:
    api_key = "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"  # Your Groq API key
    model_name = "llama3-70b-8192"
    llm = ChatGroq(api_key=api_key, model=model_name, temperature=0.1)
    
    prompt = f"""Given the following extracted text from a document, please convert it into a structured JSON format.
    Include the title of the file in pdf and image format.
    Include relevant fields such as booking details, shipping details, containers or any other important information found in the text.
    Important Note : Do not leave out any fields that are present in the file.
    I need the output format as the following.
    No backticks.
        "fileName": "231231_booking_confirmation_1672312862.pdf",
        "orgName": "TestOrg ",
        "orderId": "",
        "ocrBucketName": "unilever-international-2086",
        “docType”: "BOOKING_CONFIRMATION",
        "bookingDetails":
            "carrierBookingNum": "", //booking number
            "billOfLadingNum": "",
            "referenceNumber": "",
            "carrierName": "SeaLand",
            "originServiceMode": "CY",
            "destinationServiceMode": "CY",
            "placeOfOrigin": "ZADUR",
            "placeOfDelivery": "KRPUS",
            "portOfLoad": "ZADUR",
            "portOfDischarge": "KRPUS",
            "temperature":
                "unit": "C",
                "value": 23
            "humidity": 13,
            "gensetRequired": false,
            "travelType": "SEA",
            "containerDetails":
                "code": "",
                "quantity": 0
            "weight":
                "unit": "mt",
                "value": 10
            "commodityType": "",
            "hazardous":
                "hazCode": "",
                "imoClassType": "",
                "packageCount": 1,
                "packageType": "",
                "packageGroup": ""
            "voyageInfo":
                "vesselName": "",
                "voyageNumber": "",
                "imoNumber": "",
                "departureEstimated": "",
                "arrivalEstimated": ""
        "shipmentDetails":
            "shipmentDate": "",
            "containerPickUpDate": "",
            "railCutOffDate": "",
            "portCutOffDate": "",
            "vgmCutoffDate": "",
            "portOpenDate": "",
            "siCutOffDate": "",
            "shipOnBoardDate": "",
            "vent": "close"
        ,
        "containers": [
            "containerId": "",
            "type": "40' Dry Standard"
        ,
            "containerId": "ABCD1234568",
            "type": "20' Dry standard"
        ]
    if some of the values are missing in the given JSON format, fill it as a empty string and
    it is mandatory to create the keys for temperature, humidity, weight, hazardous, voyageInfo only fill the fields if it is available.
    Dates and voyage estimates to be formatted as YYYY-MM-DD with hypens,
    convert the date format from: [JAN,FEB,MAR,APR,MAY,JUN,JUL,AUG,SEP,OCT,NOV,DEC] to: [01,02,03,04,05,06,07,08,09,10,11,12] for shipmentDetails,voyageInfo with timestamp if available.
    Just start with json curly bracket and end with it.
    ignore apostopes in between.
    Extracted text:
    {text}
    Respond only with the JSON and no other text.
    """
    messages = [HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    return response.content

def save_json_output(file_path: str, json_str: str):
    directory = os.path.dirname(file_path)
    dir = directory+"/../output_json_file"
    filename = os.path.splitext(os.path.basename(file_path))[0]
    output_file = os.path.join(dir, f"{filename}_output.json")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(json_str)
    
    print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract data from PDF or image files and convert to JSON.")
    parser.add_argument("--file_path", type=str, required=True, help="Path to the file to be processed.")
    args = parser.parse_args()

    file_path = args.file_path
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension == ".pdf":
        extracted_text = pdf_extraction_tool.run({"file_path": file_path})
    elif file_extension in [".png", ".jpg", ".jpeg", ".tiff"]:
        extracted_text = image_extraction_tool.run({"file_path": file_path})
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

    # Process the extracted text with the LLM to generate JSON
    json_output = process_with_llm(extracted_text)

    # Print the JSON response
    print(json_output)

    # Save the JSON output to a file
    save_json_output(file_path, json_output)