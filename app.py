import pandas as pd
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from typing import List
import fitz
from paddleocr import PaddleOCR
from transformers import pipeline
import io
import os
import tempfile

app = FastAPI()

# Initialize PaddleOCR for text extraction
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Load a robust QA model optimized for CPUs with the new model
qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Load the hospital keywords from the Excel file
def load_keywords_from_excel(hospital_code: str):
    try:
        # Load the Excel file with hospital data
        df = pd.read_excel("Keywords.xlsx") 

        # Find the row that matches the hospital code
        matched_row = df[df['Hospital ID'] == hospital_code]

        if not matched_row.empty:
            # Extract keywords from the matching row
            keywords = matched_row.iloc[0]['Keywords']
            return [keyword.strip() for keyword in keywords.split(',')]
        else:
            print(f"Hospital code {hospital_code} not found in the Excel file.")
            return []
    except Exception as e:
        print(f"Error loading keywords from Excel: {str(e)}")
        return []

# Function to extract information using OCR and QA model
def extract_text_from_pdf(pdf_file, hospital_code, requested_attributes):
    doc = fitz.open(pdf_file)

    # Get the keywords for the given hospital code
    keywords = load_keywords_from_excel(hospital_code)
    if not keywords:
        print(f"No keywords found for hospital code {hospital_code}.")
        return None

    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap()
        image_path = f"temp_page_{page_num + 1}.png"
        pix.save(image_path)

        result = ocr.ocr(image_path, cls=False)
        if not result or not result[0]:
            continue
        
        extracted_text = " ".join([line[1][0] for line in result[0]])
        print(f"Extracted text from page {page_num + 1}: {extracted_text}")  # Debug print

        if all(keyword.lower() in extracted_text.lower() for keyword in keywords):
            print(f"Keywords found on page {page_num + 1}, extracting information...")  # Debug print
            return extract_information_with_qa_model(extracted_text, requested_attributes)
    
    print("No matching page found with the provided keywords.")  # Debug print
    return None

def extract_information_with_qa_model(context, requested_attributes):
    # Define the possible questions
    questions = {
        "Patient Name": "What is the full name of the patient mentioned in the context?",
        "Age": "What is the age of the patient? Provide only the number.",
        "Gender": "What is the gender of the patient?",
        "Admission Date": "What is the exact admission date of the patient in 'dd/mm/yyyy' format?",
        "Discharge Date": "What is the exact Discharge date of the patient in 'dd/mm/yyyy' format?",
        "UHID Number": "What is the UHID (unique hospital identification number) of the patient?",
        "IPD Number": "What is the IPD (in-patient department) number of the patient?",
        "IP Number": "What is the IP number of the patient?",
        "MR Number": "What is the MR number of the patient?",
        "UMR Number": "What is the UMR number of the patient?",
        "Doctor Name": "What is the Doctor Name who is treating the patient, given in the context"
    }

    # Extract answers for all questions
    extracted_info = {}
    print(f"Running QA model for all attributes...")  # Debug print
    for key, question in questions.items():
        try:
            response = qa_model(question=question, context=context)
            answer = response.get("answer", "NULL")
            extracted_info[key] = answer if answer.strip() else "NULL"
            print(f"Extracted info for {key}: {answer}")  # Debug print
        except Exception as e:
            extracted_info[key] = "NULL"
            print(f"Error extracting info for {key}: {str(e)}")  # Debug print

    # Filter out the requested attributes and return only the required answers
    filtered_info = {key: value for key, value in extracted_info.items() if key in requested_attributes}
    
    print(f"Filtered extracted info (only requested attributes): {filtered_info}")  # Debug print
    return filtered_info

@app.post("/extract-patient-info/")
async def extract_patient_info(
    requested_attributes: str = Form(...),  # Taking input as comma-separated string
    file: UploadFile = File(...)
):
    try:
        # Extract hospital code from the file name (before the first 'P')
        hospital_code = file.filename.split('P')[0]
        print(f"Hospital code extracted from file name: {hospital_code}")  # Debug print

        # Convert comma-separated string to a list
        requested_attributes_list = [attribute.strip() for attribute in requested_attributes.split(',')]
        print(f"Requested attributes (as list): {requested_attributes_list}")  # Debug print
        
        # Save the uploaded file to a temporary file with a proper filename
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name
        
        # Extract information
        extracted_info = extract_text_from_pdf(temp_file_path, hospital_code, requested_attributes_list)

        # Return the extracted information or a message if not found
        if extracted_info:
            print(f"Final extracted info: {extracted_info}")  # Debug print
            return JSONResponse(content=extracted_info)
        else:
            print("No relevant information found.")  # Debug print
            return JSONResponse(content={"message": "No relevant information found."}, status_code=404)
    except Exception as e:
        print(f"Error: {str(e)}")  # Debug print
        return JSONResponse(content={"message": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))  # Default to 8000 if PORT is not set
    uvicorn.run(app, host="0.0.0.0", port=port)