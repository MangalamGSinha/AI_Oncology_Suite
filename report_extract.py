import os
import json
import re
import pandas as pd
import pdfplumber
from google import genai

# Initialize Gemini client
api_key=os.getenv("GENAI_API_KEY")
client = genai.Client(api_key=api_key)

# STEP 1: Extract text from PDF
def extract_text_from_pdf(pdf_path):#might need to change dis function based on how pdf is given to program
    """
    Extract text content from a single PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text as a string
    """
    text_content = ''
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text_content += page.extract_text() or ''
        print(f"✓ Text extracted from {os.path.basename(pdf_path)}")
        return text_content.strip()
    except Exception as e:
        print(f"✗ Error extracting text: {e}")
        return ""


# STEP 2: Get user input for extraction mode
def get_extraction_mode():
    """
    Prompt user to choose between general or specific attribute extraction.
    
    Returns:
        prompt_string
    """
    #will change according to how user input for extraction mode is given by streamlit app, finally choices can either be "1" or "2"
    print("=" * 60)
    print("Medical Report Extraction Tool")
    print("=" * 60)
    print("\nChoose extraction mode:")
    print("1. Extract ALL information (default)")
    print("2. Extract SPECIFIC attributes only")
    print()
    
    choice = input("Enter your choice (1 or 2, default=1): ").strip()
    
    specific_attributes = []
    if choice == "2":
        print("\nEnter the attributes you want to extract (comma-separated).")
        print("Examples: tumor_type, tumor_size, histologic_grade, margins, lymph_nodes, ER_status, PR_status, HER2_status")
        attr_input = input("Attributes: ").strip()
        specific_attributes = [attr.strip() for attr in attr_input.split(",") if attr.strip()]
        
        if not specific_attributes:
            print("No attributes provided. Defaulting to extract ALL information.")
            choice = "1"
        else:
            print(f"\nWill extract: {', '.join(specific_attributes)}")
    else:
        print("\nExtracting ALL information from reports.")
    
    print("=" * 60)
    print()
    
    # Build prompt based on choice
    if choice == "2" and specific_attributes:
        attributes_list = ", ".join(specific_attributes)
        prompt = (
            "You are an AI assistant capable of understanding complex medical reports.\n"
            "The report may contain masked or anonymized patient data and some random or corrupted characters.\n"
            f"Extract ONLY the following attributes from the report: {attributes_list}\n"
            "Return only valid JSON with keys matching the requested attributes.\n"
            "Do not include patient names or identifiers.\n"
            "If an attribute is not found, set it to null.\n"
            "Your response should be ONLY JSON, no extra commentary.\n\n"
        )
    else:
        prompt = (
            "You are an AI assistant capable of understanding complex medical reports.\n"
            "The report may contain masked or anonymized patient data and some random or corrupted characters.\n"
            "Extract whatever meaningful information you can find in the report and represent it as JSON.\n"
            "Return only valid JSON with keys describing contents such as symptoms, diagnoses, treatments, labs, observations, dates, or any other useful details.\n"
            "Do not include patient names or identifiers.\n"
            "If you find nothing meaningful, return an empty JSON object {}.\n"
            "Your response should be ONLY JSON, no extra commentary.\n\n"
        )
    
    return prompt

# STEP 3: Clean JSON response from API
def extract_clean_json(api_response_string):
    """
    Remove markdown code fences from API response.
    
    Args:
        api_response_string: Raw API response text
        
    Returns:
        Clean JSON string
    """
    clean_string = re.sub(r'^\s*```json\s*', '',api_response_string)
    clean_string = re.sub(r'\s*```\s*$', '', clean_string, flags=re.MULTILINE)
    return clean_string

# STEP 4: Send text to Gemini API for extraction
def extract_attributes_with_gemini(text_content, prompt):
    """
    Send extracted text to Gemini API for attribute extraction.
    
    Args:
        text_content: Extracted text from PDF
        prompt: Extraction prompt (general or specific)
        
    Returns:
        Parsed JSON dictionary or error dict
    """
    try:
        # Trim text to reasonable size
        text_input = text_content[:8000]
        
        print("Sending request to Gemini API...")
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=prompt + "\nMedical Report:\n" + text_input
        )
        
        try:
            clean_json = extract_clean_json(response.text)
            structured_data = json.loads(clean_json)
            print("✓ Successfully extracted attributes")
            return structured_data
        except json.JSONDecodeError as je:
            print(f"✗ JSON decode error: {je}")
            return {"error": "Invalid JSON", "raw_response": response.text}
    
    except Exception as e:
        print(f"✗ API error: {e}")
        return {"error": str(e)}


# STEP 5: Save JSON to Excel

def save_to_excel(json_data, excel_path): #might have to change depending on how streamlit app handle excel as output
    """
    Normalize JSON and save to Excel file.
    
    Args:
        json_data: Dictionary containing extracted attributes
        excel_path: Path to save Excel file
    """
    try:
        # Normalize the JSON to a flat DataFrame
        df = pd.json_normalize(json_data)
        
        # Save to Excel
        df.to_excel(excel_path, index=False)
        print(f"✓ Data saved to {excel_path}")
        
        # Print dimensions
        print(f"  Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        
    except Exception as e:
        print(f"✗ Error saving to Excel: {e}")



# MAIN PIPELINE: Process single PDF

def process_single_pdf(pdf_path):
    """
    Complete pipeline to process a single PDF file.
    
    Args:
        pdf_path: Path to the PDF file to process
    """
    # Get filename without extension
    filename = os.path.splitext(os.path.basename(pdf_path))[0]
    
    print(f"\n{'='*60}")
    print(f"Processing: {filename}")
    print(f"{'='*60}\n")
    
    # Step 1: Extract text from PDF
    text_content = extract_text_from_pdf(pdf_path)
    if not text_content:
        print("No text extracted. Exiting.")
        return
    
    # Step 2: Get extraction mode from user
    prompt= get_extraction_mode() 
    
    # Step 3 & 4: Extract attributes using Gemini
    extracted_data = extract_attributes_with_gemini(text_content, prompt)
    
    # Add filename to the extracted data
    extracted_data['source_file'] = filename
    
    # Step 5: Save to Excel
    
    excel_filename = f"{filename}_attributes.xlsx"
    
    save_to_excel(extracted_data, excel_filename)
    
    print(f"\n{'='*60}")
    print(f"✓ Processing complete for {filename}")
    print(f"{'='*60}\n")

# ENTRY POINT
if __name__ == "__main__":
    # Get PDF file path from user
    pdf_file = ".\\Data\\00a7f79f-49bd-46b9-bde2-1a28ac6eecce.pdf" #will have to change depending on how streamlit app will provide the pdf
    
    if not os.path.exists(pdf_file):
        print(f"Error: File not found - {pdf_file}")
    elif not pdf_file.lower().endswith('.pdf'):
        print("Error: File must be a PDF")
    else:
        process_single_pdf(pdf_file)

