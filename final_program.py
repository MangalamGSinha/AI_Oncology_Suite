import os
import json
import re
import time
import pandas as pd
import pdfplumber
import google.generativeai as genai

# NOTE: Replace with your actual API key or use os.environ
# genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
# Using a placeholder key for completeness, but recommend using environment variables.
api_key = os.getenv("GENAI_API_KEY")
genai.configure(api_key=api_key)


# STEP 1: Extract text from PDF
def extract_text_from_pdf(pdf_path):
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
        # print(f"✓ Text extracted from {os.path.basename(pdf_path)}")
        return text_content.strip()
    except Exception as e:
        # print(f"✗ Error extracting text: {e}")
        return ""

def split_into_sections(text):
    """
    Splits long medical text into logical sections using headings or large gaps.
    """
    # Split by uppercase section titles or large gaps/newlines
    sections = re.split(r'\n\s*(?:[A-Z][A-Z\s]{3,}|[A-Z][a-z]+:)\s*\n', text)
    # Filter out very short sections and strip whitespace
    sections = [s.strip() for s in sections if len(s.strip()) > 200]

    # Fallback if no large sections are detected
    if not sections:
        # Simple character-based chunking with overlap
        chunk_size = 3000
        overlap = int(chunk_size * 0.1) # 10% overlap
        sections = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size - overlap)]

    return sections


def extract_attributes_with_gemini(text_content, prompt, required_keys=None, chunk_delay=0.5):
    """
    Enhanced version with:
    - Smart section chunking
    - Chunk-wise Gemini extraction
    - Unified summary-of-chunks pass
    """
    # print("🧠 Starting enhanced extraction with smart chunking...")
    
    # Simple mode fallback (using only the first 8000 chars if not in advanced mode)
    if not required_keys: # used as a signal for simple extraction mode in app.py
        text_content = text_content[:8000]

    sections = split_into_sections(text_content) if required_keys else [text_content]
    # print(f"🔹 Found {len(sections)} sections for processing.")

    all_partial_results = []
    model = genai.GenerativeModel("gemini-2.0-flash") # Switched to a standard fast model

    for i, section in enumerate(sections, 1):
        # print(f"\n🧩 Processing section {i}/{len(sections)} ({len(section)} chars)...")

        section_prompt = (
            prompt
            + "\n\nMedical Report Section (Extract data from this section only):\n"
            + section
            + "\n\nReturn structured JSON only, no extra text or markdown."
        )

        try:
            response = model.generate_content(
                section_prompt,
                generation_config={"max_output_tokens": 1500}
            )

            # Clean JSON output
            clean_json = re.sub(r'^\s*```json\s*', '', response.text or "")
            clean_json = re.sub(r'\s*```\s*$', '', clean_json, flags=re.MULTILINE)

            try:
                json_obj = json.loads(clean_json)
                # Filter out empty dictionaries
                if json_obj:
                    all_partial_results.append(json_obj)
                # print(f"✅ Section {i} extracted successfully (data size: {len(json_obj)} keys).")
            except json.JSONDecodeError:
                # print(f"⚠️ Section {i} returned invalid JSON, skipping.")
                pass

        except Exception as e:
            # print(f"❌ Error in section {i}: {e}")
            pass
        time.sleep(chunk_delay)
    
    # If only one section was processed, or if advanced mode is disabled
    if len(sections) == 1:
        return all_partial_results[0] if all_partial_results else {}

    if not all_partial_results:
        # print("⚠️ No valid sections extracted.")
        return {"error": "No valid sections extracted"}

    # Step 2: Combine all partial JSONs into one unified dictionary
    # print("\n🧩 Combining all partial results into one unified JSON...")
    try:
        combine_prompt = (
            "You are an AI assistant. Combine the following partial JSON medical report "
            "extractions into ONE unified, clean JSON object. Merge overlapping fields, "
            "preserve all data, and remove duplicates. The final JSON must be a single flat "
            "object, converting any nested lists or dictionaries into concise strings (if necessary). "
            "Return ONLY valid JSON.\n\n"
            "Partial JSONs:\n" + json.dumps(all_partial_results, indent=2)
        )

        response = model.generate_content(
            combine_prompt,
            generation_config={"max_output_tokens": 2048}
        )

        combined_json = re.sub(r'^\s*```json\s*', '', response.text or "")
        combined_json = re.sub(r'\s*```\s*$', '', combined_json, flags=re.MULTILINE)
        combined_data = json.loads(combined_json)

        # Final key validation (mostly done by the model, but a safe check)
        if required_keys:
            final_data = {}
            for key in required_keys:
                final_data[key] = combined_data.get(key, None)
            combined_data = final_data
        
        # print("✅ Successfully combined all partial data.")
        return combined_data

    except Exception as e:
        # print(f"❌ Error combining chunk outputs: {e}")
        return {"error": str(e), "partial_results": all_partial_results}

# New function for the Streamlit UI to handle a single file processing
def process_single_uploaded_file(uploaded_file, prompt, specific_attributes, advanced_mode):
    """
    Handles the full processing pipeline for a single Streamlit uploaded file.
    
    Returns:
        A dictionary of the extracted data.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name
    
    try:
        text_content = extract_text_from_pdf(temp_path)
        if not text_content:
            return {"error": "Could not extract text from PDF."}

        # The presence of specific_attributes (a list) signals advanced mode in the backend
        # Even if advanced_mode is False in the UI, we only pass required_keys=None 
        # to the backend for the simple extraction path (text[:8000]).
        keys_for_backend = specific_attributes if advanced_mode else None
        
        result = extract_attributes_with_gemini(text_content, prompt, required_keys=keys_for_backend)
        
        # Add filename for context
        result['source_file'] = uploaded_file.name.split('.')[0]
        
        return result
        
    finally:
        os.unlink(temp_path)


# ================================================================
# Batch Processing Utilities (for a cleaner final_program.py)
# ================================================================

import tempfile # Make sure tempfile is imported

def process_pdf_list(uploaded_files, prompt, specific_attributes, advanced_mode):
    """
    Handles the full processing pipeline for a list of Streamlit uploaded files.
    
    Returns:
        A list of dictionaries, one for each file.
    """
    all_results = []
    
    for uploaded_file in uploaded_files:
        result = process_single_uploaded_file(uploaded_file, prompt, specific_attributes, advanced_mode)
        
        # Check for errors before appending
        if not result.get("error"):
             all_results.append(result)
        
    return all_results
