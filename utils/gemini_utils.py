import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not set in environment variables")

genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-1.5-flash")

def try_parse_json(text):
    try:
        return json.loads(text)
    except Exception:
        return None

def get_gemini_answer(query, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = f"""You are a knowledgeable legal assistant specialized in Indian law.
Use the following context carefully to answer the question precisely and simply:

{context}

Question: {query}
Answer:"""
    response = model.generate_content(prompt)
    
    # Try parse JSON from response.text first
    if hasattr(response, "text") and response.text:
        parsed = try_parse_json(response.text)
        if parsed is not None:
            return parsed
    
    # Try parse JSON from candidates output if any
    if hasattr(response, "candidates") and len(response.candidates) > 0:
        output_text = response.candidates[0].output
        parsed = try_parse_json(output_text)
        if parsed is not None:
            return parsed
    
    # Fallback to raw text
    if hasattr(response, "text") and response.text:
        return response.text.strip()
    if hasattr(response, "candidates") and len(response.candidates) > 0:
        return response.candidates[0].output.strip()
    
    return str(response)

def extract_glossary_terms_and_explanations(text):
    prompt = f"""
You are a legal expert assistant. From the following Indian legal document text, identify:
- Complex legal terms a layman may not understand,
- Acts, Laws, and Sections mentioned,
- For each term/law, provide a short, simple explanation,
- Rate the difficulty of understanding each term as Easy, Medium, or Hard.

Return the result as a JSON list of objects with keys:
term, type (Legal Term / Act / Section), explanation, difficulty

Here is the text:
{text}
"""
    response = model.generate_content(prompt)
    
    # Try parse JSON from text or candidates with fallback
    if hasattr(response, "text") and response.text:
        parsed = try_parse_json(response.text)
        if parsed is not None:
            return parsed
    
    if hasattr(response, "candidates") and len(response.candidates) > 0:
        output_text = response.candidates[0].output
        parsed = try_parse_json(output_text)
        if parsed is not None:
            return parsed

    # Return error info if parsing fails
    return [{"term": "ParsingError", "type": "Error", "explanation": "Failed to parse JSON from Gemini response", "difficulty": "N/A"}]

def rate_term_difficulty(term, context=""):
    prompt = f"""
You are a legal language difficulty expert. Given the legal term below, rate
its difficulty for a common person as Easy, Medium, or Hard. Provide a
short reason.

Term: {term}
Context: {context}

Response format:
Difficulty: <Easy|Medium|Hard>
Reason: <short explanation>
"""
    response = model.generate_content(prompt)
    
    # Return raw text — optionally you can parse and structure this too
    return response.text.strip() if hasattr(response, "text") and response.text else str(response)
