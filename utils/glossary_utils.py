# utils/glossary_utils.py

import json
from utils.gemini_utils import model  # import the initialized Gemini model from gemini_utils

def extract_glossary_terms_and_explanations(text: str):
    """
    Extract complex legal terms, Acts, and laws from the text and provide
    short simple explanations for each in JSON format.
    """
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
    
    try:
        glossary = json.loads(response.text)
        return glossary
    except json.JSONDecodeError:
        # If the output is not a valid JSON, return the raw text wrapped in a dict for fallback
        return [{
            "term": "ParsingError",
            "type": "Error",
            "explanation": "Failed to parse Gemini API response as JSON",
            "difficulty": "N/A",
            "raw_response": response.text
        }]

def rate_term_difficulty(term: str, context: str = ""):
    """
    Given a single legal term and optional context, get difficulty rating from Gemini.
    """
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
    return response.text.strip()
