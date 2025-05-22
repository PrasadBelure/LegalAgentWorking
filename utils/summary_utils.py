# utils/summary_utils.py

from utils.gemini_utils import model

def generate_simple_summary(text: str) -> str:
    """
    Generate a simple, easy-to-understand summary of the given legal document text
    using the Gemini API.
    """
    prompt = f"""
You are a legal assistant specialized in summarizing Indian legal documents.
Please provide a clear and simple summary of the following text so that a
layperson can understand the main points:

{text}

Summary:"""

    response = model.generate_content(prompt)
    return response.text.strip()
