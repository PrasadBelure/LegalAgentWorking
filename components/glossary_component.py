# components/glossary_component.py

import streamlit as st

def display_glossary(glossary_list):
    """
    Display glossary terms with expandable explanations and difficulty ratings.

    glossary_list: list of dicts, each dict has:
      - term: str
      - type: str (e.g., Legal Term, Act, Section)
      - explanation: str
      - difficulty: str (Easy, Medium, Hard)
    """
    st.subheader("Glossary of Legal Terms")

    if not glossary_list or not isinstance(glossary_list, list):
        st.write("No glossary data available.")
        return

    for item in glossary_list:
        term = item.get("term", "Unknown Term")
        term_type = item.get("type", "Unknown Type")
        explanation = item.get("explanation", "No explanation provided.")
        difficulty = item.get("difficulty", "Unknown")

        with st.expander(f"{term} ({term_type}) — Difficulty: {difficulty}"):
            st.write(explanation)
