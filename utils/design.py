# utils/design.py
import streamlit as st

from pathlib import Path

# --- Functions definition ---

def inject_custom_css():
    css_path = Path("assets/style.css")
    if css_path.exists():
        # forçar UTF-8 para não dar UnicodeDecodeError no Windows
        with open(css_path, encoding="utf-8") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)