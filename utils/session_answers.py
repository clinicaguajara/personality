# utils/session_answers.py
import streamlit as st

from typing import Dict, Any

# --- Functions definition ---

def save_scale_answers(scale_name: str, answers_raw: Dict[int|str, Any]) -> None:
    key_data   = "escalas_respondidas"
    key_names  = "escalas_display_names"
    norm_key   = scale_name.strip().lower()

    if key_data not in st.session_state:
        st.session_state[key_data] = {}
    if key_names not in st.session_state:
        st.session_state[key_names] = {}

    st.session_state[key_data][norm_key]  = answers_raw
    st.session_state[key_names][norm_key] = scale_name  