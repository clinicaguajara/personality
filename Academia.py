
# --- IMPORTAÇÕES NECESSÁRIAS ---

import streamlit as st

from utils.design   import inject_custom_css
from modules.scales_forms import render_scale_selector

# --- CONFIGURAÇÕES DA PÁGINA ---

st.set_page_config(
    page_title="Academia Diagnóstica",
    page_icon="🏛️",
    layout="centered"
)

inject_custom_css()


# --- TÍTULO E LEGIBILIDADE ---

st.title("Academia Diagnóstica")
st.markdown(
    "<h2 style='color:#ffd000;'>Sistema de Correções Informatizadas</h2>",
    unsafe_allow_html=True
)
st.markdown(
    """
    <p style='text-align: justify; font-size: 0.85rem; color: gray;'>
    O conteúdo e os resultados apresentados neste aplicativo têm finalidade exclusivamente informativa e educacional. 
    As interpretações geradas baseiam-se em dados psicométricos e modelos normativos, 
    não devendo ser utilizadas isoladamente para fins diagnósticos, clínicos ou jurídicos. 
    A análise psicológica adequada requer avaliação profissional conduzida por psicólogo(a) devidamente habilitado(a), 
    considerando o contexto individual, histórico clínico e outros instrumentos complementares. 
    A <b>Academia Diagnóstica</b> e seus colaboradores não se responsabilizam por decisões tomadas a partir das informações aqui apresentadas, 
    que devem ser vistas como um apoio técnico e não como laudo ou parecer psicológico.
    </p>
    """,
    unsafe_allow_html=True
)
st.divider()

# --- ESPAÇO DE RESERVA PARA EXPANSÃO FUTURA ---

render_scale_selector("scales")
