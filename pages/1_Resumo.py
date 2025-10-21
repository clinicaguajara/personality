# pages/1_Resumo.py
from __future__ import annotations

import unicodedata
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from utils.design   import inject_custom_css

import pandas as pd
import streamlit as st

# --- Correções / cálculo ---
from modules.corrections import (
    score_pid5_facets,
    summarize_with_norms,
    build_classification_table,
)

from utils.global_variables import BLANK

inject_custom_css()

# =========================================
# Utilidades
# =========================================

def _normalize_answers(answers_raw: Dict[Any, Any]) -> Dict[int, Any]:
    """Converte chaves str->int e substitui sentinelas/strings vazias por None."""
    answers_norm: Dict[int, Any] = {}
    for k, v in answers_raw.items():
        try:
            ik = int(str(k))
        except Exception:
            continue
        answers_norm[ik] = None if v in (None, "", BLANK) else v
    return answers_norm

def _norm_str(s: str) -> str:
    s = str(s or "").strip().lower()
    s = " ".join(s.split())
    s = "".join(ch for ch in unicodedata.normalize("NFD", s) if unicodedata.category(ch) != "Mn")
    return s

def _find_scale_json_candidates(
    display_label: str,
    search_dirs: Optional[List[str]] = None,
) -> List[Tuple[Path, Dict[str, Any], str]]:
    """
    Retorna SOMENTE JSONs cujo 'scale' (ou 'name'/'titulo') é idêntico (case/acentos-insensitive)
    ao nome exibido da escala (display_label). Sem score, sem overlap de itens.
    """
    candidates_dirs = search_dirs or ["bibliography"]
    target = _norm_str(display_label)

    matches: List[Tuple[Path, Dict[str, Any], str]] = []
    for d in candidates_dirs:
        base = Path(d)
        if not base.exists():
            continue
        for p in base.glob("*.json"):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue

            cand_name_raw = data.get("scale") or data.get("name") or data.get("titulo") or p.stem
            if _norm_str(cand_name_raw) != target:
                continue

            study_bits = [
                data.get("version") or data.get("versao") or "",
                data.get("cite") or "",
                data.get("name") or data.get("titulo") or p.stem,
            ]
            label = " • ".join([b for b in study_bits if b]).strip(" •")
            matches.append((p, data, label or p.stem))

    matches.sort(key=lambda t: (t[2] or "").lower())
    return matches


def _select_norm_group_from_facets(scale_ref: Dict[str, Any]) -> tuple[str, str]:
    """
    Detecta grupos varrendo facets.*.norms e retorna (norm_key, norm_label).
    """
    pretty = {
        "clinico": "Clínico",
        "clínico": "Clínico",
        "comunitario": "Comunitário",
        "comunitário": "Comunitário",
        "total": "Total",
        "alto risco":"Alto Risco",
        "normativo":"Normativo",
    }

    groups = []
    seen = set()
    for f in (scale_ref.get("facets") or {}).values():
        for g in (f.get("norms") or {}).keys():
            key = str(g).strip().lower()
            if key not in seen:
                seen.add(key)
                groups.append(key)

    order_pref = ["clinico", "comunitario", "total"]
    groups = sorted(groups, key=lambda k: (order_pref.index(k) if k in order_pref else 99, k))

    if groups:
        labels = [pretty.get(k, k) for k in groups]
        idx = st.radio(
            "Selecione o grupo normativo:",
            options=list(range(len(groups))),
            format_func=lambda i: labels[i],
            horizontal=True,
            index=min(2, len(groups) - 1),
        )
        return groups[idx], labels[idx]

    fallback = [("clinico", "Clínico"), ("comunitario", "Comunitário"), ("total", "Total")]
    idx = st.radio(
        "Selecione o grupo normativo:",
        options=list(range(len(fallback))),
        format_func=lambda i: fallback[i][1],
        horizontal=True,
        index=2,
    )
    return fallback[idx]

# =========================================
# Página
# =========================================

st.title("Resumo da Escala")

# 1) Recupera escalas respondidas na sessão
data_key = "escalas_respondidas"
names_key = "escalas_display_names"
all_answers: Dict[str, Dict[Any, Any]] = st.session_state.get(data_key, {})
display_names: Dict[str, str] = st.session_state.get(names_key, {})

if not all_answers:
    st.info("Nenhuma escala respondida encontrada na sessão. Volte e aplique uma escala primeiro.")
    st.stop()

# 2) Seleção da escala aplicada
display_options = []
slug_lookup = {}
for slug, _ans in all_answers.items():
    label = display_names.get(slug, slug)
    display_options.append(label)
    slug_lookup[label] = slug

sel_label = st.selectbox("Escolha a escala para gerar o resumo:", sorted(display_options))
scale_key = slug_lookup[sel_label]

answers_raw = all_answers.get(scale_key, {})
if not answers_raw:
    st.error("Não encontrei respostas para essa escala. Verifique o salvamento.")
    st.stop()

# 3) Normaliza respostas (answers_item_ids não é mais necessário)
answers = _normalize_answers(answers_raw)

# 4) Seleção do ESTUDO normativo — match estrito por nome
candidates = _find_scale_json_candidates(sel_label)
if not candidates:
    st.warning(
        "Não encontrei estudo normativo cujo campo 'scale' (ou 'name'/'titulo') "
        f"seja idêntico ao nome da escala selecionada: “{sel_label}”.\n\n"
        "Verifique o JSON normativo e se ele está em 'scales/' ou 'bibliography/'."
    )
    st.stop()

labels = [lab for _, __, lab in candidates]
idx_study = st.selectbox("Estudo normativo", options=list(range(len(labels))), format_func=lambda i: labels[i])
selected_path, scale_ref, _ = candidates[idx_study]

# 5) Grupo normativo (extraído do estudo)
norm_group, norm_label = _select_norm_group_from_facets(scale_ref)
cite = scale_ref.get("cite", None)
if cite:
    st.info(cite, icon="📄")
st.divider()

# 6) Correção por FACETAS (PID-5 e afins)
has_facets = isinstance(scale_ref.get("facets"), dict)

if has_facets:
    try:
        facet_stats = score_pid5_facets(scale_ref, answers, use_item_mean=True)
        rows = summarize_with_norms(scale_ref, facet_stats, norm_group=norm_group, use_item_mean_for_z=True)
        df_classif = pd.DataFrame(build_classification_table(scale_ref, rows))
    except Exception as e:
        st.exception(e)
        st.error(
            "Erro durante a correção da escala com facetas. "
            "Revise 'response_map', 'reverse_items' e o bloco de normas (mean/sd) no JSON."
        )
        st.stop()

    study_name = str(scale_ref.get("name") or scale_ref.get("titulo") or Path(selected_path).stem)
    study_ver = str(scale_ref.get("version") or scale_ref.get("versao") or "").strip()


    # 7) DataFrames (visão por domínio + tabelão)
    df_rows = pd.DataFrame(rows)
    for c in ["faceta", "media_itens", "z", "percentil", "bruta", "mean_ref", "sd_ref", "norma"]:
        if c not in df_rows.columns:
            df_rows[c] = pd.NA

    domains = (scale_ref.get("domains") or {})
    facet_to_domain = {str(f): dom for dom, facets in domains.items() for f in facets}
    df_rows["dominio"] = df_rows["faceta"].map(lambda f: facet_to_domain.get(str(f), "—"))

    if not df_classif.empty and {"faceta", "classificacao"} <= set(df_classif.columns):
        df_master = df_rows.merge(df_classif[["faceta", "classificacao"]], on="faceta", how="left")
    else:
        df_master = df_rows.assign(classificacao=pd.NA)

    df_master = df_master.sort_values(["dominio", "faceta"], kind="stable").reset_index(drop=True)

    st.markdown("### Facetas por Domínio")
    for dom in df_master["dominio"].dropna().unique():
        sdf = df_master.loc[df_master["dominio"] == dom, ["faceta", "classificacao"]].reset_index(drop=True)
        sdf = sdf.rename(columns={"faceta": "Faceta", "classificacao": "Classificação"})
        st.markdown(f"**{dom}**")
        st.dataframe(
            sdf,
            use_container_width=True,
            hide_index=True,
            column_config={"Faceta": "Faceta", "Classificação": st.column_config.TextColumn("Classificação")},
        )
    

    # =========================================
    # Gráfico (interativo com Plotly + download HTML)
    # =========================================

    from modules.gauss_plot import NormSpec, compute_discrete_points
    import numpy as np, math
    import plotly.graph_objects as go

    # --- Seletor de faceta para o gráfico (mantido) ---
    _opts = (
        df_master[["dominio", "faceta"]]
        .dropna()
        .sort_values(["dominio", "faceta"], kind="stable")
    )
    _opt_labels = [f"{row.dominio} • {row.faceta}" for row in _opts.itertuples(index=False)]
    _opt_index = st.selectbox(
        "Selecione a faceta para visualizar a distribuição e percentil estimado:",
        options=list(range(len(_opt_labels))),
        format_func=lambda i: _opt_labels[i],
    )

    sel_dom = _opts.iloc[_opt_index]["dominio"]
    sel_fac = _opts.iloc[_opt_index]["faceta"]

    # --- n_itens da faceta no dicionário da escala ---
    def _facet_n_items(scale_ref: Dict[str, Any], faceta: str) -> int:
        f = (scale_ref.get("facets") or {}).get(str(faceta), {})
        if "n_items" in f and isinstance(f["n_items"], (int, float)):
            try:
                return int(f["n_items"])
            except Exception:
                pass
        items = f.get("items") or []
        return int(len(items))

    n_itens = _facet_n_items(scale_ref, sel_fac)

    # --- Recuperar médias/DP de referência e pontuação bruta observada para a faceta selecionada ---
    row_sel = df_master.loc[df_master["faceta"] == sel_fac].iloc[0]
    mean_ref = float(row_sel["mean_ref"]) if pd.notna(row_sel["mean_ref"]) else None
    sd_ref   = float(row_sel["sd_ref"])   if pd.notna(row_sel["sd_ref"])   else None
    raw_sum  = float(row_sel["bruta"])    if pd.notna(row_sel["bruta"])    else None

    if mean_ref is None or sd_ref in (None, 0) or raw_sum is None:
        st.info("Não há informações suficientes (média/DP de referência e/ou pontuação bruta) para plotar esta faceta.")
    else:
        # Heurística: se mean/sd > 3, assumimos que a norma está em soma bruta
        metric = "mean_items"
        if (mean_ref is not None and mean_ref > 3) or (sd_ref is not None and sd_ref > 3):
            metric = "raw_sum"

        spec = NormSpec(
            mean_ref=mean_ref,
            sd_ref=sd_ref,
            metric=metric,   # "mean_items" (0–3) por padrão; "raw_sum" se parecer soma bruta
            n_items=int(n_itens),
            max_per_item=3,
        )

        # ---- Calcula pontos discretos e parâmetros da Normal na escala de soma bruta
        calc = compute_discrete_points(spec, observed_raw_sum=int(raw_sum))
        mu_sum = calc["mu_sum"]
        sd_sum = calc["sd_sum"]
        max_sum = calc["max_sum"]
        pts = calc["points_df"]  # colunas: [raw_sum, mean_items, z, percentile]
        obs_pct = calc["observed_percentile"]

        # Curva contínua da densidade normal na escala de soma bruta
        xs = np.linspace(0, max_sum, 500)
        if sd_sum and sd_sum > 0:
            ys = (1.0 / (sd_sum * math.sqrt(2.0 * math.pi))) * np.exp(-0.5 * ((xs - mu_sum) / sd_sum) ** 2)
        else:
            ys = np.zeros_like(xs)

        # y dos pontos discretos (projetados na curva)
        xk = pts[:, 0]
        yk = (1.0 / (sd_sum * math.sqrt(2.0 * math.pi))) * np.exp(-0.5 * ((xk - mu_sum) / sd_sum) ** 2) if sd_sum and sd_sum > 0 else np.zeros_like(xk)

        # ---- Figura interativa (Plotly)
        fig = go.Figure()

        # Curva contínua
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", name="Densidade (Norma)",
            line=dict(color="#2196F3", width=2),
            hovertemplate="Soma: %{x:.1f}<br>Densidade: %{y:.4f}<extra></extra>"
        ))

        # Pontos discretos
        fig.add_trace(go.Scatter(
            x=xk, y=yk, mode="markers", name="Pontos possíveis",
            hovertemplate=(
                "Soma bruta: %{x}<br>"
                "Média por item: %{customdata[0]:.3f}<br>"
                "Z estimado: %{customdata[1]:.3f}<br>"
                "Percentil: %{customdata[2]:.1f}º<extra></extra>"
            ),
            marker=dict(size=6, color="#ffae00", line=dict(width=0)),
            customdata=np.column_stack([pts[:,1], pts[:,2], pts[:,3]])
        ))

        # Linha vertical para a pontuação observada
        obs = max(0, min(int(raw_sum), int(max_sum)))
        fig.add_vline(
            x=obs,
            line_dash="dash",
            line_color="#FFFFFF", 
            line_width=1
        )
        if obs_pct is not None:
            fig.add_annotation(
                x=obs, y=float((1.0 / (sd_sum * math.sqrt(2.0 * math.pi))) * math.exp(-0.5 * ((obs - mu_sum) / sd_sum) ** 2)) if sd_sum and sd_sum > 0 else 0,
                text=f"{obs}  •  {obs_pct:.1f}º pct",
                showarrow=False, yshift=14
            )

        # Layout
        fig.update_layout(
            title=f"{sel_dom} • {sel_fac} — distribuição de referência e pontos possíveis",
            xaxis_title="Pontuação bruta (soma dos itens)",
            yaxis_title="Densidade (Normal de referência)",
            template="plotly_white",
            margin=dict(l=10, r=10, t=60, b=10),
            height=380,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig.update_xaxes(autorange=True)
        fig.update_yaxes(autorange=True)

        template = "plotly_dark"  # ou "plotly_white"
        fig.update_layout(template=template)

        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

        # === Caption/descrição da faceta (logo abaixo do gráfico) ===
        desc_map = (scale_ref.get("facet_descriptions") or {})
        desc = desc_map.get(str(sel_fac))
        classif = str(row_sel["classificacao"]) if pd.notna(row_sel["classificacao"]) else ""

        if desc:
            # mostra sempre a descrição + classificação ao final
            st.caption(f"{desc} | **Classificação:** {classif}")

    # =========================================
    # Download
    # =========================================
    st.divider()

    from utils.pdf_export import build_pdf_table_and_graphs

    pdf_bytes, pdf_name = build_pdf_table_and_graphs(
        sel_label=sel_label,
        study_name=study_name,
        study_ver=study_ver,
        norm_label=norm_label,
        df_master=df_master,
        scale_ref=scale_ref,
    )

    st.download_button(
        label="Relatório Completo",
        data=pdf_bytes,
        file_name=pdf_name,
        mime="application/pdf",
        use_container_width=True,
    )


    
    # =========================================
    # Tabelão psicométrico
    # =========================================
    show_key = f"show_psy_table::{scale_key}::{norm_group}::{Path(selected_path).name}"
    show_psy = st.session_state.get(show_key, False)
    show_psy = st.toggle("📊 Mostrar tabela psicométrica completa", value=show_psy)
    st.session_state[show_key] = show_psy

    if show_psy:
        st.markdown("### Tabela Psicométrica Completa")
        df_big = (
            df_master[
                ["faceta", "dominio", "classificacao", "z", "percentil", "bruta", "media_itens", "mean_ref", "sd_ref"]
            ]
            .rename(columns={
                "faceta": "Faceta",
                "dominio": "Domínio",
                "classificacao": "Classificação",
                "z": "Z-score",
                "percentil": "Percentil",
                "bruta": "Pontuação bruta",
                "media_itens": "Média (itens)",
                "mean_ref": "Média ref. (norma)",
                "sd_ref": "DP ref. (norma)",
            })
        )
        df_big["Estudo"] = f"{study_name}{f' ({study_ver})' if study_ver else ''}"
        df_big["Grupo"] = norm_label

        st.dataframe(
            df_big,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Z-score": st.column_config.NumberColumn(format="%.3f"),
                "Percentil": st.column_config.NumberColumn(format="%.1f"),
                "Pontuação bruta": st.column_config.NumberColumn(format="%.3f"),
                "Média (itens)": st.column_config.NumberColumn(format="%.3f"),
                "Média ref. (norma)": st.column_config.NumberColumn(format="%.3f"),
                "DP ref. (norma)": st.column_config.NumberColumn(format="%.3f"),
            },
        )


else:
    # Aqui é onde plugaríamos o pipeline de escalas **sem** facetas (ex.: AQ-50, BIS-11 etc.)
    st.warning(
        "Esta escala não possui 'facets' no estudo selecionado. "
        "Pipelines de correção variam por instrumento; vamos plugar aqui a rotina específica "
        "(ex.: subescalas, percentis próprios, cortes clínicos)."
    )
