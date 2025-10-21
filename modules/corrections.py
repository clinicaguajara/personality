#modules\correction.py

# =========================
# Necessary Imports
# =========================

import json
import pandas as pd
import streamlit as st

from math import erf, sqrt
from typing import Dict, Any, List, Tuple, Optional

# --- Carregamento ---

def load_scale(json_path: str) -> Dict[str, Any]:
    """<docstrings>
    Carrega o dicionário da escala PID-5 a partir de um arquivo JSON.

    Calls:
        json.load(): Função para ler JSON | built-in.
        open(): Função para abrir arquivo | built-in.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

# --- Utilidades ---

def _reverse_value(v: Optional[int], max_val: int = 3) -> Optional[int]:
    """<docstrings>
    Inverte a resposta Likert (0..max_val) quando necessário.

    Calls:
        int(): Função para conversão numérica | built-in.
    """
    if v is None:
        return None
    return max_val - int(v)

def _z_to_percentile(z: Optional[float]) -> Optional[float]:
    if z is None:
        return None
    pct = 50.0 * (1.0 + erf(z / sqrt(2.0)))
    # << clamp para evitar 100.0000001
    if pct < 0.0:
        pct = 0.0
    elif pct > 100.0:
        pct = 100.0
    return pct

def _get_norm_params(scale: Dict[str, Any], facet: str, group: str) -> Tuple[Optional[float], Optional[float]]:
    """<docstrings>
    Recupera mean/sd da faceta conforme o grupo normativo.
    Faz fallback para default_norm_group ou para qualquer grupo disponível.
    """
    fdata = scale["facets"][facet]
    # Novo formato preferido
    norms = fdata.get("norms")
    if norms and group in norms:
        return norms[group].get("mean"), norms[group].get("sd")

    # Fallbacks: default -> algum -> antigo formato
    if norms:
        gdef = scale.get("default_norm_group")
        if gdef and gdef in norms:
            return norms[gdef].get("mean"), norms[gdef].get("sd")
        if len(norms) > 0:
            any_group = next(iter(norms.values()))
            return any_group.get("mean"), any_group.get("sd")

    # Antigo formato (mean/sd na raiz da faceta)
    return fdata.get("mean"), fdata.get("sd")

def _compute_z(x: Optional[float], mean: Optional[float], sd: Optional[float]) -> Optional[float]:
    """<docstrings>
    (x - mean)/sd com proteções básicas.
    """
    if x is None or mean is None or sd in (None, 0):
        return None
    return (x - mean) / sd

# ---------- Resumo com normas ----------
def summarize_with_norms(
    scale: Dict[str, Any],
    facet_stats: Dict[str, Dict[str, Any]],
    norm_group: str,
    *,
    use_item_mean_for_z: bool = True
) -> List[Dict[str, Any]]:
    """<docstrings>
    Para cada faceta, busca mean/sd do grupo, calcula z e percentil.
    Retorna linhas para DataFrame final.
    """
    rows: List[Dict[str, Any]] = []
    for facet, stats in facet_stats.items():
        mean_ref, sd_ref = _get_norm_params(scale, facet, norm_group)

        base_value = stats["mean_items"] if use_item_mean_for_z else stats["raw_sum"]
        z = _compute_z(base_value, mean_ref, sd_ref)
        pct = _z_to_percentile(z)

        rows.append({
            "faceta": facet,
            "media_itens": None if stats["mean_items"] is None else round(stats["mean_items"], 3),
            "z": None if z is None else round(z, 3),
            "percentil": None if pct is None else round(pct, 1),
            "bruta": None if stats["raw_sum"] is None else round(stats["raw_sum"], 3),
            "norma": norm_group,
            "mean_ref": mean_ref,
            "sd_ref": sd_ref,
        })
    return rows

# ---------- Classificação separada ----------
def build_classification_table(
    scale: Dict[str, Any],
    rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """<docstrings>
    Gera tabela de classificação baseada nos limites do JSON (campo 'classification').
    Usa z de cada linha para escolher label_above/below.
    """
    classes = scale.get("classification", [])
    out = []
    for r in rows:
        z = r.get("z")
        if z is None:
            label = None
        else:
            abs_z = abs(z)
            label_above = None
            label_below = None

            # encontra a primeira regra aplicável
            chosen = None
            for rule in classes:
                max_abs = rule.get("max_abs_z")
                if max_abs is None or abs_z <= float(max_abs):
                    chosen = rule
                    break

            if chosen is None:
                label = None
            else:
                label_above = chosen.get("label_above")
                label_below = chosen.get("label_below")
                label = label_above if z >= 0 else label_below

        out.append({
            "faceta": r["faceta"],
            "z": r["z"],
            "classificacao": label
        })
    return out

# =========================
# PID-5
# =========================

def score_pid5_facets(
    scale: Dict[str, Any],
    answers: Dict[int, Any],
    *,
    use_item_mean: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """<docstrings>
    Calcula pontuação bruta e média por item para cada faceta.
    - answers: dict mapeando número do item (1..N) -> resposta (0..3) ou string equivalente ao response_map.
    - Respeita reverse_items da escala.
    Calls:
        dict.get(): Método do dict | instanciado por dict.
        _reverse_value(): Função para inversão.
    """
    response_map = scale.get("response_map", {})
    reverse_items = set(scale.get("reverse_items", []))

    out: Dict[str, Dict[str, Any]] = {}
    for facet, fdata in scale["facets"].items():
        item_ids: List[int] = fdata["items"]
        scored: List[int] = []

        for item in item_ids:
            raw = answers.get(item)
            if raw is None:
                continue
            # Converte strings do response_map para 0..3, se necessário
            if isinstance(raw, str):
                if raw not in response_map:
                    continue
                val = response_map[raw]
            else:
                val = int(raw)

            # Aplica inversão se item estiver em reverse_items
            if item in reverse_items:
                val = _reverse_value(val, max_val=3)

            scored.append(val)

        if len(scored) == 0:
            raw_sum = None
            mean_items = None
        else:
            raw_sum = float(sum(scored))
            mean_items = (raw_sum / len(scored)) if use_item_mean else None

        out[facet] = {
            "raw_sum": raw_sum,
            "mean_items": mean_items,
            "n_answered": len(scored),
            "n_items": len(item_ids),
        }
    return out

def render_pid5_results(
    scale: Dict[str, Any],
    answers: Dict[int, Any],
    *,
    norm_group: Optional[str] = None,        # <- novo: usa o grupo já escolhido fora
    default_norm: Optional[str] = None,
    use_item_mean_for_z: bool = True,
    show_norm_selector: bool = True,         # <- opcional: permite ligar/desligar o radio interno
) -> None:
    """<docstrings>
    Renderiza no Streamlit:
    1) Tabela principal (faceta | média_itens | z | percentil | bruta)
    2) Tabela de classificação separada
    """

    # grupos disponíveis
    norm_groups = scale.get("norm_groups", ["total"])
    fallback_default = scale.get("default_norm_group", norm_groups[0])

    # Decide o grupo a usar:
    # 1) se veio `norm_group`, usa ele e NÃO mostra radio
    # 2) senão, se show_norm_selector=True, mostra radio interno
    # 3) senão, usa default_norm (ou fallback)
    if norm_group is not None:
        chosen_group = norm_group
    elif show_norm_selector:
        # mostra o radio interno (com default coerente)
        default_group = default_norm or fallback_default
        try:
            idx = norm_groups.index(default_group)
        except ValueError:
            idx = 0
        chosen_group = st.radio(
            "Norma para correção",
            options=norm_groups,
            index=idx,
            horizontal=True,
        )
    else:
        chosen_group = default_norm or fallback_default

    # ---- Pontuação e resumo
    facet_stats = score_pid5_facets(scale, answers, use_item_mean=True)
    rows = summarize_with_norms(
        scale, facet_stats, chosen_group, use_item_mean_for_z=use_item_mean_for_z
    )

    df = pd.DataFrame(
        rows,
        columns=["faceta", "media_itens", "z", "percentil", "bruta", "norma", "mean_ref", "sd_ref"]
    )

    st.subheader("Resultados por Faceta")
    st.dataframe(
        df[["faceta", "media_itens", "z", "percentil", "bruta"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "faceta": "Faceta",
            "media_itens": st.column_config.NumberColumn(
                "Média (itens)", format="%.3f", help="Média das respostas (0–3)"
            ),
            "z": st.column_config.NumberColumn(
                "Z-score", format="%.3f", help="(valor - média de referência) / desvio-padrão"
            ),
            "percentil": st.column_config.NumberColumn(
                "Percentil", format="%.1f", help="CDF da Normal padrão (em %)"
            ),
            "bruta": st.column_config.NumberColumn(
                "Pontuação bruta", format="%.3f", help="Soma das respostas pós-reversão"
            ),
        }
    )

    # --- Classificação separada ---
    class_rows = build_classification_table(scale, rows)
    dfc = pd.DataFrame(class_rows, columns=["faceta", "z", "classificacao"])

    # Formatação amigável
    dfc["z"] = pd.Series(dfc["z"], dtype="Float64")
    dfc_display = dfc.fillna(pd.NA)

    st.subheader("Classificação")
    st.dataframe(
        dfc_display[["faceta", "z", "classificacao"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "faceta": "Faceta",
            "z": st.column_config.NumberColumn("Z-score", format="%.3f"),
            "classificacao": st.column_config.TextColumn("Classificação"),
        },
    )



