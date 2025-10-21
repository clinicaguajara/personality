# modules/gauss_plot.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import math
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class NormSpec:
    mean_ref: float      # média de referência (na MÉTRICA informada em `metric`)
    sd_ref: float        # DP de referência (na MÉTRICA informada em `metric`)
    metric: str          # "mean_items" (0–3) ou "raw_sum"
    n_items: int         # número de itens da faceta
    max_per_item: int = 3  # Likert máximo por item (default 3 → range bruto 0..3*n)


def _normal_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return np.zeros_like(x)
    return (1.0 / (sigma * math.sqrt(2.0 * math.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _normal_cdf(x: np.ndarray) -> np.ndarray:
    # CDF da Normal padrão via erf
    return 0.5 * (1.0 + (2.0 / math.sqrt(math.pi)) * np.vectorize(math.erf)(x / math.sqrt(2.0)))


def _as_sum_units(norm: NormSpec) -> Tuple[float, float, int]:
    """
    Converte média/DP de referência para a escala de SOMA BRUTA (0..max_sum), que é o eixo do gráfico.
    - Se vierem em "mean_items", converte: μ_sum = μ_mean * n,  σ_sum = σ_mean * n.
    - Se vierem em "raw_sum", mantém.
    Retorna (mu_sum, sd_sum, max_sum).
    """
    max_sum = norm.n_items * norm.max_per_item
    if norm.metric == "mean_items":
        mu_sum = norm.mean_ref * norm.n_items
        sd_sum = norm.sd_ref * norm.n_items
        return mu_sum, sd_sum, max_sum
    elif norm.metric == "raw_sum":
        return norm.mean_ref, norm.sd_ref, max_sum
    else:
        raise ValueError("metric deve ser 'mean_items' ou 'raw_sum'.")


def compute_discrete_points(norm: NormSpec, *, observed_raw_sum: Optional[int] = None):
    """
    Retorna um dicionário com:
      - points_df: np.ndarray com colunas [raw_sum, mean_items, z, percentile]
      - observed_percentile: Optional[float]
    """
    mu_sum, sd_sum, max_sum = _as_sum_units(norm)

    raw_vals = np.arange(0, max_sum + 1, dtype=int)  # 0..max_sum inclusive
    # z e percentil estimados na escala de soma bruta
    z = (raw_vals - mu_sum) / sd_sum if sd_sum not in (0, None) else np.full_like(raw_vals, np.nan, dtype=float)
    pct = _normal_cdf(z) * 100.0
    # << clamp vetor
    pct = np.clip(pct, 0.0, 100.0)

    mean_items = raw_vals / norm.n_items if norm.n_items else np.zeros_like(raw_vals, dtype=float)

    points_df = np.column_stack([raw_vals, mean_items, z, pct])  # shape: (N, 4)

    obs_pct = None
    if observed_raw_sum is not None:
        # clamp para [0, max_sum]
        obs = max(0, min(int(observed_raw_sum), max_sum))
        z_obs = (obs - mu_sum) / sd_sum if sd_sum not in (0, None) else float("nan")
        obs_pct = float(_normal_cdf(np.array([z_obs]))[0] * 100.0)
        # << clamp escalar
        obs_pct = max(0.0, min(100.0, obs_pct))

    return {
        "points_df": points_df,
        "observed_percentile": obs_pct,
        "mu_sum": mu_sum,
        "sd_sum": sd_sum,
        "max_sum": max_sum,
    }


def render_gauss_curve_with_points(
    norm: NormSpec,
    *,
    observed_raw_sum: Optional[int] = None,
    title: Optional[str] = None,
):
    """
    Desenha:
      - Curva Normal em SOMA BRUTA (0..max_sum), usando μ/σ convertidos para soma.
      - Todos os pontos discretos 0..max_sum (bolinhas) na curva.
      - Linha vertical e anotação do percentil no ponto observado (se fornecido).
    Retorna (fig, aux), onde aux inclui a tabela de pontos (np.ndarray).
    """
    calc = compute_discrete_points(norm, observed_raw_sum=observed_raw_sum)
    mu_sum = calc["mu_sum"]
    sd_sum = calc["sd_sum"]
    max_sum = calc["max_sum"]
    points_df = calc["points_df"]
    obs_pct = calc["observed_percentile"]

    # domínio x (um pouco além para a curva “respirar”)
    x_lo = -0.5
    x_hi = max_sum + 0.5
    xs = np.linspace(x_lo, x_hi, 500)
    ys = _normal_pdf(xs, mu_sum, sd_sum)

    # y dos pontos discretos (projetados na curva)
    xk = points_df[:, 0]
    yk = _normal_pdf(xk, mu_sum, sd_sum)

    # >>> altere a criação da figura para usar constrained_layout (melhor encaixe)
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=110, constrained_layout=True)
    ax.plot(xs, ys, linewidth=1.6)           # curva
    ax.scatter(xk, yk, s=16, zorder=3)       # pontos discretos

    # ==== NOVO: margens seguras nos eixos ====
    y_top = float(ys.max() if len(ys) else 1.0)
    ax.set_xlim(0, max_sum)
    ax.set_ylim(0, y_top * 1.25)             # folga vertical p/ texto acima da curva
    ax.margins(x=0.02)                       # folga mínima no X

    # marcação da pontuação observada (com anotação “esperta”)
    if observed_raw_sum is not None:
        obs = max(0, min(int(observed_raw_sum), max_sum))
        y_obs = _normal_pdf(np.array([obs]), mu_sum, sd_sum)[0]

        # posição relativa no eixo X para decidir o lado do rótulo
        rel = obs / max(1e-9 + max_sum, 1.0)
        if rel < 0.15:
            # muito à esquerda → joga texto para a direita
            xytext = (35, 0)
            ha = "left"
            va = "bottom"
        elif rel > 0.85:
            # muito à direita → joga texto para a esquerda
            xytext = (-35, 0)
            ha = "right"
            va = "bottom"
        else:
            # centro → acima
            xytext = (0, 18)
            ha = "center"
            va = "bottom"

        ax.axvline(obs, linestyle="--", linewidth=1.2)

        if obs_pct is not None:
            ax.annotate(
                f"{obs}  •  {obs_pct:.1f}º pct",
                xy=(obs, y_obs),
                xytext=xytext,
                textcoords="offset points",
                ha=ha, va=va,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.9),
                arrowprops=dict(arrowstyle="->", lw=0.8, color="0.4"),
                annotation_clip=True,   # mantém dentro do eixo
            )

    # eixos e rótulos (mantidos)
    ax.set_xlabel("Pontuação bruta (soma dos itens)")
    ax.set_ylabel("Densidade (Normal de referência)")
    if title:
        ax.set_title(title)

    ax.grid(alpha=0.25, linestyle=":")
    # fig.tight_layout()  # NÃO precisa com constrained_layout=True

    return fig, {
        "points_df": points_df,         # colunas: [raw_sum, mean_items, z, percentile]
        "observed_percentile": obs_pct,
        "mu_sum": mu_sum,
        "sd_sum": sd_sum,
        "max_sum": max_sum,
    }