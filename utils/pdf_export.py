# utils/pdf_export.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Mapping, Any, Optional
from io import BytesIO

import matplotlib.pyplot as plt

from datetime import datetime

# ReportLab
from reportlab.lib.pagesizes import A4
from reportlab.platypus import Image
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
)
from reportlab.pdfgen import canvas
from reportlab.lib.enums import TA_JUSTIFY

from modules.gauss_plot import NormSpec, render_gauss_curve_with_points
from utils.global_variables import DISCLAIMER_TEXT

# ============
# Data models
# ============

@dataclass
class PdfTable:
    title: str
    columns: Sequence[str]
    rows: Sequence[Sequence[Any]]
    note: Optional[str] = None
    # Caso queira quebrar a tabela em múltiplas páginas automaticamente:
    repeat_header: bool = True


@dataclass
class PdfMeta:
    app_title: str = "Academia Diagnóstica"
    scale_display_name: str = ""           # ex.: "PID-5 | Autorrelato Completo"
    study_name: str = ""                   # ex.: "Markon et al. (2013)"
    study_version: str = ""                # ex.: "Versão X • Amostra Y"
    norm_group_label: str = ""             # ex.: "Clínico"
    patient_id: Optional[str] = None       # se quiser registrar identificadores
    examiner: Optional[str] = None
    extra_lines: List[str] = field(default_factory=list)  # linhas extras livres
    cite: Optional[str] = "" 

@dataclass
class PdfFigure:
    title: Optional[str]
    img_bytes: bytes
    width_cm: float = 14.0
    height_cm: float = 6.0
    caption: Optional[str] = None
    meta_text: Optional[str] = None  

@dataclass
class PdfPayload:
    meta: PdfMeta
    summary_blocks: List[Mapping[str, Any]] = field(default_factory=list)
    tables: List[PdfTable] = field(default_factory=list)
    figures: List[PdfFigure] = field(default_factory=list)  # << novo
    footer_left: str = "Academia Diagnóstica"
    footer_right: str = ""
    filename_hint: str = "resultado_escala"



# =====================
# Helpers de formatação
# =====================

def _styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="TitleYellow",
        parent=styles["Title"],
        textColor=colors.HexColor("#000000"),
        spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="H2",
        parent=styles["Heading2"],
        spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="H3",
        parent=styles["Heading3"],
        spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        name="BodySmall",
        parent=styles["BodyText"],
        fontSize=9,
        leading=12,
    ))
    styles.add(ParagraphStyle(
        name="Tiny",
        parent=styles["BodyText"],
        fontSize=8,
        leading=10,
        textColor=colors.grey,
    ))
    styles.add(ParagraphStyle(
        name="TableHead",
        parent=styles["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=11,
        textColor=colors.black,
    ))
    styles.add(ParagraphStyle(
        name="TableCell",
        parent=styles["BodyText"],
        fontSize=9,
        leading=11,
        wordWrap="CJK",   # força quebra mesmo em palavras longas
    ))
    styles.add(ParagraphStyle(
        name="Disclaimer",
        parent=styles["BodyText"],
        fontSize=8.5,
        leading=11,
        textColor=colors.grey,
        alignment=TA_JUSTIFY,  # << justificado
        spaceBefore=6,
    ))
    
    return styles


def _footer(canvas: canvas.Canvas, doc, left: str, right: str):
    canvas.saveState()
    w, h = A4
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.grey)
    margin = 1.5 * cm
    y = 1.1 * cm
    # esquerda
    canvas.drawString(margin, y, left)
    # direita
    text_right = f"{right}    ·    pág. {doc.page}"
    tw = canvas.stringWidth(text_right, "Helvetica", 8)
    canvas.drawString(w - margin - tw, y, text_right)
    canvas.restoreState()


# =====================
# Construção do PDF
# =====================

def build_results_pdf(payload: PdfPayload) -> bytes:
    """
    Gera o PDF a partir de um payload genérico (meta + blocos + tabelas).
    Retorna bytes prontos para download.
    """
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=1.8 * cm, rightMargin=1.8 * cm,
        topMargin=1.7 * cm, bottomMargin=1.7 * cm,
        title=payload.meta.scale_display_name or payload.meta.app_title,
        author="Academia Diagnóstica",
    )
    S = _styles()

    story: List[Any] = []

    # Cabeçalho
    story.append(Paragraph(payload.meta.app_title, S["TitleYellow"]))
    if payload.meta.scale_display_name:
        story.append(Paragraph(payload.meta.scale_display_name, S["H2"]))

    # Exibe a CITAÇÃO (se houver)
    if getattr(payload.meta, "cite", None):
        story.append(Paragraph(payload.meta.cite, S["BodySmall"]))

    # Linha de metadados (somente info operacional; sem Estudo/Grupo)
    meta_lines = []
    meta_lines.append(f"<b>Data:</b> {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    if payload.meta.patient_id:
        meta_lines.append(f"<b>ID:</b> {payload.meta.patient_id}")
    if payload.meta.examiner:
        meta_lines.append(f"<b>Examinador(a):</b> {payload.meta.examiner}")
    if payload.meta.norm_group_label:
        meta_lines.append(f"<b>Grupo:</b> {payload.meta.norm_group_label}")
    for x in payload.meta.extra_lines:
        meta_lines.append(x)

    if meta_lines:
        story.append(Paragraph(" &nbsp; · &nbsp; ".join(meta_lines), S["BodySmall"]))

    story.append(Spacer(1, 10))

    # Blocos-resumo (livres): cada bloco pode ter título + linhas
    for block in payload.summary_blocks:
        title = block.get("title")
        lines: Sequence[str] = block.get("lines", [])
        if title:
            story.append(Paragraph(title, S["H3"]))
        for ln in lines:
            story.append(Paragraph(ln, S["BodySmall"]))
        story.append(Spacer(1, 6))

    # Tabelas
    for idx, t in enumerate(payload.tables):
        story.append(Spacer(1, 6))
        story.append(Paragraph(t.title, S["H3"]))

        # Header e células como Paragraph (com quebra)
        head = [Paragraph(str(h), S["TableHead"]) for h in t.columns]
        body = [[_as_para(c, S["TableCell"]) for c in row] for row in t.rows]
        data = [head] + body

        table = Table(
            data,
            repeatRows=1 if t.repeat_header else 0,
            hAlign="LEFT",
            colWidths=_auto_col_widths(
                [[str(x) for x in t.columns]] + [[_fmt_cell(c) for c in row] for row in t.rows],
                max_width=A4[0] - (doc.leftMargin + doc.rightMargin),
            ),
            splitByRow=1,  # quebra a tabela entre páginas sem “comer” linhas
        )

        table.setStyle(TableStyle([
            # header
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F5F5F5")),
            ("TEXTCOLOR",  (0, 0), (-1, 0), colors.black),
            ("ALIGN",      (0, 0), (-1, 0), "CENTER"),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 8),

            # corpo
            ("VALIGN",     (0, 1), (-1, -1), "MIDDLE"),
            ("GRID",       (0, 0), (-1, -1), 0.25, colors.HexColor("#CCCCCC")),

            # paddings generosos para evitar “colagem” visual
            ("LEFTPADDING",  (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING",   (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 4),

            # se quiser forçar quebra em palavras muito longas sem espaço:
            ("WORDWRAP", (0, 0), (-1, -1), "CJK"),
        ]))
        story.append(table)

        if t.note:
            story.append(Spacer(1, 4))
            story.append(Paragraph(t.note, S["Tiny"]))

        if idx < len(payload.tables) - 1:
            story.append(PageBreak())

    # --- build_results_pdf(): loop das figuras ---
    for fig in (payload.figures or []):
        if fig.title:
            story.append(Paragraph(fig.title, S["H3"]))
        # 1) gráfico primeiro
        story.append(Image(BytesIO(fig.img_bytes), width=fig.width_cm * cm, height=fig.height_cm * cm))
        story.append(Spacer(1, 6))
        # 2) descrição logo abaixo do gráfico
        if getattr(fig, "caption", None):
            story.append(Paragraph(fig.caption, S["BodySmall"]))
            story.append(Spacer(1, 2))
        # 3) classificação + norma (média/DP) na linha seguinte
        if getattr(fig, "meta_text", None):
            story.append(Paragraph(fig.meta_text, S["Tiny"]))
            story.append(Spacer(1, 10))

    story.append(Spacer(1, 12))
    story.append(Paragraph(DISCLAIMER_TEXT, S["Disclaimer"]))

    # Footer handler
    def on_page(c, d):
        _footer(c, d, payload.footer_left, payload.footer_right or payload.meta.app_title)

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    return buf.getvalue()


# =====================
# Utilidades auxiliares
# =====================

def _is_number(x) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False

def _as_para(val, style):
    from reportlab.platypus import Paragraph
    # None → traço; números ficam como string formatada pelo _fmt_cell
    if val is None:
        return Paragraph("—", style)
    s = _fmt_cell(val)
    return Paragraph(s, style)

def _fmt_cell(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        # formatação simples; ajuste conforme a coluna se quiser
        return f"{v:.3f}".rstrip("0").rstrip(".")
    return str(v)

def _auto_col_widths(data, max_width: float) -> Optional[List[float]]:
    """
    Larguras proporcionais com piso maior para colunas textuais.
    Detecta 'numéricas' olhando 8 primeiras linhas (exclui cabeçalho).
    """
    if not data or not data[0]:
        return None

    header = data[0]
    body = data[1:]

    n_cols = len(header)
    # detecta colunas numéricas
    numeric = []
    sample_rows = body[:8] if body else []
    for j in range(n_cols):
        col_vals = [r[j] for r in sample_rows if len(r) > j]
        is_num = len(col_vals) > 0 and all(_is_number(v) for v in col_vals)
        numeric.append(is_num)

    # score de “comprimento” por coluna
    lens = [0.0] * n_cols
    for row in data:
        for j, cell in enumerate(row):
            lens[j] += len(str(cell))

    # pisos: texto 3.0cm, numérico 2.2cm
    from reportlab.lib.units import cm
    min_text = 3.0 * cm
    min_num = 2.2 * cm

    # proporção básica
    total = sum(lens) or 1.0
    raw_widths = [(l / total) * max_width for l in lens]

    # aplica pisos por tipo
    widths = []
    for j, w in enumerate(raw_widths):
        floor = min_num if numeric[j] else min_text
        widths.append(max(floor, w))

    # normaliza para caber no max_width
    k = max_width / max(1e-6, sum(widths))
    return [w * k for w in widths]


# =====================
# Montagem
# =====================

def build_pdf_table_and_graphs(
    *,
    sel_label: str,
    study_name: str,
    study_ver: str,
    norm_label: str,
    df_master: Any,   # precisa ter colunas: faceta, dominio, z, percentil, media_itens, mean_ref, sd_ref, bruta
    scale_ref: dict,
    filename_hint: Optional[str] = None,
) -> tuple[bytes, str]:
    import pandas as pd

    # -------- Meta / payload base
    meta = PdfMeta(
        app_title="Academia Diagnóstica",
        scale_display_name=sel_label,
        study_name=study_name,
        study_version=study_ver,
        norm_group_label=norm_label,
    )
    
    # Insere a citação no meta
    meta.cite = (scale_ref.get("cite") or "")

    payload = PdfPayload(
        meta=meta,
        summary_blocks=[],
        tables=[],
        figures=[],
        footer_left="Academia Diagnóstica",
        footer_right="Relatório de resultados",
        filename_hint=filename_hint or f"resultado_{sel_label.replace(' ', '_').replace('|','')}",
    )

    # -------- Tabela principal
    header_map = {
        "faceta": "Faceta",
        "dominio": "Domínio",
        "z": "Z-score",
        "percentil": "Percentil",
        "media_itens": "Média",
        "bruta": "Escore",
    }
    wanted = ["faceta", "dominio", "z", "percentil", "media_itens", "bruta"]  # << sem mean_ref/sd_ref
    cols_src = [c for c in wanted if c in df_master.columns]

    table = PdfTable(
        title="Resultados psicométricos",
        columns=[header_map[c] for c in cols_src],
        rows=df_master[cols_src].values.tolist(),
        note="Z-score: (valor − média de referência) / DP de referência.",
    )
    payload.tables.append(table)

    desc_map = (scale_ref.get("facet_descriptions") or {}) 
    
    # -------- Gráficos — um por faceta
    for _, row in df_master.iterrows():
        fac = row.get("faceta")
        dom = row.get("dominio")
        mean_ref, sd_ref, raw_sum = row.get("mean_ref"), row.get("sd_ref"), row.get("bruta")

        # pula facetas sem dados suficientes
        if pd.isna(mean_ref) or pd.isna(sd_ref) or float(sd_ref) == 0.0 or pd.isna(raw_sum):
            continue

        metric = "mean_items" if (float(mean_ref) < 3 and float(sd_ref) < 3) else "raw_sum"

        fdef = (scale_ref.get("facets") or {}).get(str(fac), {})
        n_itens = int(fdef.get("n_items") or len(fdef.get("items") or []))

        spec = NormSpec(mean_ref=float(mean_ref), sd_ref=float(sd_ref), metric=metric, n_items=n_itens)
        fig, _ = render_gauss_curve_with_points(
            spec,
            observed_raw_sum=int(float(raw_sum)),
            title=f"{dom} • {fac}",
        )

        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", pad_inches=0.2)
        plt.close(fig)
        buf.seek(0)

        caption = desc_map.get(str(fac))  # descrição curta da faceta
        classif = row.get("classificacao") or "—"
        # formata média/DP da norma (3 casas)
        meta_text = f"Classificação: {classif} · Norma: média = {float(mean_ref):.3f}, DP = {float(sd_ref):.3f}"

        payload.figures.append(
            PdfFigure(
                title=f"{dom} • {fac}",
                img_bytes=buf.getvalue(),
                width_cm=14.0,
                height_cm=6.0,
                caption=caption,
                meta_text=meta_text,   # << exibido sob a descrição
            )
        )

    # -------- Render final
    pdf_bytes = build_results_pdf(payload)
    return pdf_bytes, f"{payload.filename_hint}.pdf"


