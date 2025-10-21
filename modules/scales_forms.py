# modules/scales_forms.py

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import streamlit as st
from streamlit_scroll_to_top import scroll_to_here

# =========================
# Data models
# =========================

@dataclass
class ScaleItem:
    id: str
    text: str


@dataclass
class ScaleData:
    name: str
    answer_options: List[str]
    items: List[ScaleItem]
    instruction_html: Optional[str] = None


@dataclass
class ScaleConfig:
    page_size: int = 36
    allow_blank: bool = True
    blank_sentinel: str = "__BLANK__"
    show_id_badge: bool = True
    test_prefill: bool = False  # for development/demo
    form_instruction_html: Optional[str] = None


@dataclass
class ScaleKeys:
    slug: str
    page_key: str
    answers_key: str
    init_key: str
    signature_key: str


SCROLL_FLAG = "_needs_scroll_top"

# =========================
# Public API
# =========================

def _inject_styles_once() -> None:
    key = "_styles_scales_forms"
    if not st.session_state.get(key):
        st.markdown("""
        <style>
        .item-badge{
            display:inline-block;
            background:#2196F3;
            color:#fff;
            border:1px solid #0b74d6;
            border-radius:10px;
            padding:2px 10px;
            font-weight:700;
            min-width:2.4rem;
            text-align:center;
        }
        .item-row{ display:flex; align-items:flex-start; gap:.6rem; margin:.4rem 0 .2rem 0; }
        .item-text{ flex:1; }
        </style>
        """, unsafe_allow_html=True)
        st.session_state[key] = True


def _scroll_to_top_if_needed() -> None:
    if st.session_state.get(SCROLL_FLAG):
        st.session_state[SCROLL_FLAG] = False
        k = st.session_state.get("_scroll_exec_counter", 0)
        st.session_state["_scroll_exec_counter"] = k + 1
        # rola até Y=0 (topo). key único força execução a cada navegação
        scroll_to_here(0, key=f"top_{k}")


def render_scale_selector(scales_dir: str | Path) -> None:
    """<docstrings>
    Renderiza um seletor de escalas a partir de arquivos .json e chama o formulário.

    Returns:
        None. Persiste as respostas em st.session_state ao submeter a última página.

    Calls:
        Path.glob(): Busca de arquivos | instanciado por pathlib.Path.
        json.load(): Carrega JSON de arquivo | built-in.
        render_scale_form(): Controlador do formulário por página | definida neste arquivo.
        st.session_state.__setitem__(): Define valores no estado | instanciado por st.session_state.
        st.selectbox(): Widget de seleção | instanciado por streamlit.
        st.form(): Cria formulário | instanciado por streamlit.
    """
    scales_dir = Path(scales_dir)
    files = sorted(scales_dir.glob("*.json"))
    if not files:
        st.info("Nenhuma escala .json encontrada no diretório informado.")
        return

    # Mapeia "label bonitinho" -> Path
    labels = []
    lookup: Dict[str, Path] = {}
    for f in files:
        try:
            with f.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            name = str(data.get("name") or data.get("titulo") or f.stem)
        except Exception:
            name = f.stem
        labels.append(name)
        lookup[name] = f

    st.subheader("Selecione a escala")
    label = st.selectbox("Escala", labels, index=0)
    cfg = ScaleConfig(
        page_size=36,
        allow_blank=True,
        blank_sentinel="__BLANK__",
        show_id_badge=True,
        test_prefill=True,
    )

    submitted, answers = render_scale_form(lookup[label], cfg=cfg)

    if submitted:
        key_data = "escalas_respondidas"
        key_names = "escalas_display_names"
        if key_data not in st.session_state:
            st.session_state[key_data] = {}
        if key_names not in st.session_state:
            st.session_state[key_names] = {}

        norm_key = label.strip().lower()
        st.session_state[key_data][norm_key] = answers
        st.session_state[key_names][norm_key] = label

        st.switch_page("pages/1_Resumo.py")


def render_scale_form(
    scale_ref: str | Path | Dict[str, Any],
    *,
    cfg: Optional[ScaleConfig] = None,
) -> Tuple[bool, Dict[str, str]]:
    cfg = cfg or ScaleConfig()

    raw = _load_scale(scale_ref)
    data = _normalize_scale(raw)

    keys = _build_keys(data.name)

    # (opcional) CSS global, se você tiver a função no arquivo
    try:
        _inject_styles_once()
    except Exception:
        pass

    signature = _compute_signature(cfg, data)
    _ensure_initial_state(cfg, keys, data.items, data.answer_options, signature)

    _scroll_to_top_if_needed()

    current_page = st.session_state[keys.page_key]
    total_pages = _total_pages(len(data.items), cfg.page_size)

    _render_header(data, current_page, total_pages)

    # Form por página -> o próprio botão valida e avança
    go_next, finished = _render_items_form(cfg, data, keys, current_page, total_pages)

    if go_next:
        st.session_state[keys.page_key] = min(total_pages, current_page + 1)
        st.session_state[SCROLL_FLAG] = True
        st.rerun()

    if finished:
        answers = st.session_state[keys.answers_key].copy()
        return True, answers

    return False, {}


# =========================
# Internals
# =========================

def _load_scale(scale_ref: str | Path | Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(scale_ref, dict):
        return scale_ref
    if isinstance(scale_ref, Path):
        text = scale_ref.read_text(encoding="utf-8")
        return json.loads(text)
    # aceita string com caminho ou com JSON
    try:
        p = Path(str(scale_ref))
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return json.loads(str(scale_ref))


def _normalize_scale(obj: Dict[str, Any]) -> ScaleData:
    # Nome
    name = str(obj.get("name") or obj.get("titulo") or "Escala")

    # Opções de resposta (garante lista de strings)
    options = obj.get("respostas") or obj.get("answers") or []

    if isinstance(options, dict):
        # se vier como dict (p.ex. { "Nunca": 0, "Às vezes": 1, ... }), preserve a ordem natural das chaves
        options = list(options.keys())
    elif isinstance(options, (str, int, float)):
        options = [str(options)]
    else:
        options = [str(o) for o in options]

    # Garante sentinel no início (para permitir "em branco" depois)
    if options and options[0] != "__BLANK__":
        options = ["__BLANK__"] + options

    # Itens: aceita lista de dicts OU lista de strings
    items_raw = obj.get("items") or obj.get("itens") or []
    items: List[ScaleItem] = []
    for idx, it in enumerate(items_raw, start=1):
        if isinstance(it, dict):
            it_id = str(it.get("id") or it.get("numero") or it.get("index") or idx)
            it_text = str(it.get("text") or it.get("texto") or it.get("label") or "")
        else:
            # string, número, etc. → vira texto; id = posição
            it_id = str(idx)
            it_text = str(it)
        items.append(ScaleItem(id=it_id, text=it_text))
    
    ui = obj.get("ui") if isinstance(obj.get("ui"), dict) else {}
    instr = (
        obj.get("instructions")
    )

    instruction_html = str(instr) if instr is not None else None

    return ScaleData(name=name, answer_options=list(options), items=items, instruction_html=instruction_html)


def _slugify(name: str) -> str:
    s = "".join(ch if ch.isalnum() else "_" for ch in name.strip())
    s = "_".join([t for t in s.split("_") if t])
    return s.lower()


def _build_keys(scale_name: str) -> ScaleKeys:
    slug = _slugify(scale_name)
    return ScaleKeys(
        slug=slug,
        page_key=f"{slug}__page",
        answers_key=f"{slug}__answers",
        init_key=f"{slug}__initialized",
        signature_key=f"{slug}__signature",
    )


def _compute_signature(cfg: ScaleConfig, data: ScaleData) -> Tuple[Any, ...]:
    return (
        data.name,
        tuple(o for o in data.answer_options),
        len(data.items),
        cfg.allow_blank,
        cfg.page_size,
        cfg.show_id_badge,
    )


def _ensure_initial_state(
    cfg: ScaleConfig,
    keys: ScaleKeys,
    items: Sequence[ScaleItem],
    options: Sequence[str],
    signature: Tuple[Any, ...],
) -> None:
    # Página
    if keys.page_key not in st.session_state:
        st.session_state[keys.page_key] = 1

    need_init = (keys.answers_key not in st.session_state) or (st.session_state.get(keys.signature_key) != signature)

    if need_init:
        if cfg.test_prefill:
            default_value = options[1] if len(options) > 1 else (options[0] if options else "")
        else:
            default_value = cfg.blank_sentinel if cfg.allow_blank else (options[1] if len(options) > 1 else options[0])

        st.session_state[keys.answers_key] = {str(it.id): default_value for it in items}
        st.session_state[keys.init_key] = True
        st.session_state[keys.signature_key] = signature


def _total_pages(n_items: int, page_size: int) -> int:
    return max(1, (n_items + page_size - 1) // page_size)


def _page_window(page: int, page_size: int, n_items: int) -> Tuple[int, int]:
    start = (page - 1) * page_size
    end = min(n_items, start + page_size)
    return start, end


def _render_header(data: ScaleData, current_page: int, total_pages: int) -> None:
    st.markdown(f"### {data.name}")
    st.caption(f"Página {current_page} de {total_pages}")
    st.divider()


def _render_item_row(
    cfg: ScaleConfig,
    keys: ScaleKeys,
    it: ScaleItem,
    options: Sequence[str],
    current_value: str,
) -> None:
    display_options = list(options)
    if not cfg.allow_blank:
        display_options = [o for o in display_options if o != cfg.blank_sentinel]

    # Ajusta valor inválido
    if current_value not in display_options:
        current_value = display_options[0] if display_options else ""

    try:
        idx = display_options.index(current_value)
    except ValueError:
        idx = 0

    # Linha do item
    badge_html = (
        f"<span class='item-badge' "
        f"style='display:inline-block;background:#2196F3;color:#fff;border:1px solid #fff;"
        f"border-radius:10px;padding:2px 10px;font-weight:700;min-width:2.4rem;text-align:center;'>"
        f"#{it.id}</span>"
    ) if cfg.show_id_badge else ""

    st.markdown(
        f"""
        <div class="item-row" style="display:flex;align-items:flex-start;gap:.6rem;margin:.4rem 0 .4rem 0;">
            {badge_html}
            <div class="item-text" style="flex:1;">{it.text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Selectbox
    select_key = f"{keys.slug}__sb__{it.id}"

    def _fmt(idx: int) -> str:
        val = display_options[idx]
        return "" if val == cfg.blank_sentinel else str(val)

    chosen_idx = st.selectbox(
        " ",
        options=list(range(len(display_options))),
        index=idx,
        key=select_key,
        format_func=_fmt,            # <<<<<< aqui
        label_visibility="collapsed",
    )
    chosen_value = display_options[chosen_idx]
    st.session_state[keys.answers_key][str(it.id)] = chosen_value


def _validate_answers(
    cfg: ScaleConfig,
    data: ScaleData,
    answers: Dict[str, str],
    scope_items: Optional[Sequence[ScaleItem]] = None,
) -> List[Dict[str, Any]]:
    # Se allow_blank=True, ainda assim exigimos resposta antes de avançar de página
    # (blank_sentinel é tratado como "faltando").
    scope = scope_items if scope_items is not None else data.items
    missing: List[Dict[str, Any]] = []
    # Página de cada item para compor a tabela (informativa)
    for pos, it in enumerate(data.items, start=1):
        if scope_items is not None and it not in scope_items:
            continue
        val = answers.get(str(it.id), cfg.blank_sentinel)
        if val == cfg.blank_sentinel or val is None or val == "":
            page = ((pos - 1) // cfg.page_size) + 1
            missing.append({"Item": it.id, "Página": page, "Pergunta": it.text})
    return missing


def _render_items_form(
    cfg: ScaleConfig,
    data: ScaleData,
    keys: ScaleKeys,
    current_page: int,
    total_pages: int,
) -> Tuple[bool, bool]:
    start, end = _page_window(current_page, cfg.page_size, len(data.items))
    page_items = data.items[start:end]

    form_key = f"{keys.slug}__form__{current_page}"
    go_next = False
    finished = False

    instr_raw = data.instruction_html
    instr_raw = data.instruction_html
    if instr_raw:
        st.markdown(
            "<h4 style='color:#FFB300; margin:0 0 .25rem 0;'>Instruções</h4>",
            unsafe_allow_html=True
        )

        import re, html as ihtml
        # tira qualquer HTML que eventualmente venha no JSON e mantém só texto puro
        txt = re.sub(r"<[^>]+>", "", str(instr_raw)).strip()
        # escapa o texto para não virar HTML e depois trata quebras de linha
        txt = ihtml.escape(txt)
        txt = txt.replace("\n\n", "<br><br>").replace("\n", "<br>")

        st.markdown(
            f"""<div style="text-align: justify; text-justify: inter-word; line-height: 1.45;">
                    {txt}
                </div>""",
            unsafe_allow_html=True
        )
    
        st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    with st.form(key=form_key, clear_on_submit=False):

        # Render itens da página
        for it in page_items:
            val = st.session_state[keys.answers_key][str(it.id)]
            _render_item_row(cfg, keys, it, data.answer_options, val)

        # Placeholder para aviso (fica ACIMA do botão)
        warn_box = st.empty()

        # Botão do form
        is_last = (current_page == total_pages)
        label = "Enviar escala" if is_last else "Próxima página"
        submitted = st.form_submit_button(label, use_container_width=True)

        # Validação e decisão — tudo DENTRO do form para manter o aviso "colado" ao botão
        if submitted:
            # Valida somente os itens da página atual
            missing = _validate_answers(cfg, data, st.session_state[keys.answers_key], scope_items=page_items)
            if missing:
                # Monta uma mensagem curta com os IDs faltantes
                ids = ", ".join(str(row["Item"]) for row in missing)
                warn_box.warning(f"Ainda faltam respostas nesta página: itens {ids}.")
            else:
                if is_last:
                    # Checagem global final (opcional, mas garante integridade)
                    missing_all = _validate_answers(cfg, data, st.session_state[keys.answers_key], scope_items=None)
                    if missing_all:
                        ids_all = ", ".join(str(row["Item"]) for row in missing_all)
                        warn_box.warning(f"Faltam respostas em outras páginas: itens {ids_all}.")
                    else:
                        finished = True
                else:
                    go_next = True

    return go_next, finished

