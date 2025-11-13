import streamlit as st
from typing import Optional

KEY_STATUS = "index_status"   # "idle" | "building" | "ready" | "error" | "error: ..."
KEY_STEP   = "index_step"     # "init" | "split" | "embed" | "index" | "ready"
KEY_PCT    = "index_pct"      # float 0..1
KEY_LOG    = "index_log"      # list[str]
KEY_RERUN  = "__needs_rerun"  # bool
KEY_ERRMSG = "index_error"    # str (detalhe do erro, quando existir)

def progress():
    """Reseta o estado de progresso antes de iniciar a indexação."""
    st.session_state[KEY_STATUS] = "building"
    st.session_state[KEY_STEP]   = "init"
    st.session_state[KEY_PCT]    = 0.0
    st.session_state[KEY_LOG]    = []
    st.session_state.pop(KEY_ERRMSG, None)

def adicionar_log(msg: str):
    
    buf = st.session_state.get(KEY_LOG, [])
    buf.append(msg)
    st.session_state[KEY_LOG] = buf[-200:]

def atualizar(step: Optional[str] = None, pct: Optional[float] = None, log: Optional[str] = None):
   
    if step is not None:
        st.session_state[KEY_STEP] = step
    if pct is not None:
      
        p = float(pct)
        if p > 1.0:
            p = p / 100.0
        p = max(0.0, min(1.0, p))
        st.session_state[KEY_PCT]  = p
    if log:
        adicionar_log(log)

def mark_ready():
    
    st.session_state[KEY_STATUS] = "ready"
    st.session_state[KEY_STEP]   = "ready"
    st.session_state[KEY_PCT]    = 1.0
    st.session_state[KEY_RERUN]  = True
    adicionar_log("Índice RAG pronto ok")

def mark_error(err: Exception):
    msg = str(err)
    st.session_state[KEY_STATUS]  = "error"
    st.session_state[KEY_ERRMSG]  = msg
    adicionar_log(f"Erro: {msg}")

def is_ready() -> bool:
    return st.session_state.get(KEY_STATUS) == "ready"

def is_building() -> bool:
    return st.session_state.get(KEY_STATUS) == "building"

def mark_idle():
  
    st.session_state[KEY_STATUS]  = "idle"
    st.session_state[KEY_STEP]    = None
    st.session_state[KEY_PCT]     = 0.0
    st.session_state[KEY_LOG]     = []
    st.session_state.pop(KEY_ERRMSG, None)
    st.session_state.pop(KEY_RERUN, None)

def render_status():
    status = st.session_state.get(KEY_STATUS, "idle")
    step   = st.session_state.get(KEY_STEP, None)
    pct    = st.session_state.get(KEY_PCT, 0.0)
    logs   = st.session_state.get(KEY_LOG, [])
    errmsg = st.session_state.get(KEY_ERRMSG, None)

    pct_int = int(round((pct or 0.0) * 100))

    if status == "building":
        try:
            with st.status("Indexando documento para RAG…", expanded=True):
                st.progress(pct)
                st.write(f"Progresso: **{pct_int}%**")
                if step:
                    st.write(f"Etapa: **{step}**")
                for line in logs[-6:]:
                    st.write(f"- {line}")
        except Exception:
            st.info(f"Indexando… etapa: {step or '...'}")
            st.progress(pct)
            st.write(f"{pct_int}%")
    elif status == "ready":
        st.success("Índice RAG pronto ok.")
    elif (isinstance(status, str) and status.startswith("error")) or status == "error":
        detalhe = errmsg or (status if ":" in status else "Erro na indexação.")
        st.error(detalhe)
        if logs:
            with st.expander("Detalhes (logs)"):
                for line in logs[-20:]:
                    st.write(f"- {line}")
    else:
        st.info("RAG aguardando inicialização.")
