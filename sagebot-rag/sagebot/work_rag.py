import os, threading
import streamlit as st
from langchain_community.vectorstores import FAISS
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
from langchain.schema import Document

# importe só o que usa (evita colisão de nomes)
from .rag import *
from .progress import * 
from .utils import *
from openai import APIConnectionError, RateLimitError, AuthenticationError
from typing import List, Optional
from .rag import retriever as base_retriever


BASE_INDEX = "data/index"

def save_index(vs: FAISS, h):
    path = os.path.join(BASE_INDEX, h)
    os.makedirs(path, exist_ok=True)
    vs.save_local(path)

def load_index(h, emb):
    path = os.path.join(BASE_INDEX, h)
    if os.path.isdir(path):
        return FAISS.load_local(path, emb, allow_dangerous_deserialization=True)
    return None

def iniciar_async(documento, embed_model = "text-embedding-3-small", dims=None, k = 4):
    if st.session_state.get("index_status") == "building":
        return

    progress()
    st.session_state["index_status"] = "building"
    atualizar(step="init", pct=0.0, log="Iniciando indexação...")

    ctx = get_script_run_ctx()

    def job():
        try:
            atualizar(step="split", pct=0.10, log="Preparando splitter...")
            splitter = split_text(chunk_size=1500, chunk_overlap=100)
            chunks = splitter.split_text(documento)
            docs = [Document(page_content=c) for c in chunks]

            h = doc_hash(documento, model=embed_model, dims=dims)

            emb = embeddings(model=embed_model, dimensions=dims)

            cached = load_index(h, emb)
            if cached:
                st.session_state["vs"] = cached
                st.session_state["retriever"] = retriever(cached, k=k)
                st.session_state["index_status"] = "ready"
                atualizar(step="done", pct=1.0, log="Índice carregado do cache.")
                return

            atualizar(step="embed", pct=0.40, log="Gerando embeddings (OpenAI)...")
            vs = build_vectorstore(docs, embed_model=embed_model, dims=dims)

            atualizar(step="index", pct=0.80, log="Salvando índice no disco...")
            save_index(vs, h)

            st.session_state["vs"] = vs
            st.session_state["retriever"] = retriever(vs, k=k)
            st.session_state["index_status"] = "ready"
            atualizar(step="done", pct=1.0, log="Indexação concluída.")
        except AuthenticationError:
            st.session_state["index_status"] = "error"
            st.session_state["index_error"] = "OPENAI_API_KEY inválida ou ausente."
            atualizar(step="error", pct=0.0, log="OPENAI_API_KEY inválida ou ausente.")
        except RateLimitError:
            st.session_state["index_status"] = "error"
            st.session_state["index_error"] = "Rate limit excedido."
            atualizar(step="error", pct=0.0, log="Rate limit excedido.")
        except APIConnectionError:
            st.session_state["index_status"] = "error"
            st.session_state["index_error"] = "Problemas de conexão com a OpenAI."
            atualizar(step="error", pct=0.0, log="Problemas de conexão com a OpenAI.")
        except Exception as e:
            st.session_state["index_status"] = "error"
            st.session_state["index_error"] = str(e)
            atualizar(step="error", pct=0.0, log=f"Erro: {e}")

    t = threading.Thread(target=job, daemon=True)
    add_script_run_ctx(t, ctx=ctx)
    t.start()



def listar_indices_existentes() -> List[str]:
    path = BASE_INDEX
    if not os.path.isdir(path):
        return []
    dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    dirs.sort(key=lambda d: os.path.getmtime(os.path.join(path, d)), reverse=True)
    return dirs

def carregar_indice_existente(h: str, embed_model="text-embedding-3-small", dims=None, k=4, compressed=True):
    progress()
    st.session_state["index_status"] = "building"
    try:
        atualizar(step="load", pct=0.10, log=f"Carregando índice: {h}")
        emb = embeddings(model=embed_model, dimensions=dims)
        vs = load_index(h, emb)
        if vs is None:
            st.session_state["index_status"] = "error"
            st.session_state["index_error"] = f"Índice {h} não encontrado."
            atualizar(step="error", pct=0.0, log=f"Índice {h} não encontrado.")
            return

        st.session_state["vs"] = vs
        if compressed:
            st.session_state["retriever"] = compressed_retriever(vs, k=k)
        else:
            st.session_state["retriever"] = base_retriever(vs, k=k)
        st.session_state["index_status"] = "ready"
        atualizar(step="done", pct=1.0, log=f"Índice {h} carregado.")
    except Exception as e:
        st.session_state["index_status"] = "error"
        st.session_state["index_error"] = str(e)
        atualizar(step="error", pct=0.0, log=f"Erro ao carregar índice {h}: {e}")

def incrementar_indice(docs_novos: List[Document], embed_model="text-embedding-3-small", dims=None, k=4):
    
    from progress import atualizar
    vs = st.session_state.get("vs")
    if vs is None:
        raise RuntimeError("Nenhum índice carregado. Carregue um índice antes de incrementar.")

    atualizar(step="embed", pct=0.40, log="Gerando embeddings para novos documentos...")
    emb = embeddings(model=embed_model, dimensions=dims)
    vs.add_documents(docs_novos, embedding=emb)

    if isinstance(st.session_state.get("retriever"), ContextualCompressionRetriever):
        st.session_state["retriever"] = compressed_retriever(vs, k=k)
    else:
        st.session_state["retriever"] = base_retriever(vs, k=k)

    atualizar(step="index", pct=0.80, log="Atualizando índice em disco...")
   
    current_hash = st.session_state.get("current_index_hash")
    if current_hash:
        save_index(vs, current_hash)
    else:
        save_index(vs, "ad-hoc-updated")

    st.session_state["index_status"] = "ready"
    atualizar(step="done", pct=1.0, log="Índice atualizado com novos documentos.")
