from pathlib import Path
from langchain_community.document_loaders import TextLoader, WebBaseLoader,PyPDFLoader
import tempfile
import os

DEFAULT_UA = os.environ.get("USER_AGENT", "SageBot/1.0 (Streamlit)")

def carrega_md(arquivo):
    uploads = arquivo if isinstance(arquivo, list) else [arquivo]
    textos = []
    for up in uploads:
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as tmp:
            tmp.write(up.read())
            caminho = tmp.name
        docs = TextLoader(caminho, encoding="utf-8").load()
        textos.append("\n\n".join([d.page_content for d in docs]))
    return "\n\n".join(textos)

def carrega_site(url):
    loader = WebBaseLoader(url)
    lista_documentos = loader.load()
    documento = '\n\n'.join([doc.page_content for doc in lista_documentos])
    return documento


def carrega_pdf(caminho):
    loader = PyPDFLoader(caminho)
    lista_documentos = loader.load()
    documento = '\n\n'.join([doc.page_content for doc in lista_documentos])
    return documento

def salvar_tmp(upload, suffix):
    paths = []
    if upload is None:
        return paths
    uploads = upload if isinstance(upload, list) else [upload]
    for up in uploads:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(up.read())
            paths.append(tmp.name)
    return paths
