# rag.py
from typing import List, Optional, Sequence, Union
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter

def split_text(chunk_size = 1500, chunk_overlap = 100):
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "? ", " ", ""],
    )

def to_documents(chunks: Sequence[Union[str, Document]]) -> List[Document]:

    docs: List[Document] = []
    for c in chunks:
        if isinstance(c, Document):
            docs.append(c)
        else:
            text = (c or "").strip()
            if text:
                docs.append(Document(page_content=text))
    return docs

def embeddings(model= "text-embedding-3-small", dimensions= None) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=model,dimensions=dimensions)


def build_vectorstore(
    docs: Sequence[Union[str, Document]],
    embed_model = "text-embedding-3-small",
    dims = None,
) -> FAISS:
    emb = embeddings(model=embed_model, dimensions=dims)
    docs_ = to_documents(docs)
    
    step = 128
    vs: Optional[FAISS] = None 
    
    for i in range(0, len(docs_), step):
        batch = docs_[i:i+step]
        if vs is None:
            vs = FAISS.from_documents(batch, emb)
        else:
            vs.add_documents(batch, embedding=emb)

    if vs is None:
        vs = FAISS.from_documents([Document(page_content="")], emb)

    return vs

def retriever(vs: FAISS, k= 4):
    return vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": 30, "lambda_mult": 0.5, "score_threshold": 0.3}
    )

def compressed_retriever(vs: FAISS, k: int = 4, similarity_threshold: float = 0.35):
    base = retriever(vs, k=k)
    comp = EmbeddingsFilter(embeddings=embeddings(), similarity_threshold=similarity_threshold)
    return ContextualCompressionRetriever(base_compressor=comp, base_retriever=base)
