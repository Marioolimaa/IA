import hashlib,json
from langchain.schema import Document
from typing import Iterable


def doc_hash(content, model= "", dims=None):
    cfg = {
        "model": model or "",
        "dims": dims if dims is not None else "native"
    }
    payload = json.dumps(
        {"content": content, "cfg": cfg},
        ensure_ascii=False,
        sort_keys=True
    ).encode("utf-8")

    return hashlib.sha1(payload).hexdigest()

def corpus_hash(docs: Iterable[Document], model= "", dims=None) -> str:

    cfg = {"model": model or "", "dims": dims if dims is not None else "native"}
    items = []
    for d in docs:
        items.append({
            "text": (d.page_content or ""),
            "src": d.metadata.get("source"),
            "page": d.metadata.get("page"),
        })
    payload = json.dumps({"docs": items, "cfg": cfg}, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()