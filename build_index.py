# build_index.py — indexador sem FAISS (usa scikit-learn na consulta)
import json
from pathlib import Path

import numpy as np
from openai import OpenAI

# ===== .env =====
from dotenv import load_dotenv
load_dotenv()  # carrega OPENAI_API_KEY do arquivo .env

# Leitura de arquivos
from pypdf import PdfReader
from docx import Document
from bs4 import BeautifulSoup
import chardet
import markdown as md

OPENAI_EMBED_MODEL = "text-embedding-ada-002"
DATA_DIR = Path("data")
OUT_DIR = Path("index"); OUT_DIR.mkdir(exist_ok=True)

client = OpenAI()  # pega a chave do ambiente (.env)

def read_txt_like(p: Path) -> str:
    raw = p.read_bytes()
    enc = chardet.detect(raw).get("encoding") or "utf-8"
    return raw.decode(enc, errors="ignore")

def read_pdf(p: Path) -> str:
    r = PdfReader(str(p))
    return "\n".join([(pg.extract_text() or "") for pg in r.pages])

def read_docx(p: Path) -> str:
    return "\n".join([para.text for para in Document(str(p)).paragraphs])

def read_html(p: Path) -> str:
    soup = BeautifulSoup(read_txt_like(p), "html.parser")
    return soup.get_text(separator=" ")

def read_md(p: Path) -> str:
    html = md.markdown(read_txt_like(p))
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator=" ")

def load_text(p: Path) -> str:
    ext = p.suffix.lower()
    if ext == ".pdf": return read_pdf(p)
    if ext == ".docx": return read_docx(p)
    if ext in (".html", ".htm"): return read_html(p)
    if ext == ".md": return read_md(p)
    return read_txt_like(p)

def chunk_text(text: str, chunk_size=900, overlap=150):
    if not text: return []
    chunks, i, step = [], 0, max(1, chunk_size - overlap)
    while i < len(text):
        c = text[i:i+chunk_size].strip()
        if c: chunks.append(c)
        i += step
    return chunks

def embed_batch(texts):
    resp = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=texts)
    return [np.array(d.embedding, dtype=np.float32) for d in resp.data]

def main():
    files = [p for p in DATA_DIR.rglob("*")
             if p.is_file() and p.suffix.lower() in {".pdf",".docx",".txt",".md",".html",".htm"}]
    if not files:
        print("Coloque arquivos dentro de ./data e rode novamente.")
        return

    vectors, metas = [], []
    for f in files:
        try:
            text = load_text(f)
            chunks = chunk_text(text)
            if not chunks: 
                print(f"(vazio) {f}")
                continue
            B = 64
            for i in range(0, len(chunks), B):
                b = chunks[i:i+B]
                embs = embed_batch(b)
                for j, e in enumerate(embs):
                    vectors.append(e)
                    metas.append({"file": str(f), "chunk_id": i+j, "text": b[j][:1000]})
            print(f"OK: {f} -> {len(chunks)} chunks")
        except Exception as e:
            print(f"ERRO em {f}: {e}")

    if not vectors:
        print("Nenhum vetor gerado.")
        return

    mat = np.vstack(vectors).astype(np.float32)
    # normaliza para similaridade do cosseno
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat = mat / np.clip(norms, 1e-12, None)

    np.savez_compressed(OUT_DIR / "vectors_normed.npz", mat=mat)
    with open(OUT_DIR / "meta.jsonl", "w", encoding="utf-8") as fw:
        for m in metas:
            fw.write(json.dumps(m, ensure_ascii=False) + "\n")

    print(f"Vetores salvos em index/vectors_normed.npz ({mat.shape[0]} x {mat.shape[1]})")
    print("Metadados salvos em index/meta.jsonl")

if __name__ == "__main__":
    main()
