# build_index_incremental.py — incremental + metadado de projeto (ID antes do primeiro "_")
import os, json, hashlib, re
from pathlib import Path
import numpy as np
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

from pypdf import PdfReader
from docx import Document
from bs4 import BeautifulSoup
import chardet
import markdown as md

OPENAI_EMBED_MODEL = "text-embedding-ada-002"
DATA_DIR  = Path("data")
INDEX_DIR = Path("index"); INDEX_DIR.mkdir(exist_ok=True)

VEC_PATH   = INDEX_DIR / "vectors_normed.npz"
META_PATH  = INDEX_DIR / "meta.jsonl"
MANIFEST   = INDEX_DIR / "manifest.json"

client = OpenAI()

# ---------- utils ----------
def fhash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()

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
    if ext in (".html",".htm"): return read_html(p)
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

def infer_project(path: Path) -> str:
    """
    Extrai o ID do projeto a partir do nome do arquivo:
      - Regra principal: tudo que vem ANTES do primeiro '_' é o ID.
        Ex.: '202100185-7_relatorio.pdf' -> '202100185-7'
      - Validação simples: começa com dígito; aceita hífens.
      - Fallback 1: usa a primeira subpasta em data/<projeto>/arquivo
      - Fallback 2: '__default__'
    """
    name = path.name
    if "_" in name:
        proj = name.split("_", 1)[0].strip()
        if re.match(r"^\d[\d-]*$", proj):
            return proj

    rel = path.relative_to(DATA_DIR)
    if len(rel.parts) > 1:
        return rel.parts[0]

    return "__default__"

# ---------- load existing ----------
def load_existing():
    if VEC_PATH.exists():
        data = np.load(VEC_PATH)
        mat = data["mat"]
    else:
        mat = np.empty((0, 1536), dtype=np.float32)

    metas = []
    if META_PATH.exists():
        with open(META_PATH, "r", encoding="utf-8") as fr:
            metas = [json.loads(l) for l in fr]

    manifest = {}
    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    return mat, metas, manifest

# ---------- save all ----------
def save_all(mat, metas, manifest):
    # normaliza (cosine) e salva
    if mat.size:
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        mat = mat / np.clip(norms, 1e-12, None)

    np.savez_compressed(VEC_PATH, mat=mat)

    with open(META_PATH, "w", encoding="utf-8") as fw:
        for m in metas:
            fw.write(json.dumps(m, ensure_ascii=False) + "\n")

    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

# ---------- main ----------
def main():
    mat, metas, manifest = load_existing()
    print(f"Índice atual: {mat.shape[0]} vetores")

    files = [p for p in DATA_DIR.rglob("*")
             if p.is_file() and p.suffix.lower() in {".pdf",".docx",".txt",".md",".html",".htm"}]
    if not files:
        print("Sem arquivos em ./data")
        return

    appended_vecs, appended_meta = [], []

    for f in files:
        h = fhash(f)
        key = str(f)
        prev = manifest.get(key)
        if prev and prev.get("sha256") == h:
            # nada mudou → pula
            continue

        try:
            text = load_text(f)
            chunks = chunk_text(text)
            if not chunks:
                continue
            project = infer_project(f)
            B = 64
            local_count = 0

            for i in range(0, len(chunks), B):
                b = chunks[i:i+B]
                embs = embed_batch(b)
                for j, e in enumerate(embs):
                    appended_vecs.append(e.astype(np.float32))
                    appended_meta.append({
                        "file": key,
                        "project": project,
                        "chunk_id": i+j,
                        "text": b[j][:1000]
                    })
                    local_count += 1

            manifest[key] = {
                "sha256": h,
                "project": project,
                "count": local_count
            }
            print(f"OK: [{project}] {f} -> {local_count} chunks (atualizado)")
        except Exception as e:
            print(f"ERRO em {f}: {e}")

    if appended_vecs:
        new_mat = np.vstack([mat] + appended_vecs) if mat.size else np.vstack(appended_vecs)
        new_metas = metas + appended_meta
        save_all(new_mat, new_metas, manifest)
        print(f"Índice atualizado: {new_mat.shape[0]} vetores")
    else:
        print("Nada para atualizar (sem arquivos novos/alterados).")

if __name__ == "__main__":
    main()
