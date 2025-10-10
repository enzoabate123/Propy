# rag_query.py — RAG com filtro por projeto e k ajustado automaticamente
# k_eff = max(1, min(k, n_total_do_subset))

import json
from pathlib import Path
import numpy as np
from sklearn.neighbors import NearestNeighbors
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # carrega OPENAI_API_KEY do .env

OPENAI_EMBED_MODEL = "text-embedding-ada-002"
CHAT_MODEL = "o4-mini"   # troque se quiser
INDEX_DIR = Path("index")

client = OpenAI()  # usa a chave do ambiente

def _ensure_index_files():
    data_path = INDEX_DIR / "vectors_normed.npz"
    meta_path = INDEX_DIR / "meta.jsonl"
    if not data_path.exists() or not meta_path.exists():
        raise RuntimeError("Índice não encontrado. Rode `python build_index_incremental.py` (ou build_index.py) primeiro.")
    return data_path, meta_path

def load_raw():
    data_path, meta_path = _ensure_index_files()
    data = np.load(data_path)
    mat = data["mat"]  # (n_vectors x 1536)
    metas = [json.loads(l) for l in open(meta_path, encoding="utf-8")]
    return mat, metas

def subset_by_project(mat: np.ndarray, metas: list[dict], project: str | None):
    """Filtra o índice pelo projeto (metadado 'project'). Se None, retorna tudo."""
    if not project:
        return mat, metas
    idxs = [i for i, m in enumerate(metas) if m.get("project") == project]
    if not idxs:
        # subset vazio
        return np.empty((0, mat.shape[1]), dtype=mat.dtype), []
    return mat[idxs, :], [metas[i] for i in idxs]

def embed_query(q: str) -> np.ndarray:
    r = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=[q])
    v = np.array(r.data[0].embedding, dtype=np.float32)
    v = v / max(np.linalg.norm(v), 1e-12)  # normaliza (cosine)
    return v.reshape(1, -1)

def search(q: str, k: int = 5, project: str | None = None):
    """Busca top-k no subset do projeto (ou global se None). k é limitado por n_total."""
    mat, metas = load_raw()
    sub_mat, sub_meta = subset_by_project(mat, metas, project)
    n_total = sub_mat.shape[0]
    if n_total == 0:
        return []

    # Para treinar o NN, n_neighbors não pode exceder n_total (cap em 5 para performance)
    n_for_fit = max(1, min(5, n_total))
    nn = NearestNeighbors(n_neighbors=n_for_fit, metric="cosine").fit(sub_mat)

    # >>> k efetivo limitado ao total de vetores do subset
    k_eff = max(1, min(k, n_total))

    qv = embed_query(q)
    dists, idxs = nn.kneighbors(qv, n_neighbors=k_eff)

    results = []
    for dist, i_local in zip(dists[0], idxs[0]):
        results.append({
            "score": float(1.0 - dist),  # 1 - distância (cosine) ~ similaridade
            "meta": sub_meta[int(i_local)]
        })
    return results

def answer(q: str, k: int = 5, project: str | None = None):
    hits = search(q, k=k, project=project)
    if not hits:
        alvo = f" para o projeto '{project}'" if project else ""
        return f"Não encontrei contextos no índice{alvo}.", []

    context = "\n\n".join(
        [f"- [{h['meta'].get('project','?')}] {h['meta']['file']} (chunk {h['meta']['chunk_id']}) {h['meta']['text']}"
         for h in hits]
    )

    prompt = (
        "Responda com base no contexto. Se a informação não estiver presente, diga isso claramente.\n\n"
        f"Contexto:\n{context}\n\nPergunta: {q}\nResposta:"
    )

    # Chamada minimalista para compatibilidade ampla de modelos:
    r = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "Você é objetivo e cita a origem (projeto/arquivo/chunk) quando útil."},
            {"role": "user", "content": prompt},
        ],
    )
    return r.choices[0].message.content, hits

if __name__ == "__main__":
    pergunta = input("Pergunte: ")
    proj = input("Projeto (ENTER = todos): ").strip() or None
    resp, refs = answer(pergunta, k=3, project=proj)  # k padrão menor; ainda assim limitado por n_total
    print("\n=== RESPOSTA ===\n", resp)
    print("\n=== FONTES ===")
    for r in refs:
        print(f"[{r['score']:.3f}] [{r['meta'].get('project','?')}] {r['meta']['file']} (chunk {r['meta']['chunk_id']})")
