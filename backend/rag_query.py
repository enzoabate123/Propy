# rag_query.py — busca top-k com NearestNeighbors + geração via Chat
# k ajustado automaticamente: k_eff = min(k, total_de_vetores)

import json
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # carrega OPENAI_API_KEY do .env

OPENAI_EMBED_MODEL = "text-embedding-ada-002"
CHAT_MODEL = "o4-mini"        # troque se quiser
INDEX_DIR = Path("index")

client = OpenAI()  # usa a chave do ambiente

def load_index():
    data_path = INDEX_DIR / "vectors_normed.npz"
    meta_path = INDEX_DIR / "meta.jsonl"
    if not data_path.exists() or not meta_path.exists():
        raise RuntimeError("Índice não encontrado. Rode `python build_index.py` primeiro.")

    data = np.load(data_path)
    mat = data["mat"]                      # (n_vectors x 1536)
    metas = [json.loads(l) for l in open(meta_path, encoding="utf-8")]
    n_total = mat.shape[0]

    # Para treinar o NN, n_neighbors não pode exceder n_total
    n_for_fit = max(1, min(5, n_total))
    nn = NearestNeighbors(n_neighbors=n_for_fit, metric="cosine").fit(mat)
    return mat, metas, nn, n_total

def embed_query(q: str):
    r = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=[q])
    v = np.array(r.data[0].embedding, dtype=np.float32)
    v = v / max(np.linalg.norm(v), 1e-12)  # normaliza (cosine)
    return v.reshape(1, -1)

def search(q: str, k=5):
    _, metas, nn, n_total = load_index()
    if n_total == 0:
        return []

    # >>> AQUI: k limitado pelo total de vetores <<<
    k_eff = max(1, min(k, n_total))

    qv = embed_query(q)
    dists, idxs = nn.kneighbors(qv, n_neighbors=k_eff)

    results = []
    for dist, idx in zip(dists[0], idxs[0]):
        results.append({
            "score": float(1.0 - dist),     # converte distância-cosseno em similaridade
            "meta": metas[int(idx)]
        })
    return results

def answer(q: str, k=5):
    hits = search(q, k)
    if not hits:
        return "Não encontrei contextos no índice. Adicione arquivos em ./data e reindexe.", []

    context = "\n\n".join(
        [f"- ({h['meta']['file']} | chunk {h['meta']['chunk_id']}) {h['meta']['text']}" for h in hits]
    )

    prompt = (
        "Responda com base no contexto. Se a informação não estiver presente, diga isso claramente.\n\n"
        f"Contexto:\n{context}\n\nPergunta: {q}\nResposta:"
    )

    # Chamada minimalista (compatível com qualquer modelo do Chat Completions):
    r = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "Você é objetivo e cita a origem quando útil."},
            {"role": "user", "content": prompt},
        ],
    )
    return r.choices[0].message.content, hits

if __name__ == "__main__":
    q = input("Pergunte: ")
    resp, refs = answer(q, k=3)  # k padrão menor; ainda assim limitado por n_total
    print("\n=== RESPOSTA ===\n", resp)
    print("\n=== FONTES ===")
    for r in refs:
        print(f"[{r['score']:.3f}] {r['meta']['file']} (chunk {r['meta']['chunk_id']})")
