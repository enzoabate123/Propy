import argparse, json, re, unicodedata
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# ====== Config ======
EMBED_MODEL = "text-embedding-ada-002"
THRESH = 0.82  # limiar para "aprovado"
load_dotenv()
client = OpenAI()

# ====== Normalização e métricas lexicais ======
def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

def normalize(s: str) -> str:
    s = strip_accents(s.lower().strip())
    s = re.sub(r"\s+", " ", s)
    return s

def difflib_ratio(a: str, b: str) -> float:
    import difflib
    return float(difflib.SequenceMatcher(None, a, b).ratio())

def jaccard_unigram(a: str, b: str) -> float:
    A, B = set(a.split()), set(b.split())
    if not A and not B: return 1.0
    if not A or not B:  return 0.0
    return len(A & B) / len(A | B)

def rouge_l(a: str, b: str) -> float:
    A, B = a.split(), b.split()
    m, n = len(A), len(B)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m):
        for j in range(n):
            dp[i+1][j+1] = dp[i][j]+1 if A[i]==B[j] else max(dp[i][j+1], dp[i+1][j])
    lcs = dp[m][n]
    denom = max(m, n) or 1
    return lcs/denom

# ====== Embeddings / semântica ======
def embed_texts(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 1536), dtype=np.float32)
    res = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vecs = [np.array(d.embedding, dtype=np.float32) for d in res.data]
    M = np.vstack(vecs)
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    M = M / np.clip(norms, 1e-12, None)
    return M

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))

# ====== Avaliadores por tipo ======
def score_strings(pred: str, gold: str) -> Dict[str, float]:
    # semântico
    E = embed_texts([pred, gold])
    sem = cosine(E[0], E[1])

    # lexical
    a, g = normalize(pred), normalize(gold)
    d = difflib_ratio(a, g)
    j = jaccard_unigram(a, g)
    r = rouge_l(a, g)

    combined = 0.8*sem + 0.2*((d + r)/2)
    return {
        "cosine_semantic": round(sem, 4),
        "difflib": round(d, 4),
        "jaccard": round(j, 4),
        "rougeL": round(r, 4),
        "combined": round(combined, 4)
    }

def score_list_of_strings(pred_list: List[str], gold_list: List[str]) -> Dict[str, Any]:
    # limpa vazios
    P = [p.strip() for p in pred_list if str(p).strip()]
    G = [g.strip() for g in gold_list if str(g).strip()]
    if not G and not P:
        return {"coverage": 1.0, "avg_similarity": 1.0, "pairs": [], "combined": 1.0}
    if not G or not P:
        return {"coverage": 0.0, "avg_similarity": 0.0, "pairs": [], "combined": 0.0}

    # embeddings
    E = embed_texts(P + G)
    EP, EG = E[:len(P)], E[len(P):]

    # matriz de similaridade P x G
    sim = EP @ EG.T  # cosines

    # matching guloso: para cada G, pega o melhor P disponível
    used_p = set()
    pairs: List[Tuple[int,int,float]] = []
    for j in range(len(G)):
        i_best = int(np.argmax(sim[:, j]))
        score = float(sim[i_best, j])
        if i_best not in used_p:
            used_p.add(i_best)
            pairs.append((i_best, j, score))

    coverage = len(pairs) / len(G)
    avg_sim  = float(np.mean([s for _,_,s in pairs])) if pairs else 0.0
    combined = 0.8*avg_sim + 0.2*coverage
    return {
        "coverage": round(coverage, 4),
        "avg_similarity": round(avg_sim, 4),
        "pairs": [
            {"pred_index": i, "gold_index": j, "similarity": round(s, 4), "pred": P[i], "gold": G[j]}
            for i, j, s in pairs
        ],
        "combined": round(combined, 4)
    }

# ====== Comparação recursiva ======
def compare_nodes(pred: Any, gold: Any, path: str="") -> Dict[str, Any]:
    node = {"path": path, "type": None, "score": None, "approved": None, "detail": None}

    # tipagens
    if isinstance(gold, dict):
        node["type"] = "dict"
        detail = []
        keys = set(gold.keys()) | (set(pred.keys()) if isinstance(pred, dict) else set())
        for k in sorted(keys):
            sub_pred = pred.get(k) if isinstance(pred, dict) else None
            sub_gold = gold.get(k)
            detail.append(compare_nodes(sub_pred, sub_gold, f"{path}.{k}" if path else k))
        # média simples dos filhos (ignora None)
        scores = [d.get("score") for d in detail if isinstance(d.get("score"), (int, float))]
        avg = round(float(np.mean(scores)),4) if scores else None
        node["score"] = avg
        node["approved"] = (avg is not None and avg >= THRESH)
        node["detail"] = detail
        return node

    if isinstance(gold, list):
        node["type"] = "list"
        # lista de strings?
        if all(isinstance(x, str) for x in gold) and isinstance(pred, list) and all(isinstance(x, str) for x in pred):
            res = score_list_of_strings(pred, gold)
            node["score"] = res["combined"]
            node["approved"] = node["score"] >= THRESH
            node["detail"] = res
        else:
            # fallback: compara por posição, média dos filhos
            detail, scores = [], []
            for i, g in enumerate(gold):
                p = (pred[i] if isinstance(pred, list) and i < len(pred) else None)
                child = compare_nodes(p, g, f"{path}[{i}]")
                detail.append(child)
                if isinstance(child.get("score"), (int, float)):
                    scores.append(child["score"])
            avg = round(float(np.mean(scores)),4) if scores else 0.0
            node["score"] = avg
            node["approved"] = avg >= THRESH
            node["detail"] = detail
        return node

    # strings
    if isinstance(gold, str):
        node["type"] = "string"
        if isinstance(pred, str) and pred.strip():
            sc = score_strings(pred, gold)
            node["score"] = sc["combined"]
            node["approved"] = node["score"] >= THRESH
            node["detail"] = sc
        else:
            node["score"] = 0.0
            node["approved"] = False
            node["detail"] = {"missing_pred": True}
        return node

    # outros tipos (número, bool, None) -> igualdade exata
    eq = (pred == gold)
    node["type"] = type(gold).__name__
    node["score"] = 1.0 if eq else 0.0
    node["approved"] = eq
    node["detail"] = {"equal": eq}
    return node

# ====== CLI ======
def main():
    ap = argparse.ArgumentParser(description="Compara respostas_NI.json com gabarito (semântica + lexical).")
    ap.add_argument("--gold", required=True, help="caminho do gabarito (JSON)")
    ap.add_argument("--pred", required=True, help="caminho do respostas_NI.json")
    ap.add_argument("--out", default="relatorio_avaliacao.json", help="relatório de saída (JSON)")
    ap.add_argument("--thresh", type=float, default=THRESH, help="limiar de aprovação (default: 0.82)")
    args = ap.parse_args()

    global THRESH
    THRESH = args.thresh

    gold = json.loads(Path(args.gold).read_text(encoding="utf-8"))
    pred = json.loads(Path(args.pred).read_text(encoding="utf-8"))

    report = compare_nodes(pred, gold)

    # resumo
    def collect_leaves(n):
        if "detail" in n and isinstance(n["detail"], list):
            out = []
            for d in n["detail"]:
                out.extend(collect_leaves(d))
            return out
        return [n]

    leaves = collect_leaves(report)
    valid = [x for x in leaves if isinstance(x.get("score"), (int,float))]
    avg_all = round(float(np.mean([x["score"] for x in valid])),4) if valid else 0.0
    approved = avg_all >= THRESH

    summary = {
        "avg_score_all_fields": avg_all,
        "approved_global": approved,
        "threshold": THRESH
    }
    out = {"summary": summary, "report": report}

    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ Avaliação salva em {args.out}")
    print(f"→ Média global: {avg_all} | THRESH={THRESH} | APROVADO={approved}")

if __name__ == "__main__":
    main()
