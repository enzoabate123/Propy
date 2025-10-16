# fill_ni.py — preenche perguntas_NI.json usando seu RAG (índice + GPT)
import json
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
from sklearn.neighbors import NearestNeighbors
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

OPENAI_EMBED_MODEL = "text-embedding-ada-002"
CHAT_MODEL = "o4-mini"   # troque se quiser
INDEX_DIR = Path("index")

client = OpenAI()

# ---------- carga do índice (1x na memória) ----------
def _ensure_index():
    vec = INDEX_DIR / "vectors_normed.npz"
    meta = INDEX_DIR / "meta.jsonl"
    if not vec.exists() or not meta.exists():
        raise RuntimeError("Índice não encontrado. Rode seu build do índice primeiro.")
    return vec, meta

_VEC, _META = _ensure_index()
_MAT = np.load(_Vec := _VEC)["mat"] if (_Vec := _VEC) else None  # noqa
_METAS = [json.loads(l) for l in open(_META, encoding="utf-8")]

# mapa de linhas por projeto
_PROJECT_ROWS: Dict[str, List[int]] = {}
for i, m in enumerate(_METAS):
    proj = m.get("project") or "__default__"
    _PROJECT_ROWS.setdefault(proj, []).append(i)

def _subset_by_project(project: Optional[str]):
    if not project:
        return _MAT, _METAS
    rows = _PROJECT_ROWS.get(project, [])
    if not rows:
        return np.empty((0, _MAT.shape[1]), dtype=_MAT.dtype), []
    return _MAT[rows, :], [_METAS[i] for i in rows]

def _embed_query(q: str) -> np.ndarray:
    r = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=[q])
    v = np.array(r.data[0].embedding, dtype=np.float32)
    v = v / max(np.linalg.norm(v), 1e-12)
    return v.reshape(1, -1)

def _search(q: str, k: int = 5, project: Optional[str] = None):
    sub_mat, sub_meta = _subset_by_project(project)
    n_total = sub_mat.shape[0]
    if n_total == 0:
        return []

    n_for_fit = max(1, min(5, n_total))
    nn = NearestNeighbors(n_neighbors=n_for_fit, metric="cosine").fit(sub_mat)

    k_eff = max(1, min(k, n_total))
    qv = _embed_query(q)
    dists, idxs = nn.kneighbors(qv, n_neighbors=k_eff)

    out = []
    for dist, i_local in zip(dists[0], idxs[0]):
        out.append({
            "score": float(1.0 - dist),
            "meta": sub_meta[int(i_local)]
        })
    return out

def _answer(q: str, k: int = 5, project: Optional[str] = None):
    hits = _search(q, k=k, project=project)
    if not hits:
        return None, []

    context = "\n\n".join(
        [f"- [{h['meta'].get('project','?')}] {h['meta']['file']} (chunk {h['meta']['chunk_id']}) {h['meta']['text']}"
         for h in hits]
    )

    prompt = (
        "Você é um assistente de P&D. Responda com base EXCLUSIVA no contexto. "
        "Se não houver informação suficiente, responda 'NÃO ENCONTRADO NO CONTEXTO'.\n\n"
        f"Contexto:\n{context}\n\nPergunta: {q}\nResposta objetiva:"
    )

    r = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "Mantenha respostas factuais e curtas; não invente."},
            {"role": "user", "content": prompt},
        ],
    )
    return r.choices[0].message.content.strip(), hits

# ---------- mapeamento de perguntas por campo ----------
# path do campo -> instrução específica
FIELD_QUESTIONS: Dict[str, str] = {
    "Descricao_da_Invencao.Titulo_da_Invencao": "Qual é o título técnico e conciso da invenção?",
    "Descricao_da_Invencao.Palavras_Chave": "Liste 3 a 8 palavras-chave técnicas da invenção (apenas termos, separados por vírgula).",
    "Descricao_da_Invencao.Problema_Tecnico": "Qual problema técnico a invenção resolve?",
    "Descricao_da_Invencao.Abordagem_Antes_da_Invencao": "Como o problema era tratado antes da invenção?",
    "Descricao_da_Invencao.Exemplos_ou_Resultados": "Há resultados, protótipos ou casos de teste? Resuma objetivamente.",
    "Descricao_da_Invencao.Como_a_Invencao_Soluciona": "Explique como a invenção resolve ou minimiza o problema, de forma objetiva.",
    "Descricao_da_Invencao.Aplicacao_da_Invencao": "Onde e como a invenção pode ser aplicada (produto, processo, sistema)?",
    "Descricao_da_Invencao.Vantagens_Esperadas": "Liste 3 a 6 vantagens objetivas da invenção (bullets curtos).",
    "Descricao_da_Invencao.Anexos": "Liste anexos disponíveis (ex.: desenho técnico, fluxograma, fotos).",
    "Descricao_da_Invencao.Componentes_da_Invencao": "Liste os principais componentes/partes da invenção.",
    "Descricao_da_Invencao.Interligacao_Entre_Componentes": "Como os componentes se conectam ou interagem?",
    "Descricao_da_Invencao.Estagio_de_Desenvolvimento": "Qual o estágio de desenvolvimento? (Ideia/Protótipo/Teste Piloto/Aplicação Industrial).",

    "Dados_Gerais.Gerencia_de_Origem": "Qual a gerência/unidade de origem?",
    "Dados_Gerais.Origem_da_Invencao": "A invenção surgiu de projeto, serviço técnico ou trabalho acadêmico?",
    "Dados_Gerais.Tipo_de_Invencao": "Classifique como Produto, Processo, Software, Equipamento ou Aperfeiçoamento.",
    "Dados_Gerais.Relaciona_a_NI_Anterior": "Há relação com NI anterior? Se sim, informe número/nome.",
    "Dados_Gerais.Area_de_Aplicacao": "Cite áreas de aplicação (ex.: Engenharia de Corrosão, Processos Químicos).",
    "Dados_Gerais.Pais_de_Deposito": "Qual o país de depósito recomendado (Brasil/EUA/Outro)?",
    "Dados_Gerais.Titularidade": "Qual a titularidade prevista? (Petrobras/Universidade Parceira/Outro).",

    "Dados_Gerais.Utilizacao_na_Petrobras.Esta_Sendo_Utilizada": "A invenção já está sendo utilizada? (Sim/Não).",
    "Dados_Gerais.Utilizacao_na_Petrobras.Descricao": "Onde e como está sendo utilizada (se aplicável)?",

    "Dados_Gerais.Divulgacao.Ja_Foi_Divulgada": "Já houve divulgação prévia? (Sim/Não).",
    "Dados_Gerais.Divulgacao.Detalhes": "Detalhes da divulgação (evento, artigo, congresso, etc.).",

    "Dados_Gerais.Comercializacao.Prevista": "Há previsão de comercialização? (Sim/Não).",
    "Dados_Gerais.Comercializacao.Empresa_Envolvida": "Qual empresa envolvida e estágio da negociação (se houver)?",

    "Dados_Gerais.Urgencia.Existe": "Existe urgência? (Sim/Não).",
    "Dados_Gerais.Urgencia.Motivo": "Explique a urgência (ex.: divulgação prevista em X dias).",

    "Observacoes_Finais": "Observações finais relevantes (confidencialidade, alinhamento P&D, etc.).",
}

# campos que devem virar lista (vamos dividir por linha/virgula)
LIST_FIELDS = {
    "Descricao_da_Invencao.Palavras_Chave",
    "Descricao_da_Invencao.Vantagens_Esperadas",
    "Descricao_da_Invencao.Anexos",
    "Descricao_da_Invencao.Componentes_da_Invencao",
}

def _to_list(text: str) -> List[str]:
    # aceita bullets, quebras de linha ou vírgulas
    raw = [t.strip(" -•\t") for t in text.replace(";", ",").splitlines() if t.strip()]
    if len(raw) <= 1:
        raw = [t.strip() for t in text.split(",") if t.strip()]
    # remove duplicatas preservando ordem
    seen, out = set(), []
    for x in raw:
        if x not in seen:
            out.append(x); seen.add(x)
    return out

def _fill_field(path: str, k: int, project: Optional[str]) -> Any:
    question = FIELD_QUESTIONS.get(path, f"Forneça conteúdo para o campo '{path}'.")
    ans, _ = _answer(question, k=k, project=project)
    if not ans or ans.upper().startswith("NÃO ENCONTRADO"):
        return None
    if path in LIST_FIELDS:
        return _to_list(ans)
    return ans

# navegação/atribuição por caminho "A.B.C"
def _set_path(obj: dict, path: str, value: Any):
    parts = path.split(".")
    cur = obj
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value

def _walk_template(template: dict) -> List[str]:
    """Lista todos os paths definidos no FIELD_QUESTIONS que existem/pertencem ao template."""
    return list(FIELD_QUESTIONS.keys())

def fill_file(in_path: Path, out_path: Path, project: Optional[str], k: int):
    tpl = json.loads(in_path.read_text(encoding="utf-8"))
    result = json.loads(json.dumps(tpl, ensure_ascii=False))  # deep copy

    paths = _walk_template(tpl)
    for p in paths:
        val = _fill_field(p, k=k, project=project)
        if val is not None:
            _set_path(result, p, val)

    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"OK: salvo em {out_path}")

def main():
    ap = argparse.ArgumentParser(description="Preenche perguntas_NI.json usando o RAG (índice + GPT)")
    ap.add_argument("--in", dest="inp", default="perguntas_NI.json", help="arquivo de entrada (template)")
    ap.add_argument("--out", dest="outp", default="respostas_NI.json", help="arquivo de saída")
    ap.add_argument("--project", help="ID do projeto (filtra o índice)")
    ap.add_argument("-k", type=int, default=5, help="top-k (ajustado automaticamente ao subset)")
    args = ap.parse_args()

    fill_file(Path(args.inp), Path(args.outp), args.project, args.k)

if __name__ == "__main__":
    main()
