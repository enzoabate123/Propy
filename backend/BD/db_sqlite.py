import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "db.sqlite3"

# Cria a tabela se não existir
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS arquivos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nome TEXT NOT NULL,
            conteudo BLOB NOT NULL,
            metadados TEXT
        )
    ''')
    conn.commit()
    conn.close()

def inserir_arquivo(nome, conteudo, metadados=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO arquivos (nome, conteudo, metadados) VALUES (?, ?, ?)
    ''', (nome, conteudo, metadados))
    conn.commit()
    conn.close()

def buscar_arquivo_por_nome(nome):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        SELECT id, nome, conteudo, metadados FROM arquivos WHERE nome = ?
    ''', (nome,))
    resultado = c.fetchone()
    conn.close()
    return resultado

def listar_arquivos():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        SELECT id, nome, metadados FROM arquivos
    ''')
    resultados = c.fetchall()
    conn.close()
    return resultados

if __name__ == "__main__":
    init_db()
    # Teste: inserir e buscar arquivo
    inserir_arquivo("teste.txt", b"conteudo de teste", "{'descricao': 'arquivo de teste'}")
    print(buscar_arquivo_por_nome("teste.txt"))
    print(listar_arquivos())