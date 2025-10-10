from flask import Flask, jsonify
from BD.db_sqlite import listar_arquivos

app = Flask(__name__)

@app.route("/api/arquivos", methods=["GET"])
def get_arquivos():
    arquivos = listar_arquivos()
    # Retorna apenas id, nome e metadados
    return jsonify([
        {"id": a[0], "nome": a[1], "metadados": a[2]} for a in arquivos
    ])

if __name__ == "__main__":
    app.run(debug=True, port=5000)