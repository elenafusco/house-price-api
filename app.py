from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import joblib
import pandas as pd


# ============================================================
# 1. CARICAMENTO DEL MODELLO
# ============================================================

model = joblib.load("model.pkl")

print("Modello caricato correttamente!")


# ============================================================
# 2. CREAZIONE DEL SERVER
# ============================================================

class PredictionHandler(BaseHTTPRequestHandler):

    def do_POST(self):

        # Controlliamo che la richiesta sia /predict
        if self.path != "/predict":
            self.send_response(404)
            self.end_headers()
            return

        # Leggiamo i dati ricevuti
        content_length = int(
            self.headers["Content-Length"]
        )

        body = self.rfile.read(content_length)

        # Convertiamo JSON → Python
        data = json.loads(body)

        # ====================================================
        # 3. ESTRAIAMO I DATI DELLA CASA
        # ====================================================

        metri_quadri = data["metri_quadri"]
        numero_bagni = data["numero_bagni"]
        giardino = 1 if data["giardino"] else 0

        # ====================================================
        # 4. PREPARIAMO I DATI PER IL MODELLO
        # ====================================================

        nuova_casa = pd.DataFrame({
            "Metri_Quadri": [metri_quadri],
            "Numero_Bagni": [numero_bagni],
            "Giardino_bin": [giardino]
        })

        # ====================================================
        # 5. PREDIZIONE
        # ====================================================

        prezzo = model.predict(nuova_casa)[0]

        # ====================================================
        # 6. CREIAMO LA RISPOSTA JSON
        # ====================================================

        response = {
            "prezzo_stimato": round(float(prezzo), 2)
        }

        response_json = json.dumps(response)

        # ====================================================
        # 7. INVIO DELLA RISPOSTA
        # ====================================================

        self.send_response(200)

        self.send_header(
            "Content-Type",
            "application/json"
        )

        self.end_headers()

        self.wfile.write(
            response_json.encode("utf-8")
        )


# ============================================================
# 8. AVVIO SERVER
# ============================================================

server = HTTPServer(
    ("localhost", 8000),
    PredictionHandler
)

print("==========================================")
print("PYTHON API AVVIATA")
print("==========================================")
print("Server: http://localhost:8000")
print("Endpoint: POST /predict")
print("Premi CTRL+C per fermare il server.")

server.serve_forever()