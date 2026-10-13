import urllib.request
import json


# Dati della nuova casa
data = {
    "metri_quadri": 150,
    "numero_bagni": 2,
    "giardino": True
}


# Convertiamo i dati in JSON
json_data = json.dumps(data).encode("utf-8")


# Creiamo la richiesta
request = urllib.request.Request(
    "http://localhost:8000/predict",
    data=json_data,
    headers={
        "Content-Type": "application/json"
    },
    method="POST"
)


# Inviamo la richiesta
with urllib.request.urlopen(request) as response:

    result = response.read().decode("utf-8")

    print("==========================================")
    print("RISPOSTA DELLA API")
    print("==========================================")

    print(result)