# Basis-Image verwenden
FROM python:3.8-slim

# Umgebungsvariablen setzen (optional)
ENV PYTHONUNBUFFERED=1

# Arbeitsverzeichnis erstellen
WORKDIR /app

# Systemabhängigkeiten installieren (falls erforderlich)
# RUN apt-get update && apt-get install -y [Pakete]

# Anforderungen kopieren und installieren
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Quellcode kopieren
COPY . .

# Ports freigeben (für TensorBoard oder andere Dienste)
EXPOSE 6006  # Port für TensorBoard

# Startbefehl festlegen
CMD ["python", "train.py"]
