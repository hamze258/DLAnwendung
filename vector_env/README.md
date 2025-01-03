# Deep Learning Anwendung - DQN & PPO Vektor based

Dieses Repository enthält die Struktur, Code und Konfigurationsdateien, damit man Modelle für Reinforcement Learning trainiert und evaluiert.

## Repository-Struktur

Hier ist eine Übersicht der Ordner und Dateien im Repository:

- **`vector_env`**: Parent Folder
- **`agents`**: Implementierungen von Agenten und Environments, kategorisiert in Reward Funktionen
- **`assets`**: Zusätzliche Dateien oder Ressourcen, um die Environment zu bauen
- **`evaluation`**:
  - **`reward1`, `reward2`, `reward3`**: Unterschiedliche Belohnungsstrategien oder Tests
  - **`eval.py`**: Skript zur Evaluation des Modells
- **`logs`**: Enthält Logs für Experimente oder Training
- **`models`**: Gespeicherte oder vortrainierte Modelle, mitunter auch checkpoints, kategoriesiert in die Netzwerke
- **`src`**: Quellcode für Kernkomponenten des Projekts
- **`tensorboard`**: TensorBoard-Dateien zur Visualisierung von Metriken
- **`training`**: Skripte für Training und Modellentwicklung
- **`videos`**: Videos von Simulationen
- **`main.py`**: Hauptskript zum Starten des Spiels in manuellem Modus
- **`config.yaml`**: Konfigurationsdatei für das Projekt
- **`requirements.txt`**: Liste der Python-Abhängigkeiten
- **`README.md`**: Diese Dokumentation

## Voraussetzungen

- Python 3.8 oder neuer
- Installiere die Abhängigkeiten:
  ```bash
  pip install -r requirements.txt
  ```
