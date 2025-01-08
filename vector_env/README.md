# Flappy Birds Vector Env

## Überblick
Das Projekt `vector_env` ist eine Umgebung zur Entwicklung und Evaluation von KI-Agenten. Es enthält mehrere Module zur Implementierung von Agenten, Trainings- und Bewertungsstrategien sowie zur Protokollierung und Visualisierung.

## Details zu den Ordnern

### `agents/`
Enthält die Implementierungen von KI-Agenten. Jeder Unterordner (`reward1`, `reward2`, etc.) zeigt eine die Reward-Strategie. Die Datei `baseline_agent.py` enthält einen einfachen Basisagenten, der als Referenz oder Vergleich dient

### `assets/`
Hier werden Dateien wie Bilder, Konfigurationsdateien oder andere Ressourcen gespeichert die für das Projekt benötigt werden

### `evaluation/`
Dieser Ordner enthält Skripte Bewertung der Performance

### `src/`
Dieser Ordner besitzt den Quellcode des Projekts. Er enthält die Game-Logik und alle benötigten Dateien. Ich habe hier das Spiel von dem folgenden Github Repository geklont und auf Basis dessen Vektoren hinzugefügt, die vorher nicht implementiert waren. Das war sehr dynamisch, da manche bereits vorhanden waren, andere wiedrum nicht. [Flappy Bird - sourabhv ](https://github.com/sourabhv/FlapPyBird/tree/master) Dazu gibt es auch ein Paper, welches darauf basiert aber das Reinforcement Learning mit Frames anstatt Vektoren gelernt hat. [Stanford Paper - DRL Flappy](https://cs229.stanford.edu/proj2015/362_report.pdf)

### `training/`
Enthält Skripte für das Training der Agenten. Jeder Unterordner (`reward1`, `reward2`, etc.) ist auf eine spezifische Reward-Strategie ausgerichtet. Beispielsweise enthalten die Dateien:
- `train_vector_DQN.py`: Training eines Agenten für die DQN-Methode
- `train_vector_PPO.py`: Training eines Agenten für die PPO-Methode
*Achtung* im Ordner `reward4` wurde die erste Reward Funktion verwendet mit den optimierten Hyperparametern für das PPO! Der DQN ist gleich geblieben und zeigt das geliche wie die erste Reward Funktion.

### `videos/`
Dieser Ordner enthält Videos, die für die Präsentation generiert worden sind, um die Leistung der Agenten zu visualisieren.

### Weitere Dateien
- `main.py`: Das Ausführen dieser Datei ermöglicht es das Flappy Bird Spiel selber zu spielen.

## Anforderungen
- Python 3.10.8
- Abhängigkeiten sind zu finden in der `requirements.txt`