# 🐅 wildlife-reid-tiger – ML-Projekt zur visuellen Wiedererkennung von Wildtieren

## 🌱 Ziel des Projekts

Dieses Projekt entwickelt ein Machine-Learning-Modell zur visuellen Wiedererkennung einzelner Tiere – mit Fokus auf Tiger. Ziel ist es, anhand von Kamerafallen-Bildern zu erkennen, ob dasselbe Tier mehrfach erscheint – trotz unterschiedlicher Perspektiven, Lichtverhältnisse oder Körperhaltungen.

Solche Re-Identification-Modelle werden bereits in der Wildtierforschung eingesetzt, um z. B. Populationsgrößen besser zu erfassen oder Bewegungsmuster zu analysieren.

## 🔍 Motivation

- **Ökologisch relevant**: Unterstützt Forschung zum Artenschutz
- **Technisch spannend**: Anwendung von Deep Learning mit Siamese-Architektur
- **Portfolio-wirksam**: Kombination aus Computer Vision, Re-ID und Datenverständnis

## 🧠 Geplante Schritte

1. **Projektplanung**
   - Zielsetzung und Recherche zu Re-ID im Kontext Wildtiere
   - Auswahl geeigneter öffentlich zugänglicher Datensätze
   - Definition der Erfolgskriterien (Accuracy, Ähnlichkeitsmetriken)

2. **Datenvorbereitung**
   - Strukturieren und ggf. Zuschneiden der Bilder
   - Kategorisierung nach Tier-ID
   - Sicherstellung einheitlicher Größen, Formate und Helligkeit

3. **Modellaufbau**
   - Entwicklung eines Siamese Networks in PyTorch
   - Vergleich von Bildpaaren (Triplet Loss oder Contrastive Loss)
   - Trainingsdatengenerierung (Positiv-/Negativpaare)

4. **Training und Evaluation**
   - Durchführung des Trainings
   - Visualisierung von Beispielen (z. B. ähnlich vs. unähnlich)
   - Confusion Matrix oder Ranking der Top-N ähnlichsten Bilder

5. **Dokumentation & Reflexion**
   - Erkenntnisse über Modellverhalten und Grenzen
   - Vorschläge zur Weiterentwicklung (mehr Daten, Tierarten, Web-Demo)

## 🧰 Technologien & Tools

- Python 3.10+
- PyTorch
- OpenCV
- Visual Studio Code
- (optional: TensorBoard, Streamlit, scikit-learn)

## 📁 Projektstruktur (geplant)

<p>
wildlife-reid-tiger/
├── data/ # Rohdaten und bearbeitete Bilder
├── models/ # Trainierte Modelle
├── src/ # Trainingslogik, Netzarchitekturen, Hilfsfunktionen
├── outputs/ # Beispielbilder, Visualisierungen
├── README.md
├── requirements.txt
└── .gitignore
</p>