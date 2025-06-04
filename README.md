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
      Ziel dieses Projekts ist es, ein Machine-Learning-Modell zu entwickeln, das individuelle afrikanische Leoparden anhand von Kamerafallen-Bildern wiedererkennt.  
      Die Herausforderung besteht darin, trotz unterschiedlicher Blickwinkel, Lichtverhältnisse und Posen das gleiche Tier zuverlässig zu identifizieren.  
      Der Datensatz „Leopard ID 2022“ liefert dafür echte Tier-IDs, Bounding Boxes und zusätzliche Informationen wie Blickrichtung und Zeitstempel.  
      Das Projekt soll die visuelle Wiedererkennung (Re-Identification) durch ein Siamese- oder Triplet-Netzwerk ermöglichen und die Qualität der Zuordnung mithilfe von geeigneten Metriken evaluieren.
   - Auswahl geeigneter öffentlich zugänglicher Datensätze:
     - **Verglichen**: Desert Lion, WCS Camera Traps, Leopard ID 2022
     - **Entschieden**: Leopard ID 2022, da echte Tier-IDs, Bounding Boxes, Blickrichtung, Zeitinfos enthalten sind → optimal für Re-Identification
      Datensatz: Botswana Predator Conservation Trust (2022). Panthera pardus CSV-Export. Abgerufen aus African Carnivore Wildbook vom 28.04.2022.
     - Erste Analyse der Metadaten ergab, dass nur 69 Bilder eine eindeutige Zuordnung zu genau einer Individuen-ID enthalten.  
      Um die Komplexität niedrig zu halten und einen realistischen Startpunkt zu schaffen, wird das initiale Modell auf genau diesen 69 Bildern aufgebaut.  
      Die restlichen Annotationen mit Mehrfachzuordnungen bleiben vorerst ungenutzt, könnten aber in einem zweiten Schritt eingebunden werden.

      Unter den 69 Datensätzen war keine Paarbildung möglich, ich habe daher für die Trainingsdaten einfach immer die erste Tier-ID genommen, wenn auf einem Bild mehrere angegeben waren. So konnte ich mehr Bilder behalten – und dadurch auch genug Paare bilden, um mein Modell zu trainieren.

   - Definition der Erfolgskriterien:
     - **Top-1 Accuracy** zur einfachen Bewertung
     - **Cosine Similarity** zur Paarbewertung im Trainingsprozess
     - Optional später: **Top-5 Accuracy**, **mean Average Precision (mAP)**

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