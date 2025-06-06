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

      Die ursprünglich gewählten 69 Datensätze enthielten jeweils nur eine eindeutige Individuen-ID, aber kein Tier kam mehrfach vor – daher war keine Paarbildung möglich.
      Um ein echtes Trainingsset zu erzeugen, habe ich das Filterkriterium erweitert:
      Alle Annotationen mit mindestens einer individual_id wurden zugelassen, und wenn mehrere IDs enthalten waren, habe ich einfach die erste ID verwendet.
      So konnte ich 6.825 Bilder nutzen und daraus über 471.000 positive sowie 471.000 negative Bildpaare generieren – eine gute Grundlage für das Training eines Re-ID-Modells.

   - Definition der Erfolgskriterien:
      Ich verwende Top-1 Accuracy als Hauptmetrik: Das Modell soll das richtige Tierbild an erster Stelle erkennen.
      Während des Trainings verwende ich außerdem die Cosine Similarity, um zu prüfen, wie ähnlich zwei Bilder sind.
      Optional könnte ich später noch Top-5 Accuracy oder mean Average Precision (mAP) nutzen, wenn das Modell stabil läuft.


2. **Datenvorbereitung**
   - Strukturieren und ggf. Zuschneiden der Bilder
   - Kategorisierung nach Tier-ID
   - Sicherstellung einheitlicher Größen, Formate und Helligkeit

3. **Modellaufbau**
   - Kleines Siamese-Netzwerk in PyTorch erstellt
   - Arbeitet mit Graustufenbildern (1 Kanal, 100×100 Pixel)
   - Besteht aus gemeinsamem CNN + FC-Layer → 32-dimensionale Embeddings
   - Funktioniert mit `PairDataset` und gibt zwei Embeddings zurück
   - Erste Tests mit Zufallsbildern erfolgreich durchgeführt
   - Modell ist einsatzbereit für Training mit echten Paaren


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