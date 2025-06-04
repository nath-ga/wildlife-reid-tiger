import pandas as pd
df = pd.read_csv("data/annotations_clean.csv")

print("Anzahl Zeilen:", len(df))
print("Anzahl einzigartiger IDs:", df["individual_id"].nunique())

# Jetzt: Nur solche, die mehr als einmal vorkommen
mehrfach = df["individual_id"].value_counts()
print("IDs mit mehrfachen Bildern:", (mehrfach > 1).sum())
