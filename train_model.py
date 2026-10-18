import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

df = pd.read_csv("solver/dataset.csv")
X = df[["L", "a", "b", "H", "S", "V"]].values
y = df["label"].values

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_tr, y_tr)
pred = model.predict(X_te)
print("Acuratețe:", accuracy_score(y_te, pred))
print(classification_report(y_te, pred))

# Reantrenăm pe tot setul și salvăm
model.fit(X, y)
joblib.dump(model, "solver/color_model.joblib")
print("Model salvat în solver/color_model.joblib")