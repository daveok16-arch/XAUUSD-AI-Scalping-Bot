from pathlib import Path

p = Path("render.yaml")

text = p.read_text()

text = text.replace(
    "/app/models/xauusd_xgboost_model.joblib",
    "/app/models/xauusd_best_model.joblib"
)

p.write_text(text)

print("render.yaml patched")
