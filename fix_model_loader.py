from pathlib import Path

path = Path("src/api/app.py")

text = path.read_text()

old = '''model_paths = [
            os.environ.get("MODEL_PATH"),
            "/app/models/xauusd_xgboost_model.joblib",
            "./models/xauusd_xgboost_model.joblib",
            "../models/xauusd_xgboost_model.joblib",
            "./xauusd_xgboost_model.joblib",
        ]'''

new = '''model_paths = [
            os.environ.get("MODEL_PATH"),
            "/app/models/xauusd_best_model.joblib",
            "./models/xauusd_best_model.joblib",
            "../models/xauusd_best_model.joblib",
            "./xauusd_best_model.joblib",
            "/app/models/xauusd_xgboost_model.joblib",
            "./models/xauusd_xgboost_model.joblib",
        ]'''

if old in text:
    text = text.replace(old, new)
    path.write_text(text)
    print("Model loader patched successfully")
else:
    print("Patch target not found - checking if already fixed")

