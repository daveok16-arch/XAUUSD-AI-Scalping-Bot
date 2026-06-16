from pathlib import Path
import json

# Patch python source
src = Path("src/features/feature_engineering.py")

text = src.read_text()

text = text.replace(
    "import pandas_ta as pta",
    "import ta"
)

replacements = {
'pta.rsi(df["close"], length=14)':
'''ta.momentum.RSIIndicator(df["close"], window=14).rsi()''',

'pta.rsi(df["close"], length=7)':
'''ta.momentum.RSIIndicator(df["close"], window=7).rsi()''',

'pta.atr(df["high"], df["low"], df["close"], length=14)':
'''ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()''',

'pta.atr(df["high"], df["low"], df["close"], length=7)':
'''ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=7).average_true_range()''',

'pta.obv(df["close"], df["volume"])':
'''ta.volume.OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume()'''
}

for old,new in replacements.items():
    text=text.replace(old,new)

src.write_text(text)


# Patch notebook
nb = Path("notebooks/kaggle_training.ipynb")

data=json.loads(nb.read_text())

for cell in data["cells"]:
    if "source" in cell:
        cell["source"]=[
            line.replace("import pandas_ta as pta",
                         "import ta\n")
                 for line in cell["source"]
        ]

nb.write_text(json.dumps(data, indent=2))

print("Patch complete")
