import os
import pandas as pd
import requests

def get_xauusd_data(interval="5min", outputsize=500):
    api_key = os.getenv("TWELVEDATA_API_KEY")

    if not api_key:
        raise RuntimeError("TWELVEDATA_API_KEY not configured")

    url = (
        "https://api.twelvedata.com/time_series"
        f"?symbol=XAU/USD"
        f"&interval={interval}"
        f"&outputsize={outputsize}"
        f"&apikey={api_key}"
    )

    r = requests.get(url, timeout=30)
    data = r.json()

    if "values" not in data:
        raise RuntimeError(str(data))

    df = pd.DataFrame(data["values"])

    for col in ["open","high","low","close","volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("datetime")
    df.reset_index(drop=True, inplace=True)

    return df
