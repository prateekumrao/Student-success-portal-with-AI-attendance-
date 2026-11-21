import pandas as pd
import requests

API_URL = "https://umrao.pythonanywhere.com/api/mark"
API_KEY = "Pkumrao@9956"

df = pd.read_csv("attendance.csv", dtype=str, index_col=False)

for _, r in df.iterrows():
    try:
        res = requests.post(
            API_URL,
            json={
                "key": API_KEY,
                "rollno": r["Roll No"],
                "name": r["Name"],
                "date": r["Date"],
                "time": r["Time"],
                "status": str(r["Status"]).strip().capitalize()
            },
            timeout=10
        )
        print(r["Roll No"], r["Name"], r["Date"], res.json())
    except Exception as e:
        print(r["Roll No"], r["Name"], "failed:", e)

print("Done.")