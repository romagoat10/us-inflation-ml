"""Download a monthly US macro panel from DBnomics (https://db.nomics.world).

DBnomics mirrors the original sources: the Bureau of Labor Statistics (BLS), the Federal
Reserve Board (FED) and the IMF. The St. Louis Fed's FRED-MD file was not reachable
from this computer, so this builds a smaller panel of the same kind of series.

Each series has a FRED-MD style transformation code (McCracken and Ng, 2016):
    1 level, 2 first difference, 5 first difference of logs.

    python fetch_data.py    -> data/us_panel.csv and data/us_panel_tcodes.csv
"""
import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data"
OUT.mkdir(exist_ok=True)

# name: (DBnomics id, tcode, description)
SERIES = {
    "CPI":         ("BLS/cu/CUSR0000SA0",       5, "CPI, all items, SA (the target)"),
    "CPI_FOOD":    ("BLS/cu/CUSR0000SAF1",      5, "CPI, food, SA"),
    "CPI_ENERGY":  ("BLS/cu/CUSR0000SA0E",      5, "CPI, energy, SA"),
    "CPI_CORE":    ("BLS/cu/CUSR0000SA0L1E",    5, "CPI, all items less food and energy, SA"),
    "CPI_SHELTER": ("BLS/cu/CUSR0000SAH1",      5, "CPI, shelter, SA"),
    "CPI_SERV":    ("BLS/cu/CUSR0000SAS",       5, "CPI, services, SA"),
    "CPI_COMM":    ("BLS/cu/CUSR0000SAC",       5, "CPI, commodities, SA"),
    "PPI":         ("BLS/wp/WPU00000000",       5, "PPI, all commodities, NSA"),
    "UNRATE":      ("BLS/ln/LNS14000000",       2, "Unemployment rate, SA"),
    "PAYEMS":      ("BLS/ce/CES0000000001",     5, "Nonfarm payrolls, SA"),
    "AHE":         ("BLS/ce/CES0500000008",     5, "Average hourly earnings, production workers, SA"),
    "FEDFUNDS":    ("FED/H15/RIFSPFF_N.M",      2, "Federal funds rate, monthly"),
    "GS10":        ("FED/H15/RIFLGFCY10_N.M",   2, "10-year Treasury yield, monthly"),
    "TB3MS":       ("FED/H15/RIFSGFSM03_N.M",   2, "3-month Treasury bill, monthly"),
    "M2":          ("FED/H6_H6_M2/M2.M",        5, "M2 money stock, SA"),
    "INDPRO":      ("FED/G17_IP_MAJOR_INDUSTRY_GROUPS/IP.B50001.S", 5, "Industrial production, SA"),
    "USD_GBP":     ("FED/H10/RXI$US_N.M.UK",    5, "US dollars per pound"),
    "JPY_USD":     ("FED/H10/RXI_N.M.JA",       5, "Yen per US dollar"),
    "CAD_USD":     ("FED/H10/RXI_N.M.CA",       5, "Canadian dollars per US dollar"),
    "OIL":         ("BLS/wp/WPU0561",           5, "PPI, crude petroleum (domestic production), NSA"),
    "GAS":         ("BLS/wp/WPU0531",           5, "PPI, natural gas, NSA"),
}


def fetch(series_id: str) -> pd.Series:
    url = f"https://api.db.nomics.world/v22/series/{series_id}?observations=1&format=json"
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                doc = json.load(r)["series"]["docs"][0]
            break
        except Exception:
            if attempt == 2:
                raise
            time.sleep(3)
    s = pd.Series(doc["value"], index=pd.PeriodIndex(doc["period"], freq="M").to_timestamp())
    return pd.to_numeric(s.replace("NA", None), errors="coerce").sort_index()


def main():
    cols, tcodes, rows = {}, {}, []
    for name, (sid, code, desc) in SERIES.items():
        try:
            s = fetch(sid)
            cols[name], tcodes[name] = s, code
            rows.append(f"{name:12s} {s.first_valid_index():%Y-%m} to {s.last_valid_index():%Y-%m}  {sid}")
        except Exception as e:
            rows.append(f"{name:12s} FAILED {sid}: {type(e).__name__}")
        print(rows[-1], flush=True)
    panel = pd.DataFrame(cols)
    panel.index.name = "month"
    panel.to_csv(OUT / "us_panel.csv")
    pd.Series(tcodes, name="tcode").to_csv(OUT / "us_panel_tcodes.csv", index_label="series")
    (OUT / "fetch_log.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
