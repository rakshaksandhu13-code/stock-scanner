"""
Weekly Supertrend Scanner - Nifty 500 + NSE Indices
Fresh GREEN flip (red -> green) on weekly timeframe par Telegram alert.

Install:  pip install yfinance pandas numpy requests plyer
Alerts:   CSV file (auto) + desktop popup (pip install plyer) + optional Gmail
Email:    env vars GMAIL_USER, GMAIL_APP_PASSWORD (Google App Password), ALERT_TO (optional)
Run:      python supertrend_weekly_scanner.py            (sirf completed weekly candle)
          python supertrend_weekly_scanner.py --live     (running week ki candle bhi include)
"""
import os
import io
import sys
import datetime as dt

import numpy as np
import pandas as pd
import requests
import yfinance as yf

PERIOD = 10          # ATR period
MULTIPLIER = 3.0     # ATR multiplier
NIFTY500_URL = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"

INDICES = {
    "NIFTY 50": "^NSEI",
    "BANK NIFTY": "^NSEBANK",
    "NIFTY IT": "^CNXIT",
    "NIFTY AUTO": "^CNXAUTO",
    "NIFTY PHARMA": "^CNXPHARMA",
    "NIFTY FMCG": "^CNXFMCG",
    "NIFTY METAL": "^CNXMETAL",
    "NIFTY REALTY": "^CNXREALTY",
    "NIFTY ENERGY": "^CNXENERGY",
    "NIFTY PSU BANK": "^CNXPSUBANK",
    "NIFTY MEDIA": "^CNXMEDIA",
    "NIFTY INFRA": "^CNXINFRA",
    "NIFTY PSE": "^CNXPSE",
    "NIFTY MNC": "^CNXMNC",
    "NIFTY SERVICES": "^CNXSERVICE",
}


def get_nifty500():
    """NSE se Nifty 500 list lo; fail ho to local nifty500.csv use karo."""
    try:
        r = requests.get(NIFTY500_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
    except Exception as e:
        print(f"NSE se list nahi mili ({e}). Local 'nifty500.csv' try kar rahe hain...")
        df = pd.read_csv("nifty500.csv")  # NSE wali CSV yaha rakh do
    return {row["Symbol"]: row["Symbol"] + ".NS" for _, row in df.iterrows()}


def supertrend(df, period=PERIOD, mult=MULTIPLIER):
    """Returns direction array: 1 = green (bullish), -1 = red (bearish)."""
    h, l, c = df["High"].values, df["Low"].values, df["Close"].values
    prev_c = np.roll(c, 1)
    prev_c[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr = pd.Series(tr).ewm(alpha=1 / period, adjust=False).mean().values  # Wilder RMA

    hl2 = (h + l) / 2
    ub, lb = hl2 + mult * atr, hl2 - mult * atr
    n = len(c)
    fu, fl = ub.copy(), lb.copy()
    direction = np.ones(n, dtype=int)

    for i in range(1, n):
        fu[i] = ub[i] if (ub[i] < fu[i - 1] or c[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = lb[i] if (lb[i] > fl[i - 1] or c[i - 1] < fl[i - 1]) else fl[i - 1]
        if direction[i - 1] == -1 and c[i] > fu[i - 1]:
            direction[i] = 1
        elif direction[i - 1] == 1 and c[i] < fl[i - 1]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]
    return direction


def download_weekly(tickers, chunk=100):
    out = {}
    for i in range(0, len(tickers), chunk):
        batch = tickers[i:i + chunk]
        print(f"Downloading {i + 1}-{i + len(batch)} / {len(tickers)}")
        try:
            data = yf.download(batch, period="4y", interval="1wk", group_by="ticker",
                               auto_adjust=True, threads=True, progress=False)
        except Exception as e:
            print("Batch fail:", e)
            continue
        for t in batch:
            try:
                d = data[t] if len(batch) > 1 else data
                d = d.dropna(subset=["High", "Low", "Close"])
                if len(d) > PERIOD + 5:
                    out[t] = d
            except Exception:
                pass  # ticker data nahi mila, skip
    return out


def scan(live=False):
    """Har green stock ke liye record: name, close, flip week, weeks_ago (0 = is hafte green hua)."""
    names = {}
    for sym, tk in get_nifty500().items():
        names[tk] = sym
    for nm, tk in INDICES.items():
        names[tk] = nm

    data = download_weekly(list(names.keys()))
    today = dt.date.today()
    greens = []

    for tk, df in data.items():
        # Running (incomplete) week ki candle hata do, jab tak --live na ho
        if not live:
            if df.index[-1].date() + dt.timedelta(days=5) > today:
                df = df.iloc[:-1]
            if len(df) < PERIOD + 5:
                continue
        d = supertrend(df)
        if d[-1] != 1:
            continue
        flips = np.where((d[1:] == 1) & (d[:-1] == -1))[0] + 1
        if len(flips) == 0:
            continue  # data me kabhi red nahi tha, flip date pata nahi
        i = flips[-1]
        greens.append({
            "name": names[tk],
            "close": float(df["Close"].iloc[-1]),
            "flip_week": df.index[i].strftime("%d-%b-%Y"),
            "weeks_ago": int(len(d) - 1 - i),
        })
    greens.sort(key=lambda r: (r["weeks_ago"], r["name"]))
    return greens


def show_report(greens, recent_weeks=4):
    """Screen pe table + browser me HTML report."""
    fresh = [g for g in greens if g["weeks_ago"] == 0]
    recent = [g for g in greens if 1 <= g["weeks_ago"] <= recent_weeks]
    older = [g for g in greens if g["weeks_ago"] > recent_weeks]

    def line(g):
        when = "is hafte" if g["weeks_ago"] == 0 else f"{g['weeks_ago']} hafte pehle"
        return f"  {g['name']:<18} Rs {g['close']:>10,.2f}   green hua: {g['flip_week']} ({when})"

    print("\n" + "=" * 70)
    print(f" WEEKLY SUPERTREND ({PERIOD},{MULTIPLIER}) - {dt.date.today():%d-%b-%Y}")
    print("=" * 70)
    print(f"\n FRESH GREEN (is hafte flip hua): {len(fresh)}")
    print("\n".join(line(g) for g in fresh) or "  koi nahi")
    print(f"\n RECENT GREEN (pichle {recent_weeks} hafte me): {len(recent)}")
    print("\n".join(line(g) for g in recent) or "  koi nahi")
    print(f"\n Baaki jo abhi bhi green me hain: {len(older)} (HTML report me dekho)")

    def rows(items):
        return "".join(
            f"<tr><td>{g['name']}</td><td>{g['close']:,.2f}</td>"
            f"<td>{g['flip_week']}</td><td>{g['weeks_ago']}</td></tr>" for g in items
        ) or "<tr><td colspan=4>koi nahi</td></tr>"

    head = "<tr><th>Name</th><th>Close</th><th>Green hua (week)</th><th>Hafte pehle</th></tr>"
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Weekly Supertrend Scanner</title>
<style>body{{font-family:Arial;margin:24px;background:#f6f8f6}}
h1{{color:#1b5e20}}h2{{margin-top:28px}}
table{{border-collapse:collapse;width:100%;max-width:700px;background:#fff}}
th,td{{border:1px solid #ddd;padding:6px 10px;text-align:left}}
th{{background:#2e7d32;color:#fff}}</style></head><body>
<h1>Weekly Supertrend ({PERIOD},{MULTIPLIER}) - {dt.date.today():%d-%b-%Y}</h1>
<h2>Fresh GREEN - is hafte ({len(fresh)})</h2><table>{head}{rows(fresh)}</table>
<h2>Recent GREEN - pichle {recent_weeks} hafte ({len(recent)})</h2><table>{head}{rows(recent)}</table>
<h2>Abhi bhi green me ({len(older)})</h2><table>{head}{rows(older)}</table>
</body></html>"""
    path = os.path.abspath("scanner_report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    try:
        import webbrowser
        webbrowser.open("file://" + path)
    except Exception:
        pass
    print(f"\n Report: {path}")
    return fresh, recent


def notify(text, fresh):
    """CSV save + desktop popup + (optional) email. Telegram ki zarurat nahi."""
    # 1) CSV (hamesha)
    fname = f"signals_{dt.date.today():%Y-%m-%d}.csv"
    pd.DataFrame(fresh, columns=["Name", "Close", "Week"]).to_csv(fname, index=False)
    print(f"\nCSV saved: {os.path.abspath(fname)}")

    # 2) Desktop popup (pip install plyer) - nahi ho to skip
    try:
        from plyer import notification
        names = ", ".join(n for n, _, _ in sorted(fresh)) or "Koi fresh signal nahi"
        notification.notify(title=f"Weekly Supertrend GREEN ({len(fresh)})",
                            message=names[:250], timeout=15)
    except Exception as e:
        print("Popup skip (pip install plyer):", e)

    # 3) Email (optional) - GMAIL_USER, GMAIL_APP_PASSWORD env vars set karo
    user, pwd = os.getenv("GMAIL_USER"), os.getenv("GMAIL_APP_PASSWORD")
    if user and pwd:
        try:
            import smtplib
            from email.message import EmailMessage
            m = EmailMessage()
            m["Subject"] = f"Weekly Supertrend GREEN signals ({len(fresh)})"
            m["From"], m["To"] = user, os.getenv("ALERT_TO", user)
            m.set_content(text)
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as srv:
                srv.login(user, pwd)
                srv.send_message(m)
            print("Email bhej diya.")
        except Exception as e:
            print("Email fail:", e)


if __name__ == "__main__":
    live = "--live" in sys.argv
    greens = scan(live=live)
    fresh_rows, _ = show_report(greens)

    fresh = [(g["name"], g["close"], g["flip_week"]) for g in fresh_rows]
    msg = f"Weekly Supertrend({PERIOD},{MULTIPLIER}) GREEN - {dt.date.today():%d-%b-%Y}\n\n"
    msg += ("Fresh Buy Signals:\n" + "\n".join(f"- {n}  Rs {c:,.2f}" for n, c, _ in sorted(fresh))
            if fresh else "Is hafte koi fresh green signal nahi mila.")
    notify(msg, fresh)

    try:  # double-click se chalao to window band na ho
        input("\nBand karne ke liye Enter dabao...")
    except Exception:
        pass
