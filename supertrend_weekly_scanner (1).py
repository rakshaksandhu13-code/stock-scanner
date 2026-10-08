"""
Weekly Supertrend Scanner - Nifty 500 + NSE Indices
Weekly timeframe par red -> green flip dhoondta hai.

Report me: sector-wise summary, sector filter, aur stock pe dabate hi TradingView chart.
Install:  pip install yfinance pandas numpy requests plyer
Run:      python supertrend_weekly_scanner.py        (sirf completed weekly candle)
          python supertrend_weekly_scanner.py --live (running week ki candle bhi)
Email (optional): env vars GMAIL_USER, GMAIL_APP_PASSWORD, ALERT_TO
"""
import os
import io
import re
import sys
import html as _html
import datetime as dt
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
import yfinance as yf

PERIOD = 10          # ATR period
MULTIPLIER = 3.0     # ATR multiplier
RECENT_WEEKS = 4
NIFTY500_URL = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"

# Index ka naam: Yahoo ticker ke options (jo pehla chal jaye wo use hoga)
INDICES = {
    "NIFTY 50": ["^NSEI"],
    "SENSEX": ["^BSESN"],
    "NIFTY NEXT 50": ["^NSMIDCP", "NIFTY_NEXT_50.NS"],
    "NIFTY 100": ["^CNX100"],
    "NIFTY 200": ["^CNX200"],
    "NIFTY 500": ["^CRSLDX"],
    "NIFTY MIDCAP 50": ["^NSEMDCP50"],
    "NIFTY MIDCAP 100": ["NIFTY_MIDCAP_100.NS", "^CNXMIDCAP"],
    "NIFTY MIDCAP 150": ["NIFTYMIDCAP150.NS", "NIFTY_MIDCAP_150.NS"],
    "NIFTY MIDCAP SELECT": ["NIFTY_MID_SELECT.NS"],
    "NIFTY SMALLCAP 100": ["^CNXSC", "NIFTY_SMLCAP_100.NS"],
    "NIFTY SMALLCAP 250": ["NIFTYSMLCAP250.NS", "NIFTY_SMLCAP_250.NS"],
    "NIFTY LARGEMIDCAP 250": ["NIFTY_LARGEMID250.NS"],
    "NIFTY MIDSMALLCAP 400": ["NIFTY_MIDSML_400.NS"],
    "BANK NIFTY": ["^NSEBANK"],
    "NIFTY FIN SERVICE": ["NIFTY_FIN_SERVICE.NS", "^CNXFIN"],
    "NIFTY PRIVATE BANK": ["NIFTY_PVT_BANK.NS"],
    "NIFTY PSU BANK": ["^CNXPSUBANK"],
    "NIFTY IT": ["^CNXIT"],
    "NIFTY AUTO": ["^CNXAUTO"],
    "NIFTY PHARMA": ["^CNXPHARMA"],
    "NIFTY HEALTHCARE": ["NIFTY_HEALTHCARE.NS"],
    "NIFTY FMCG": ["^CNXFMCG"],
    "NIFTY METAL": ["^CNXMETAL"],
    "NIFTY REALTY": ["^CNXREALTY"],
    "NIFTY ENERGY": ["^CNXENERGY"],
    "NIFTY OIL & GAS": ["NIFTY_OIL_AND_GAS.NS"],
    "NIFTY CONSUMER DURABLES": ["NIFTY_CONSR_DURBL.NS"],
    "NIFTY CONSUMPTION": ["^CNXCONSUM"],
    "NIFTY COMMODITIES": ["^CNXCMDT"],
    "NIFTY MEDIA": ["^CNXMEDIA"],
    "NIFTY INFRA": ["^CNXINFRA"],
    "NIFTY PSE": ["^CNXPSE"],
    "NIFTY CPSE": ["NIFTY_CPSE.NS"],
    "NIFTY MNC": ["^CNXMNC"],
    "NIFTY SERVICES SECTOR": ["^CNXSERVICE"],
    "NIFTY INDIA MANUFACTURING": ["NIFTY_INDIA_MFG.NS"],
    "NIFTY INDIA DIGITAL": ["NIFTY_IND_DIGITAL.NS"],
}


def get_nifty500():
    """NSE se Nifty 500 list lo; fail ho to local nifty500.csv. Returns [(symbol, sector)]."""
    try:
        r = requests.get(NIFTY500_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
    except Exception as e:
        print(f"NSE se list nahi mili ({e}). Local 'nifty500.csv' try kar rahe hain...")
        df = pd.read_csv("nifty500.csv")
    out = []
    for _, row in df.iterrows():
        sym = str(row["Symbol"]).strip()
        sector = str(row.get("Industry", "Other")).strip() or "Other"
        out.append((sym, sector))
    return out


def tv_link(tv_symbol):
    return ("https://www.tradingview.com/chart/?symbol="
            + quote(tv_symbol, safe=":") + "&interval=W")


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


def prep(df, live, today):
    """Running (incomplete) week ki candle hata do, jab tak --live na ho."""
    if not live and df.index[-1].date() + dt.timedelta(days=5) > today:
        df = df.iloc[:-1]
    return df if len(df) >= PERIOD + 5 else None


def scan(live=False):
    """Returns (greens, totals, idx_rows, idx_missing)."""
    meta = {}
    for sym, sector in get_nifty500():
        meta[sym + ".NS"] = {"name": sym, "sector": sector,
                             "tv": "NSE:" + re.sub(r"[&-]", "_", sym)}
    all_tickers = list(meta.keys())
    for cands in INDICES.values():
        all_tickers += [c for c in cands if c not in all_tickers]

    data = download_weekly(all_tickers)
    today = dt.date.today()
    greens, totals = [], {}

    # ---- Stocks ----
    for tk, m in meta.items():
        df = data.get(tk)
        if df is None:
            continue
        df = prep(df, live, today)
        if df is None:
            continue
        totals[m["sector"]] = totals.get(m["sector"], 0) + 1
        d = supertrend(df)
        if d[-1] != 1:
            continue
        flips = np.where((d[1:] == 1) & (d[:-1] == -1))[0] + 1
        if len(flips) == 0:
            continue
        i = flips[-1]
        greens.append({
            "name": m["name"], "sector": m["sector"], "tv": m["tv"],
            "close": float(df["Close"].iloc[-1]),
            "flip_week": df.index[i].strftime("%d-%b-%Y"),
            "weeks_ago": int(len(d) - 1 - i),
        })
    greens.sort(key=lambda r: (r["weeks_ago"], r["name"]))

    # ---- Indices (green + red dono) ----
    idx_rows, idx_missing = [], []
    for nm, cands in INDICES.items():
        got = None
        for c in cands:
            if c in data:
                df = prep(data[c], live, today)
                if df is not None:
                    got = (c, df)
                    break
        if got is None:
            idx_missing.append(nm)
            continue
        c, df = got
        d = supertrend(df)
        changes = np.where(d[1:] != d[:-1])[0] + 1
        i = int(changes[-1]) if len(changes) else 0
        idx_rows.append({
            "name": nm, "close": float(df["Close"].iloc[-1]),
            "direction": int(d[-1]),
            "flip_week": df.index[i].strftime("%d-%b-%Y"),
            "weeks_ago": int(len(d) - 1 - i),
            "url": "https://finance.yahoo.com/chart/" + quote(c, safe=""),
        })
    idx_rows.sort(key=lambda r: (-r["direction"], r["weeks_ago"], r["name"]))
    return greens, totals, idx_rows, idx_missing


CSS = """
body{font-family:Arial,sans-serif;margin:14px;background:#f6f8f6;color:#222}
h1{color:#1b5e20;font-size:20px}h2{margin:22px 0 6px;font-size:17px}
.w{overflow-x:auto}
table{border-collapse:collapse;width:100%;max-width:760px;background:#fff;font-size:14px}
th,td{border:1px solid #ddd;padding:6px 8px;text-align:left;white-space:nowrap}
th{background:#2e7d32;color:#fff}
a{color:#0b57d0;text-decoration:none}a.sx{cursor:pointer}
select{font-size:16px;padding:6px;margin:6px 0;max-width:100%}
.hint{color:#555;font-size:13px}
"""

JS = """
function f(){
 var s=document.getElementById('sec').value;
 ['t_fresh','t_recent','t_older'].forEach(function(id){
  var n=0,none=null;
  document.querySelectorAll('#'+id+' tr[data-sec]').forEach(function(r){
   if(r.classList.contains('none')){none=r;return;}
   var show=(s===''||r.getAttribute('data-sec')===s);
   r.style.display=show?'':'none';
   if(show)n++;
  });
  if(none)none.style.display=(n===0)?'':'none';
  document.getElementById('c_'+id.slice(2)).textContent=n;
 });
}
document.addEventListener('click',function(e){
 if(e.target.classList.contains('sx')){
  e.preventDefault();
  document.getElementById('sec').value=e.target.getAttribute('data-s');
  f();
  window.scrollTo(0,document.getElementById('top_tables').offsetTop-10);
 }
});
"""


def build_html(greens, totals, idx_rows, idx_missing):
    esc = _html.escape
    fresh = [g for g in greens if g["weeks_ago"] == 0]
    recent = [g for g in greens if 1 <= g["weeks_ago"] <= RECENT_WEEKS]
    older = [g for g in greens if g["weeks_ago"] > RECENT_WEEKS]

    def rows(items):
        out = []
        for g in items:
            out.append(
                f'<tr data-sec="{esc(g["sector"], quote=True)}">'
                f'<td><a href="{esc(tv_link(g["tv"]), quote=True)}" target="_blank">{esc(g["name"])}</a></td>'
                f'<td>{esc(g["sector"])}</td><td>{g["close"]:,.2f}</td>'
                f'<td>{g["flip_week"]}</td><td>{g["weeks_ago"]}</td></tr>')
        return "".join(out) + '<tr class="none" data-sec=""><td colspan="5">koi nahi</td></tr>'

    head = ("<tr><th>Name (chart)</th><th>Sector</th><th>Close</th>"
            "<th>Green hua (week)</th><th>Hafte pehle</th></tr>")

    # Sector summary
    stat = {s: {"fresh": 0, "recent": 0, "green": 0, "total": t} for s, t in totals.items()}
    for g in greens:
        st = stat.setdefault(g["sector"], {"fresh": 0, "recent": 0, "green": 0, "total": 0})
        st["green"] += 1
        if g["weeks_ago"] == 0:
            st["fresh"] += 1
        elif g["weeks_ago"] <= RECENT_WEEKS:
            st["recent"] += 1
    order = sorted(stat.items(),
                   key=lambda kv: (-(kv[1]["fresh"] + kv[1]["recent"]), -kv[1]["green"], kv[0]))
    srows = "".join(
        f'<tr><td><a class="sx" href="#" data-s="{esc(s, quote=True)}">{esc(s)}</a></td>'
        f'<td>{v["fresh"]}</td><td>{v["recent"]}</td>'
        f'<td>{v["green"]} / {v["total"]}'
        f'{" (%d%%)" % round(100 * v["green"] / v["total"]) if v["total"] else ""}</td></tr>'
        for s, v in order)
    options = '<option value="">Sab sectors</option>' + "".join(
        f'<option value="{esc(s, quote=True)}">{esc(s)}</option>' for s, _ in sorted(stat.items()))

    # Index section
    def idx_row(r):
        if r["direction"] == 1:
            label = "🟢 FRESH GREEN" if r["weeks_ago"] == 0 else f"🟢 Green ({r['weeks_ago']} hafte se)"
            bg = "#e8f5e9"
            when = "green hua"
        else:
            label = "🔴 Fresh Red" if r["weeks_ago"] == 0 else f"🔴 Red ({r['weeks_ago']} hafte se)"
            bg = "#ffebee"
            when = "red hua"
        return (f'<tr style="background:{bg}"><td><a href="{esc(r["url"], quote=True)}" target="_blank">'
                f'{esc(r["name"])}</a></td><td>{r["close"]:,.2f}</td><td>{label}</td>'
                f'<td>{r["flip_week"]}</td></tr>')

    n_green = sum(1 for r in idx_rows if r["direction"] == 1)
    miss = (f'<p class="hint">Data nahi mila ({len(idx_missing)}): {esc(", ".join(idx_missing))}</p>'
            if idx_missing else "")
    idx_html = (f'<h2>Index Supertrend - {n_green} green / {len(idx_rows)} index</h2>'
                '<div class="w"><table><tr><th>Index (chart)</th><th>Close</th><th>Status</th>'
                '<th>Flip week</th></tr>' + "".join(idx_row(r) for r in idx_rows) + '</table></div>' + miss)

    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Weekly Supertrend Scanner</title><style>{CSS}</style></head><body>
<h1>Weekly Supertrend ({PERIOD},{MULTIPLIER}) - {dt.date.today():%d-%b-%Y}</h1>
<p class="hint">Stock ke naam pe dabao to TradingView weekly chart khulega.
Sector ke naam pe dabao to sirf wahi sector dikhega.</p>

{idx_html}
<h2>Sector-wise stocks (sabse zyada naye signal upar)</h2>
<div class="w"><table>
<tr><th>Sector</th><th>Fresh</th><th>Recent</th><th>Abhi green / Total</th></tr>{srows}</table></div>

<h2 id="top_tables">Sector filter</h2>
<select id="sec" onchange="f()">{options}</select>

<h2>Fresh GREEN - is hafte (<span id="c_fresh">{len(fresh)}</span>)</h2>
<div class="w"><table id="t_fresh">{head}{rows(fresh)}</table></div>

<h2>Recent GREEN - pichle {RECENT_WEEKS} hafte (<span id="c_recent">{len(recent)}</span>)</h2>
<div class="w"><table id="t_recent">{head}{rows(recent)}</table></div>

<h2>Abhi bhi green me (<span id="c_older">{len(older)}</span>)</h2>
<div class="w"><table id="t_older">{head}{rows(older)}</table></div>
<script>{JS}f();</script></body></html>"""


def show_report(greens, totals, idx_rows, idx_missing):
    fresh = [g for g in greens if g["weeks_ago"] == 0]
    recent = [g for g in greens if 1 <= g["weeks_ago"] <= RECENT_WEEKS]

    def line(g):
        when = "is hafte" if g["weeks_ago"] == 0 else f"{g['weeks_ago']} hafte pehle"
        return f"  {g['name']:<16} {g['sector'][:22]:<22} Rs {g['close']:>10,.2f}  ({when})"

    print("\n" + "=" * 72)
    print(f" WEEKLY SUPERTREND ({PERIOD},{MULTIPLIER}) - {dt.date.today():%d-%b-%Y}")
    print("=" * 72)
    print(f"\n FRESH GREEN (is hafte): {len(fresh)}")
    print("\n".join(line(g) for g in fresh) or "  koi nahi")
    print(f"\n RECENT GREEN (pichle {RECENT_WEEKS} hafte): {len(recent)}")
    print("\n".join(line(g) for g in recent) or "  koi nahi")

    print(f"\n INDEX SUPERTREND: {sum(1 for r in idx_rows if r['direction'] == 1)} green / {len(idx_rows)}")
    for r in idx_rows:
        tag = "GREEN" if r["direction"] == 1 else "red  "
        print(f"  {tag}  {r['name']:<26} {r['close']:>10,.2f}  ({r['weeks_ago']} hafte se)")
    if idx_missing:
        print("  Data nahi mila:", ", ".join(idx_missing))

    path = os.path.abspath("scanner_report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(build_html(greens, totals, idx_rows, idx_missing))
    try:
        import webbrowser
        webbrowser.open("file://" + path)
    except Exception:
        pass
    print(f"\n Report: {path}")
    return fresh, recent


def notify(text, fresh):
    """CSV save + desktop popup + (optional) email."""
    fname = f"signals_{dt.date.today():%Y-%m-%d}.csv"
    pd.DataFrame(fresh, columns=["Name", "Close", "Week"]).to_csv(fname, index=False)
    print(f"\nCSV saved: {os.path.abspath(fname)}")

    try:
        from plyer import notification
        names = ", ".join(n for n, _, _ in sorted(fresh)) or "Koi fresh signal nahi"
        notification.notify(title=f"Weekly Supertrend GREEN ({len(fresh)})",
                            message=names[:250], timeout=15)
    except Exception as e:
        print("Popup skip (pip install plyer):", e)

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
    greens, totals, idx_rows, idx_missing = scan(live=live)
    fresh_rows, _ = show_report(greens, totals, idx_rows, idx_missing)

    fresh = [(g["name"], g["close"], g["flip_week"]) for g in fresh_rows]
    msg = f"Weekly Supertrend({PERIOD},{MULTIPLIER}) GREEN - {dt.date.today():%d-%b-%Y}\n\n"
    msg += ("Fresh Buy Signals:\n" + "\n".join(f"- {n}  Rs {c:,.2f}" for n, c, _ in sorted(fresh))
            if fresh else "Is hafte koi fresh green signal nahi mila.")
    notify(msg, fresh)

    try:  # double-click se chalao to window band na ho
        input("\nBand karne ke liye Enter dabao...")
    except Exception:
        pass
