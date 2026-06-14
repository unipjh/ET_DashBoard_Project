import time
import requests
from fastapi import APIRouter

router = APIRouter(prefix="/api/stocks", tags=["stocks"])

_cache: dict = {"data": None, "ts": 0.0}
CACHE_TTL = 60  # 1분

_TICKERS = {
    "KOSPI":  "^KS11",
    "KOSDAQ": "^KQ11",
    "DOW":    "^DJI",
}
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ETDashboard/1.0)"}


def _fetch(symbol: str) -> dict:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    r = requests.get(url, headers=_HEADERS, timeout=6)
    r.raise_for_status()
    meta = r.json()["chart"]["result"][0]["meta"]
    price = meta.get("regularMarketPrice", 0)
    prev = meta.get("previousClose") or meta.get("chartPreviousClose") or price
    change_pct = ((price - prev) / prev * 100) if prev else 0.0
    return {"price": round(price, 2), "change_pct": round(change_pct, 2)}


@router.get("")
def get_stocks():
    global _cache
    now = time.time()
    if _cache["data"] and now - _cache["ts"] < CACHE_TTL:
        return _cache["data"]

    result: dict = {}
    for name, symbol in _TICKERS.items():
        try:
            result[name] = _fetch(symbol)
        except Exception as e:
            result[name] = {"price": None, "change_pct": None, "error": str(e)}

    _cache = {"data": result, "ts": now}
    return result
