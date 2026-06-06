"""Telegram sender — single-ticker alerts and full portfolio trade cards."""

from __future__ import annotations

import json
import os
import requests
from pathlib import Path

# Look for telegram_keys.json in the repo root first, then one level up.
_REPO_ROOT  = Path(__file__).resolve().parent.parent
_KEYS_CANDIDATES = [
    _REPO_ROOT / "telegram_keys.json",                   # QuantAgent-Build/
    _REPO_ROOT.parent / "telegram_keys.json",            # Playground/
]


def load_telegram_keys() -> dict | None:
    """
    Load {bot_token, chat_id}.

    Priority:
      1. telegram_keys.json  (repo root or one level up)
      2. .env.keys  (telegram_bot_token / telegram_chat_id)
      3. TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID env vars
    """
    # 1. JSON file
    for path in _KEYS_CANDIDATES:
        if path.is_file():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                pass

    # 2. .env.keys
    try:
        from env_keys import get_key
        token   = get_key("telegram_bot_token", "TELEGRAM_BOT_TOKEN")
        chat_id = get_key("telegram_chat_id",   "TELEGRAM_CHAT_ID")
        if token and chat_id:
            return {"bot_token": token, "chat_id": chat_id}
    except ImportError:
        pass

    # 3. OS env vars
    token   = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if token and chat_id:
        return {"bot_token": token, "chat_id": chat_id}

    candidates_str = "\n  ".join(str(p) for p in _KEYS_CANDIDATES)
    print(
        f"[Telegram] Keys not found. Checked:\n  {candidates_str}\n"
        "  .env.keys  (telegram_bot_token / telegram_chat_id)\n"
        "  TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID env vars"
    )
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  Core send
# ─────────────────────────────────────────────────────────────────────────────

def send_message(text: str, parse_mode: str = "Markdown") -> bool:
    """
    Send a pre-formatted text message to the configured Telegram chat.
    Returns True on success.
    """
    keys = load_telegram_keys()
    if not keys:
        return False
    url  = f"https://api.telegram.org/bot{keys['bot_token']}/sendMessage"
    resp = requests.post(
        url,
        json={"chat_id": keys["chat_id"], "text": text, "parse_mode": parse_mode},
        timeout=15,
    )
    if resp.status_code == 200:
        print("[Telegram] Message sent.")
        return True
    print(f"[Telegram] Send failed ({resp.status_code}): {resp.text[:200]}")
    return False


def send_photo(image_buffer, caption: str = "", parse_mode: str = "Markdown") -> bool:
    """Send a photo with an optional caption."""
    keys = load_telegram_keys()
    if not keys:
        return False
    url = f"https://api.telegram.org/bot{keys['bot_token']}/sendPhoto"
    image_buffer.seek(0)
    resp = requests.post(
        url,
        data={"chat_id": keys["chat_id"], "caption": caption, "parse_mode": parse_mode},
        files={"photo": ("chart.png", image_buffer, "image/png")},
        timeout=30,
    )
    if resp.status_code == 200:
        print("[Telegram] Photo sent.")
        return True
    print(f"[Telegram] Photo send failed ({resp.status_code}): {resp.text[:200]}")
    return False


# ─────────────────────────────────────────────────────────────────────────────
#  Legacy single-ticker alert (kept for backwards compat)
# ─────────────────────────────────────────────────────────────────────────────

def send_telegram_alert(symbol: str, report: dict, image_buffer=None) -> None:
    """Format a single-ticker AI trade report and push to Telegram."""
    action   = report.get("Action", "HOLD").upper()
    icon     = "🟢" if action == "BUY" else "🔴" if action == "SELL" else "⚪"
    pos_mgmt = report.get("Position_Management", {})

    message = (
        f"{icon} *ALGOEDGE ALERT: {symbol}* {icon}\n\n"
        f"*Action:* {action}\n"
        f"*Entry:* {report.get('Suggested_Entry', 'N/A')}\n\n"
        f"*Trend:* {report.get('Trend_Analysis', 'N/A')}\n\n"
        f"🎯 *Take Profit:* {pos_mgmt.get('Take_Profit', 'N/A')}\n"
        f"🛡️ *Stop Loss:* {pos_mgmt.get('Stop_Loss', 'N/A')}\n\n"
        f"💡 *Rationale:* {pos_mgmt.get('Rationale', 'N/A')}"
    )

    if image_buffer:
        send_photo(image_buffer, caption=message)
    else:
        send_message(message)
