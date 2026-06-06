"""Telegram alert sender — formats a trade report and pushes it to a channel."""

from __future__ import annotations

import json
import os
import requests
from pathlib import Path

# Key file location (one level above the repo root, out of VCS)
_KEYS_FILE = Path(__file__).resolve().parent.parent.parent / "telegram_keys.json"


def load_telegram_keys() -> dict | None:
    """Load bot_token and chat_id from telegram_keys.json."""
    try:
        return json.loads(_KEYS_FILE.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"Error: telegram_keys.json not found at {_KEYS_FILE}")
        return None


def send_telegram_alert(symbol: str, report: dict, image_buffer=None) -> None:
    """Format an AI trade report and push it (with optional chart) to Telegram."""
    keys = load_telegram_keys()
    if not keys:
        return

    bot_token = keys.get("bot_token")
    chat_id   = keys.get("chat_id")

    action = report.get("Action", "HOLD").upper()
    icon   = "🟢" if action == "BUY" else "🔴" if action == "SELL" else "⚪"
    pos_mgmt = report.get("Position_Management", {})

    message = (
        f"{icon} **ALGOEDGE ALERT: {symbol}** {icon}\n\n"
        f"**Action:** {action}\n"
        f"**Entry:** {report.get('Suggested_Entry', 'N/A')}\n\n"
        f"**Trend:** {report.get('Trend_Analysis', 'N/A')}\n\n"
        f"🎯 **Take Profit:** {pos_mgmt.get('Take_Profit', 'N/A')}\n"
        f"🛡️ **Stop Loss:** {pos_mgmt.get('Stop_Loss', 'N/A')}\n\n"
        f"💡 **Rationale:** {pos_mgmt.get('Rationale', 'N/A')}"
    )

    if image_buffer:
        url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
        image_buffer.seek(0)
        files = {"photo": ("chart.png", image_buffer, "image/png")}
        data  = {"chat_id": chat_id, "caption": message, "parse_mode": "Markdown"}
        response = requests.post(url, data=data, files=files)
    else:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        response = requests.post(
            url,
            json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"},
        )

    try:
        if response.status_code == 200:
            print("Successfully pushed alert to Telegram.")
        else:
            print(f"Failed to send Telegram alert: {response.text}")
    except Exception as e:
        print(f"Error sending to Telegram: {e}")
