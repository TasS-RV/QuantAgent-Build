import os
import json
import requests

def load_telegram_keys():
    """Loads Telegram credentials from the parent directory."""
    key_path = os.path.abspath(os.path.join(os.getcwd(), '..', 'telegram_keys.json'))
    try:
        with open(key_path, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: telegram_keys.json not found at {key_path}")
        return None

def send_telegram_alert(symbol, report):
    """Formats the AI report and pushes it to Telegram."""
    keys = load_telegram_keys()
    if not keys:
        return
    
    bot_token = keys.get("bot_token")
    chat_id = keys.get("chat_id")
    
    # 1. Format the JSON into a clean, readable alert message
    action = report.get("Action", "HOLD").upper()
    
    # Use emojis to make the alert scannable on mobile
    if action == "BUY":
        icon = "🟢"
    elif action == "SELL":
        icon = "🔴"
    else:
        icon = "⚪"
        
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

    # 2. Push to the Telegram API
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print("🚀 Successfully pushed alert to Telegram!")
        else:
            print(f"Failed to send Telegram alert: {response.text}")
    except Exception as e:
        print(f"Error sending to Telegram: {e}")