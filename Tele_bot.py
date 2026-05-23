import os
import json
import requests

def load_telegram_keys():
    key_path = os.path.abspath(os.path.join(os.getcwd(), '..', 'telegram_keys.json'))
    try:
        with open(key_path, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: telegram_keys.json not found at {key_path}")
        return None

def send_telegram_alert(symbol, report, image_buffer=None):
    """Formats the AI report and pushes it with an optional image to Telegram."""
    keys = load_telegram_keys()
    if not keys:
        return
    
    bot_token = keys.get("bot_token")
    chat_id = keys.get("chat_id")
    
    action = report.get("Action", "HOLD").upper()
    icon = "🟢" if action == "BUY" else "🔴" if action == "SELL" else "⚪"
        
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

    # Push to Telegram API
    if image_buffer:
        url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
        # Reset the buffer pointer to the beginning before reading
        image_buffer.seek(0)
        files = {"photo": ("chart.png", image_buffer, "image/png")}
        data = {"chat_id": chat_id, "caption": message, "parse_mode": "Markdown"}
        response = requests.post(url, data=data, files=files)
    else:
        # Fallback if no image is provided
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
        response = requests.post(url, json=payload)
    
    try:
        if response.status_code == 200:
            print("🚀 Successfully pushed alert AND chart to Telegram!")
        else:
            print(f"Failed to send Telegram alert: {response.text}")
    except Exception as e:
        print(f"Error sending to Telegram: {e}")