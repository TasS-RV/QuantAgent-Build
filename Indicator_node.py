import json
import re
import yfinance as yf
import matplotlib.pyplot as plt
from google.genai import types
from API_client import client 

def run_indicator_agent(state: dict) -> dict:
    """
    LangGraph Node: Reads technical data from state, queries Gemini, and writes the JSON report back to state.
    """
    print(f"--- [Node] Running Indicator Agent for {state['symbol']} ---")
    
    payload = state.get("technical_payload", {})
    
    prompt = f"""
    You are the Indicator Agent for AlgoEdge, an expert in technical analysis.
    Analyze the following numerical data for {state['symbol']} on a {state['timeframe']} timeframe.

    Data:
    {json.dumps(payload, indent=2)}

    Task:
    Act as a trading advisor. Analyze the EMA momentum (9 vs 14) and the current price relative to the provided Fibonacci support/resistance levels. 

    Output a strict JSON object mapping exactly to this schema:
    {{
      "Trend_Analysis": "Summary of the current trend.",
      "Action": "Buy, Sell, or Hold",
      "Suggested_Entry": "Specify numeric price.",
      "Position_Management": {{
         "User_Held_Price": "The User_Position_Price or 'None'.",
         "Take_Profit": "Specify numeric price target.",
         "Stop_Loss": "Specify numeric price to cut losses.",
         "Rationale": "Brief explanation."
      }}
    }}
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0, 
                response_mime_type="application/json" 
            )
        )
        
        report = json.loads(response.text)
        
        return {
            "indicator_report": report,
            "messages": ["Indicator Agent successfully generated technical report."]
        }
        
    except Exception as e:
        print(f"Indicator Agent Error: {e}")
        return {
            "indicator_report": None,
            "messages": [f"Indicator Agent failed: {str(e)}"]
        }

# --- Visualisation Helper Engine ---

def extract_price(text_value):
    """Safely extracts the first floating-point number from an LLM string output."""
    if text_value is None or str(text_value).lower() == 'none':
        return None
    match = re.search(r'\d+\.?\d*', str(text_value))
    return float(match.group()) if match else None

def plot_algoedge_chart(df, symbol, fib_levels, report, vis_indicators=True, vis_price_points=True):
    """Generates the visual validation chart."""
    close_price = df['Close'].squeeze()
    
    plt.figure(figsize=(14, 8))
    plt.plot(df.index, close_price, label='Close Price', color='black', linewidth=1.5)
    
    # 1. Toggle: Visualise Indicators
    if vis_indicators:
        plt.plot(df.index, df['EMA_9'], label='EMA 9', color='blue', linestyle='--')
        plt.plot(df.index, df['EMA_14'], label='EMA 14', color='orange', linestyle='--')
        
        colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
        for (level_name, price), color in zip(fib_levels.items(), colors):
            plt.axhline(y=price, color=color, linestyle=':', alpha=0.4, label=f'Fib {level_name}')

    # 2. Toggle: Visualise Price Points from AI Report
    if vis_price_points and report:
        # Extract parsed numbers from the LLM's text output
        entry = extract_price(report.get("Suggested_Entry"))
        pos_mgmt = report.get("Position_Management", {})
        take_profit = extract_price(pos_mgmt.get("Take_Profit"))
        stop_loss = extract_price(pos_mgmt.get("Stop_Loss"))
        user_held = extract_price(pos_mgmt.get("User_Held_Price"))
        
        # Plot points with bold horizontal lines covering the last 20% of the graph for visibility
        x_start = df.index[int(len(df) * 0.8)]
        x_end = df.index[-1]

        if entry:
            plt.hlines(y=entry, xmin=x_start, xmax=x_end, color='cyan', linewidth=3, label=f'AI Entry ({entry})')
        if take_profit:
            plt.hlines(y=take_profit, xmin=x_start, xmax=x_end, color='green', linewidth=3, label=f'AI Take Profit ({take_profit})')
        if stop_loss:
            plt.hlines(y=stop_loss, xmin=x_start, xmax=x_end, color='red', linewidth=3, label=f'AI Stop Loss ({stop_loss})')
        if user_held:
            plt.hlines(y=user_held, xmin=x_start, xmax=x_end, color='purple', linewidth=3, label=f'User Held ({user_held})')

    plt.title(f"AlgoEdge Visual Validation - {symbol}", fontsize=14)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Position legend outside the plot to avoid clutter
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
    plt.tight_layout()
    plt.show()

# --- Execution & Testing Block ---

if __name__ == "__main__":
    # 1. Configuration 
    test_symbol = "NVDA"
    test_duration = "6mo" 
    timeframe = "1d"
    user_entry = 220.50 
    
    visualise_indicators = True
    visualise_price_points = True

    print(f"Fetching {test_duration} of live data for {test_symbol}...")
    
    # 2. Data Processing Pipeline
    df_raw = yf.download(test_symbol, period=test_duration, interval=timeframe, progress=False)
    
    # Squeeze out the tuple formatting into a flat series
    close_series = df_raw['Close'].squeeze()
    
    # Calculate EMAs and append them to the raw dataframe for the plotter
    df_raw['EMA_9'] = close_series.ewm(span=9, adjust=False).mean()
    df_raw['EMA_14'] = close_series.ewm(span=14, adjust=False).mean()
    
    recent_high = float(close_series.to_numpy().max())
    recent_low = float(close_series.to_numpy().min())
    price_diff = recent_high - recent_low

    fib_levels = {
        '0.0%_High': recent_high,
        '23.6%': recent_high - 0.236 * price_diff,
        '38.2%': recent_high - 0.382 * price_diff,
        '50.0%': recent_high - 0.500 * price_diff,
        '61.8%': recent_high - 0.618 * price_diff,
        '100.0%_Low': recent_low
    }

    # FIX: Manually build the recent data dictionary to guarantee string keys and flat floats
    clean_recent_data = {}
    for date, _ in df_raw.tail(5).iterrows():
        clean_recent_data[str(date.date())] = {
            "Close": float(close_series.loc[date]),
            "EMA_9": float(df_raw['EMA_9'].loc[date]),
            "EMA_14": float(df_raw['EMA_14'].loc[date])
        }

    # 3. Construct the State
    test_state = {
        "symbol": test_symbol,
        "timeframe": timeframe,
        "user_entry_price": user_entry,
        "kline_data": {},
        "technical_payload": {
            "Asset": test_symbol,
            "Timeframe": timeframe,
            "Current_Price": float(close_series.iloc[-1]),
            "User_Position_Price": user_entry,
            "Recent_Action_Last_5_Days": clean_recent_data,
            "Fibonacci_Levels": fib_levels
        },
        "indicator_report": None,
        "pattern_report": None,
        "messages": []
    }
    
    # 4. Run the Agent
    result = run_indicator_agent(test_state)
    
    print("\n=== AI REPORT ===")
    print(json.dumps(result["indicator_report"], indent=4))
    
    # 5. Trigger Visualisation
    if visualise_indicators or visualise_price_points:
        print("\nGenerating graph...")
        plot_algoedge_chart(
            df=df_raw, 
            symbol=test_symbol, 
            fib_levels=fib_levels, 
            report=result.get("indicator_report"), 
            vis_indicators=visualise_indicators, 
            vis_price_points=visualise_price_points
        )