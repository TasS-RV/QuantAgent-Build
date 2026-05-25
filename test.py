import yfinance as yf
import pandas as pd
import json
import talib

# Import the native modules from the QuantAgent repository
import static_util
from trading_graph import TradingGraph
from default_config import DEFAULT_CONFIG

# --- 1. The Mathematical Verifier ---
def quantify_pattern_strength(df, llm_report):
    """
    Parses the qualitative LLM report and uses TA-Lib to strictly 
    verify if the mathematical conditions for that pattern exist.
    """
    if df.empty or len(df) < 2:
        return {"pattern_name": "None", "direction": 0, "strength_score": 0.0, "normalized_signal": 0.0}

    # Extract NumPy arrays required by TA-Lib
    o = df['Open'].values
    h = df['High'].values
    l = df['Low'].values
    c = df['Close'].values

    report_lower = str(llm_report).lower()
    
    pattern_name = "Unknown / No Actionable Pattern"
    direction = 0
    ta_score = 0

    # Evaluate Engulfing
    if "engulfing" in report_lower:
        ta_score = talib.CDLENGULFING(o, h, l, c)[-1]
        if ta_score > 0:
            pattern_name = "Bullish Engulfing"
        elif ta_score < 0:
            pattern_name = "Bearish Engulfing"

    # Evaluate Hammer / Pin Bar
    elif "hammer" in report_lower or "pin" in report_lower:
        ta_score = talib.CDLHAMMER(o, h, l, c)[-1]
        if ta_score != 0:
            pattern_name = "Hammer"

    # Evaluate Doji
    elif "doji" in report_lower:
        ta_score = talib.CDLDOJI(o, h, l, c)[-1]
        if ta_score != 0:
            pattern_name = "Doji"

    # Process the TA-Lib Score (100 = Bullish, -100 = Bearish, 0 = None)
    if ta_score > 0:
        direction = 1
    elif ta_score < 0:
        direction = -1
        
    strength = 1.0 if direction != 0 else 0.0

    return {
        "pattern_name": pattern_name,
        "direction": direction,
        "strength_score": strength,
        "normalized_signal": float(direction * strength)
    }

# --- 2. The Main Execution Engine ---
def test_native_pattern_agent(symbol="NVDA", timeframe="1d", period="3mo"):
    print(f"Fetching {period} of {timeframe} data for {symbol}...")
    
    # Fetch and clean data
    df = yf.download(tickers=symbol, period=period, interval=timeframe, progress=False)
    if df.empty:
        print("Error: No data fetched.")
        return

    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.get_level_values(0):
            df.columns = df.columns.get_level_values(0)
        else:
            df.columns = df.columns.get_level_values(1)

    df = df.reset_index()
    df = df.rename(columns={df.columns[0]: "Datetime"})
    
    required_columns = ["Datetime", "Open", "High", "Low", "Close"]
    df = df[required_columns]

    df_slice = df.tail(45).reset_index(drop=True)

    df_slice_dict = {}
    for col in required_columns:
        if col == "Datetime":
            df_slice_dict[col] = df_slice[col].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
        else:
            df_slice_dict[col] = df_slice[col].tolist()

    print("Generating tool images...")
    p_image = static_util.generate_kline_image(df_slice_dict)
    t_image = static_util.generate_trend_image(df_slice_dict)

    initial_state = {
        "kline_data": df_slice_dict,
        "analysis_results": None,
        "messages": [],
        "time_frame": timeframe,
        "stock_name": symbol,
        "pattern_image": p_image.get("pattern_image"),
        "trend_image": t_image.get("trend_image"),
    }

    # Configure Graph for Google Gemini
    print("Initializing native TradingGraph with Gemini...")
    my_config = DEFAULT_CONFIG.copy()
    my_config["agent_llm_provider"] = "google"
    my_config["graph_llm_provider"] = "google"
    my_config["agent_llm_model"] = "gemini-2.5-flash"
    my_config["graph_llm_model"] = "gemini-2.5-flash"

    graph_engine = TradingGraph(config=my_config)
    
    print("Running LangGraph pipeline... (Extracting Pattern Report)")
    
    try:
        final_state = graph_engine.graph.invoke(initial_state)
        
        # 1. Extract the Qualitative LLM Report
        qualitative_report = final_state.get("pattern_report", "")
        
        # 2. INTEGRATION POINT: Pass the report and the dataframe into TA-Lib
        print("Calculating strict TA-Lib geometric strength...")
        quantified_metrics = quantify_pattern_strength(df_slice, qualitative_report)
        
        # 3. Output the results side-by-side
        print("\n" + "="*60)
        print(f"=== 1. GEMINI QUALITATIVE ANALYSIS ({symbol}) ===")
        print("="*60)
        print(qualitative_report)
        
        print("\n" + "="*60)
        print(f"=== 2. TA-LIB QUANTITATIVE METRICS ({symbol}) ===")
        print("="*60)
        print(json.dumps(quantified_metrics, indent=4))
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"An error occurred during execution: {e}")

if __name__ == "__main__":
    test_native_pattern_agent(symbol="BTC-USD", timeframe="1d", period="3mo")