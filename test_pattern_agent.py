import yfinance as yf
import pandas as pd
import json
import talib

# Import the native modules from the QuantAgent repository
import static_util
from trading_graph import TradingGraph
from default_config import DEFAULT_CONFIG

# --- 1. The Mathematical Verifier ---
def quantify_pattern_strength(df, raw_llm_response):
    """
    Combines the LLM's macro visual JSON output with TA-Lib's micro mathematical output.
    """
    if df.empty or len(df) < 2:
        return {"error": "Insufficient data"}

    # --- 1. PARSE THE MACRO VISION (LLM) ---
    macro_data = {
        "macro_pattern_name": "None", 
        "direction": 0, 
        "confidence_score": 0.0, 
        "macro_signal": 0.0
    }
    
    try:
        # Strip potential markdown formatting if the LLM disobeys the prompt
        clean_json = raw_llm_response.replace("```json", "").replace("```", "").strip()
        llm_json = json.loads(clean_json)
        
        macro_data["macro_pattern_name"] = llm_json.get("macro_pattern_name", "None")
        macro_data["direction"] = llm_json.get("direction", 0)
        macro_data["confidence_score"] = llm_json.get("confidence_score", 0.0)
        macro_data["macro_signal"] = round(macro_data["direction"] * macro_data["confidence_score"], 3)
    except json.JSONDecodeError:
        print(f"Failed to parse LLM JSON. Raw output was: {raw_llm_response}")

    # --- 2. CALCULATE THE MICRO MATH (TA-Lib) ---
    o, h, l, c = df['Open'].values, df['High'].values, df['Low'].values, df['Close'].values
    
    micro_name = "None"
    micro_direction = 0
    
    # Run independent TA-Lib checks on the last candle
    engulfing = talib.CDLENGULFING(o, h, l, c)[-1]
    hammer = talib.CDLHAMMER(o, h, l, c)[-1]
    doji = talib.CDLDOJI(o, h, l, c)[-1]

    if engulfing > 0:
        micro_name, micro_direction = "Bullish Engulfing", 1
    elif engulfing < 0:
        micro_name, micro_direction = "Bearish Engulfing", -1
    elif hammer != 0:
        micro_name, micro_direction = "Hammer", 1
    elif doji != 0:
        micro_name, micro_direction = "Doji", 0 # Indecision

    # --- 3. COMBINE INTO FINAL AGGREGATED VECTOR ---
    return {
        "macro_vision": macro_data,
        "micro_math": {
            "micro_pattern_name": micro_name,
            "micro_signal": float(micro_direction)
        }
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
    test_native_pattern_agent(symbol="NVDA", timeframe="1d", period="1mo")