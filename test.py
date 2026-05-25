import yfinance as yf
import pandas as pd

# Import the native modules from the QuantAgent repository
import static_util
from trading_graph import TradingGraph
from default_config import DEFAULT_CONFIG

def test_native_pattern_agent(symbol="NVDA", timeframe="1d", period="3mo"):
    print(f"Fetching {period} of {timeframe} data for {symbol}...")
    
    # 1. Fetch data directly using yfinance
    df = yf.download(tickers=symbol, period=period, interval=timeframe, progress=False)
    
    if df.empty:
        print("Error: No data fetched.")
        return

    # --- BULLETPROOF DATA CLEANING ---
    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.get_level_values(0):
            df.columns = df.columns.get_level_values(0)
        else:
            df.columns = df.columns.get_level_values(1)

    df = df.reset_index()
    df = df.rename(columns={df.columns[0]: "Datetime"})
    
    required_columns = ["Datetime", "Open", "High", "Low", "Close"]
    
    # Safety check
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        print(f"Failed to parse yfinance data. Missing: {missing_cols}")
        return
        
    df = df[required_columns]

    # Slice the last 45 candles for the tool input
    df_slice = df.tail(45).reset_index(drop=True)

    # Convert to the dictionary format the tools expect
    df_slice_dict = {}
    for col in required_columns:
        if col == "Datetime":
            df_slice_dict[col] = df_slice[col].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
        else:
            df_slice_dict[col] = df_slice[col].tolist()

    print("Generating tool images...")
    # 2. Use the repository's native tool directly
    p_image = static_util.generate_kline_image(df_slice_dict)
    t_image = static_util.generate_trend_image(df_slice_dict)

    # 3. Construct the initial state
    initial_state = {
        "kline_data": df_slice_dict,
        "analysis_results": None,
        "messages": [],
        "time_frame": timeframe,
        "stock_name": symbol,
        "pattern_image": p_image.get("pattern_image"),
        "trend_image": t_image.get("trend_image"),
    }

    # 4. CONFIGURE GRAPH FOR GOOGLE GEMINI
    print("Initializing native TradingGraph with Gemini...")
    my_config = DEFAULT_CONFIG.copy()
    
    # Tell the graph to use the Google integration we just built
    my_config["agent_llm_provider"] = "google"
    my_config["graph_llm_provider"] = "google"
    
    # Define your specific models
    my_config["agent_llm_model"] = "gemini-2.5-flash"
    my_config["graph_llm_model"] = "gemini-2.5-flash"

    # Initialize the graph
    graph_engine = TradingGraph(config=my_config)
    
    print("Running LangGraph pipeline... (Extracting Pattern Report)")
    
    # 5. Invoke the graph
    try:
        final_state = graph_engine.graph.invoke(initial_state)
        
        # Extract the pattern report
        pattern_report = final_state.get("pattern_report")
        
        print("\n" + "="*50)
        print(f"=== GEMINI PATTERN REPORT FOR {symbol} ===")
        print("="*50 + "\n")
        print(pattern_report)
        print("\n" + "="*50)
        
    except Exception as e:
        print(f"An error occurred during execution: {e}")

if __name__ == "__main__":
    test_native_pattern_agent(symbol="BTC-USD", timeframe="1d", period="3mo")