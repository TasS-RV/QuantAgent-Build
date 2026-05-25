import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import linregress
import json

# Import the native modules from the QuantAgent repository
import static_util
from trading_graph import TradingGraph
from default_config import DEFAULT_CONFIG

# --- 1. THE MATHEMATICAL CALCULATOR ---
def quantify_trend_strength(df):
    """
    Calculates a Linear Regression Channel and scores the current price position.
    Returns a normalized vector [-1.0, 1.0] for the Decision Agent.
    """
    if df.empty or len(df) < 10:
        return {"trend_direction": "None", "slope": 0.0, "normalized_signal": 0.0}

    # Prepare the data
    y = df['Close'].values
    x = np.arange(len(y))

    # Calculate Linear Regression (The middle line)
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    regression_line = slope * x + intercept

    # Calculate Channel Boundaries (2 Standard Deviations)
    std_dev = np.std(y - regression_line)
    upper_channel = regression_line + (2 * std_dev) # Resistance
    lower_channel = regression_line - (2 * std_dev) # Support

    # Get current state (the final candle)
    current_price = y[-1]
    current_upper = upper_channel[-1]
    current_lower = lower_channel[-1]
    
    direction_label = "Uptrend" if slope > 0 else "Downtrend"

    # Calculate Channel Position (0.0 = at support, 1.0 = at resistance)
    channel_range = current_upper - current_lower
    if channel_range == 0:
        return {"trend_direction": "Flat", "slope": 0.0, "normalized_signal": 0.0}
        
    position = (current_price - current_lower) / channel_range

    # Calculate the Normalized Signal [-1.0 to 1.0]
    if position > 1.0:
        normalized_signal = 1.0 
    elif position < 0.0:
        normalized_signal = -1.0 
    else:
        normalized_signal = 1.0 - (position * 2.0)

    # Trend-continuation multiplier
    if slope > 0 and normalized_signal > 0:
        normalized_signal = min(normalized_signal * 1.2, 1.0) 
    elif slope < 0 and normalized_signal < 0:
        normalized_signal = max(normalized_signal * 1.2, -1.0) 

    return {
        "trend_direction": direction_label,
        "slope": round(slope, 4),
        "current_price": round(current_price, 2),
        "support_level": round(current_lower, 2),
        "resistance_level": round(current_upper, 2),
        "channel_position": round(position, 3),
        "normalized_signal": round(normalized_signal, 3)
    }

# --- 2. THE EXECUTION ENGINE ---
def test_native_trend_agent(symbol="NVDA", timeframe="1d", period="3mo", window=45, offset=0):
    print(f"Fetching {period} of {timeframe} data for {symbol}...")
    
    # Fetch data directly using yfinance
    df = yf.download(tickers=symbol, period=period, interval=timeframe, progress=False)
    
    if df.empty:
        print("Error: No data fetched.")
        return

    # Bulletproof data cleaning
    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.get_level_values(0):
            df.columns = df.columns.get_level_values(0)
        else:
            df.columns = df.columns.get_level_values(1)

    df = df.reset_index()
    df = df.rename(columns={df.columns[0]: "Datetime"})
    
    required_columns = ["Datetime", "Open", "High", "Low", "Close"]
    df = df[required_columns]
    
    
    # --- NEW SLICING LOGIC ---
    # Slice the window for the tool input based on the offset
    if offset > 0:
        # e.g., if window=45 and offset=10, slice from -55 to -10
        df_slice = df.iloc[-(window + offset) : -offset].reset_index(drop=True)
    else:
        # Default behavior: grab the most recent 'window' of candles
        df_slice = df.tail(window).reset_index(drop=True)

    # Safety check in case the offset pushes beyond available data
    if len(df_slice) < window:
        print(f"Warning: Only {len(df_slice)} candles available for this slice (requested {window}).")


    df_slice_dict = {}
    for col in required_columns:
        if col == "Datetime":
            df_slice_dict[col] = df_slice[col].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
        else:
            df_slice_dict[col] = df_slice[col].tolist()

    print("Generating tool images (Kline & Trend)...")
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

    print("Initializing native TradingGraph with Gemini...")
    my_config = DEFAULT_CONFIG.copy()
    my_config["agent_llm_provider"] = "google"
    my_config["graph_llm_provider"] = "google"
    
    # Using flash lite/1.5 to prevent Rate Limits
    my_config["agent_llm_model"] = "gemini-2.5-flash-lite"
    my_config["graph_llm_model"] = "gemini-2.5-flash-lite"

    graph_engine = TradingGraph(config=my_config)
    
    print("Running LangGraph pipeline... (Extracting Trend Report)")
    
    try:
        # Invoke the LLM graph
        final_state = graph_engine.graph.invoke(initial_state)
        qualitative_report = final_state.get("trend_report", "")
        
        # INVOCATION: Run the pure math calculator on the same dataframe slice
        print("Calculating rigid linear regression channel...")
        quantitative_metrics = quantify_trend_strength(df_slice)
        
        # Print outputs
        print("\n" + "="*60)
        print(f"=== 1. GEMINI QUALITATIVE ANALYSIS ({symbol}) ===")
        print("="*60)
        print(qualitative_report)
        
        print("\n" + "="*60)
        print(f"=== 2. MATHEMATICAL TREND METRICS ({symbol}) ===")
        print("="*60)
        print(json.dumps(quantitative_metrics, indent=4))
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"An error occurred during execution: {e}")

if __name__ == "__main__":
    # window = Size of the trend channel (default 45)
    # offset = How many candles backward to shift the end date (0 = today)
    
    # Example 1: Run on the current most recent 45 days
    # test_native_trend_agent(symbol="BTC-USD", timeframe="1d", period="3mo", window=45, offset=0)
    
    # Example 2: Backtest a 45-day trend from exactly 14 days ago
    test_native_trend_agent(symbol="NVDA", timeframe="1d", period="1mo", window=10, offset=5)