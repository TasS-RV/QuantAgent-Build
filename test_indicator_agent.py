import yfinance as yf
import pandas as pd
import numpy as np
import talib
import json

# --- 1. THE MATHEMATICAL INDICATOR CALCULATOR ---
def quantify_indicators(df):
    """
    Calculates technical indicators using TA-Lib and normalizes them 
    into a [-1.0, 1.0] signal for the Decision Agent.
    """
    if df.empty or len(df) < 35:
        return {"error": "Not enough data to compute indicators. Need at least 35 candles."}

    # Extract NumPy arrays required by TA-Lib
    h = df['High'].values
    l = df['Low'].values
    c = df['Close'].values

    # Compute the 5 core indicators from the original toolkit
    # Note: Using standard HFT periods (e.g., 14 for RSI)
    rsi = talib.RSI(c, timeperiod=14)
    macd, macdsignal, macdhist = talib.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)
    roc = talib.ROC(c, timeperiod=10)
    slowk, slowd = talib.STOCH(h, l, c, fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    willr = talib.WILLR(h, l, c, timeperiod=14)

    # Grab the most recent values (the current state of the market)
    curr_rsi = rsi[-1]
    curr_macd = macd[-1]
    curr_macdhist = macdhist[-1]
    curr_roc = roc[-1]
    curr_stoch = slowk[-1]
    curr_willr = willr[-1]

    # --- NORMALIZATION LOGIC [-1.0 to 1.0] ---
    
    # 1. RSI Normalization: >70 = Bearish (-1), <30 = Bullish (+1)
    rsi_sig = max(min((50 - curr_rsi) / 20.0, 1.0), -1.0) if not np.isnan(curr_rsi) else 0.0

    # 2. Stochastic Normalization: >80 = Bearish (-1), <20 = Bullish (+1)
    stoch_sig = max(min((50 - curr_stoch) / 30.0, 1.0), -1.0) if not np.isnan(curr_stoch) else 0.0

    # 3. Williams %R Normalization: > -20 = Bearish (-1), < -80 = Bullish (+1)
    willr_sig = max(min((-50 - curr_willr) / 30.0, 1.0), -1.0) if not np.isnan(curr_willr) else 0.0

    # 4. MACD Histogram Normalization: Positive diff = Bullish momentum
    # CHANGED: Use np.nanstd to safely ignore TA-Lib warm-up NaNs
    if not np.isnan(curr_macdhist):
        hist_volatility = np.nanstd(macdhist[-30:])
        macd_sig = np.tanh(curr_macdhist / (hist_volatility + 1e-9)) if hist_volatility > 0 else 0.0
    else:
        macd_sig = 0.0

    # 5. ROC Normalization: Rate of change momentum
    # CHANGED: Use np.nanstd to safely ignore TA-Lib warm-up NaNs
    if not np.isnan(curr_roc):
        roc_volatility = np.nanstd(roc[-30:])
        roc_sig = np.tanh(curr_roc / (roc_volatility + 1e-9)) if roc_volatility > 0 else 0.0
    else:
        roc_sig = 0.0

    # --- AGGREGATION ---
    # Average the 5 normalized signals to get the final vector
    signals = [rsi_sig, stoch_sig, willr_sig, macd_sig, roc_sig]
    final_normalized_signal = np.mean(signals)

    return {
        "raw_metrics": {
            "RSI_14": round(curr_rsi, 2) if not np.isnan(curr_rsi) else 0.0,
            "MACD_Hist": round(curr_macdhist, 4) if not np.isnan(curr_macdhist) else 0.0,
            "Stochastic_K": round(curr_stoch, 2) if not np.isnan(curr_stoch) else 0.0,
            "Williams_R": round(curr_willr, 2) if not np.isnan(curr_willr) else 0.0,
            "ROC_10": round(curr_roc, 2) if not np.isnan(curr_roc) else 0.0
        },
        "component_signals": {
            "rsi_signal": round(rsi_sig, 3),
            "stoch_signal": round(stoch_sig, 3),
            "willr_signal": round(willr_sig, 3),
            "macd_signal": round(macd_sig, 3),
            "roc_signal": round(roc_sig, 3)
        },
        "final_indicator_signal": round(final_normalized_signal, 3)
    }

# --- 2. THE EXECUTION ENGINE ---
def test_quantitative_indicator_agent(symbol="NVDA", timeframe="1d", period="3mo", window=45, offset=0):
    print(f"Fetching {period} of {timeframe} data for {symbol}...")
    
    # Fetch data directly using yfinance
    df = yf.download(tickers=symbol, period=period, interval=timeframe, progress=False)
    
    if df.empty:
        print("Error: No data fetched.")
        return

    # Bulletproof data cleaning for yfinance MultiIndex updates
    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.get_level_values(0):
            df.columns = df.columns.get_level_values(0)
        else:
            df.columns = df.columns.get_level_values(1)

    df = df.reset_index()
    df = df.rename(columns={df.columns[0]: "Datetime"})
    
    required_columns = ["Datetime", "Open", "High", "Low", "Close", "Volume"]
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        # If Volume is missing, mock it safely (some crypto pairs lack it)
        if "Volume" in missing_cols:
            df["Volume"] = 1.0
        else:
            print(f"Failed to parse data. Missing: {missing_cols}")
            return
            
    df = df[required_columns]

    # Slice the historical data based on window and offset
    if offset > 0:
        df_slice = df.iloc[-(window + offset) : -offset].reset_index(drop=True)
    else:
        df_slice = df.tail(window).reset_index(drop=True)

    print(f"Running quantitative indicator math on {len(df_slice)} candles...")
    
    # --- INVOCATION ---
    quantitative_metrics = quantify_indicators(df_slice)
    
    print("\n" + "="*60)
    print(f"=== QUANTITATIVE INDICATOR METRICS ({symbol}) ===")
    print("="*60)
    print(json.dumps(quantitative_metrics, indent=4))
    print("="*60 + "\n")


if __name__ == "__main__":
    # Test on the current most recent n number of days
    test_quantitative_indicator_agent(symbol="GOOG", timeframe="1d", period="3mo", window=60, offset=0)