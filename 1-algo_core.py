import yfinance as yf #Library used for polling live data 
import matplotlib.pyplot as plt

# Define Phase 1 Asset & Time Horizon Logic
symbol = "AAPL"
timeframe = "1d" # Daily time horizon for large-caps

print(f"Polling live data for {symbol} on the {timeframe} timeframe...")

# Pull the last 6 months of data
df = yf.download(symbol, period="1mo", interval=timeframe)

def sample_plotter(df):
    # Check if data was successfully pulled
    if df.empty:
        print(f"Failed to fetch data for {symbol}. Check your connection or symbol name.")
    else:
        print("Data fetched successfully! Generating graph...")
        
        # Plotting the Closing Price
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['Close'], label=f'{symbol} Close Price', linewidth=1.5)
        
        # Formatting the graph
        plt.title(f"{symbol} Daily Closing Price (Phase 1 MVP Test)", fontsize=14)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Price (USD)", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Display the graph
        plt.show()

# sample_plotter(df)

def fib_retracement(df):
    if df.empty:
        print("Failed to fetch data.")
    else:
        # Extract Close price and force it into a 1D Pandas Series to handle yfinance updates
        close_price = df['Close'].squeeze()

        # 1. Calculate Exponential Moving Averages (EMA 9 & 14)
        df['EMA_9'] = close_price.ewm(span=9, adjust=False).mean()
        df['EMA_14'] = close_price.ewm(span=14, adjust=False).mean()

        # 2. Calculate dynamic Fibonacci Retracement Levels 
        # Using .to_numpy() guarantees we get a single number, not a Pandas Series
        recent_high = float(close_price.to_numpy().max())
        recent_low = float(close_price.to_numpy().min())
        price_diff = recent_high - recent_low

        fib_levels = {
            '0.0% (High)': recent_high,
            '23.6%': recent_high - 0.236 * price_diff,
            '38.2%': recent_high - 0.382 * price_diff,
            '50.0%': recent_high - 0.500 * price_diff,
            '61.8%': recent_high - 0.618 * price_diff,
            '100.0% (Low)': recent_low
        }

        print("Indicators calculated successfully! Generating graph...")
        
        # 3. Plot the Data and Indicators
        plt.figure(figsize=(14, 8))
        
        # Plot Price and EMAs
        plt.plot(df.index, close_price, label='Close Price', color='black', linewidth=1.5)
        plt.plot(df.index, df['EMA_9'], label='EMA 9', color='blue', linestyle='--')
        plt.plot(df.index, df['EMA_14'], label='EMA 14', color='orange', linestyle='--')
        
        # Plot Fibonacci Levels
        colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
        for (level_name, price), color in zip(fib_levels.items(), colors):
            # The 'price' variable is now guaranteed to be a flat number, satisfying Matplotlib
            plt.axhline(y=price, color=color, linestyle=':', alpha=0.6, label=f'Fib {level_name}')

        plt.title(f"{symbol} - Phase 1 Indicators (EMA & Fibonacci)", fontsize=14)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Price (USD)", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.legend(loc='upper left', fontsize=9)
        plt.show()

fib_retracement(df)