import json
from google.genai import types
# Assuming your client is initialized in a config file or passed in
from API_client import client 

def run_indicator_agent(state: dict) -> dict:
    """
    LangGraph Node: Reads technical data from state, queries Gemini, and writes the JSON report back to state.
    """
    print(f"--- [Node] Running Indicator Agent for {state['symbol']} ---")
    
    # 1. Extract what we need from the shared state
    payload = state.get("technical_payload", {})
    
    # 2. Construct the Prompt (Same as our standalone script)
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

    # 3. Call the LLM
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0, 
                response_mime_type="application/json" 
            )
        )
        
        # Parse the JSON response
        report = json.loads(response.text)
        
        # 4. Return the specific keys of the state we want to UPDATE
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




# Generic test run 

if __name__ == "__main__":
    # 1. Create a mock AlgoState to simulate incoming data
    test_state = {
        "symbol": "AAPL",
        "timeframe": "1d",
        "user_entry_price": 225.50,
        "kline_data": {},
        "technical_payload": {
            "Asset": "AAPL",
            "Timeframe": "1d",
            "Current_Price": 230.10,
            "User_Position_Price": 225.50,
            "Recent_Action_Last_5_Days": {
                "2023-10-01": {"Close": 228.0, "EMA_9": 225.0, "EMA_14": 223.0},
                "2023-10-02": {"Close": 229.0, "EMA_9": 226.0, "EMA_14": 224.0},
                "2023-10-03": {"Close": 230.10, "EMA_9": 227.0, "EMA_14": 225.0}
            },
            "Fibonacci_Levels": {
                "0.0%_High": 235.0,
                "23.6%": 230.0,
                "38.2%": 225.0,
                "50.0%": 220.0,
                "61.8%": 215.0,
                "100.0%_Low": 200.0
            }
        },
        "indicator_report": None,
        "pattern_report": None,
        "messages": []
    }

    print("Initiating local test run for Indicator Node...")
    
    # 2. Execute the function with the mock state
    result = run_indicator_agent(test_state)
    
    # 3. Print the results to verify the payload updates
    print("\n=== TEST RESULT: STATE UPDATE PAYLOAD ===")
    print(json.dumps(result, indent=4))