import json
import time
import copy
import numpy as np
from scipy.stats import linregress

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from openai import RateLimitError


# --- Retry wrapper for LLM invocation ---
def invoke_with_retry(call_fn, *args, retries=3, wait_sec=4):
    """
    Retry a function call with exponential backoff for rate limits or errors.
    """
    for attempt in range(retries):
        try:
            result = call_fn(*args)
            return result
        except RateLimitError:
            print(f"Rate limit hit, retrying in {wait_sec}s (attempt {attempt + 1}/{retries})...")
        except Exception as e:
            print(f"Other error: {e}, retrying in {wait_sec}s (attempt {attempt + 1}/{retries})...")
        # Only sleep if not the last attempt
        if attempt < retries - 1:
            time.sleep(wait_sec)
    raise RuntimeError("Max retries exceeded")


def create_trend_agent(tool_llm, graph_llm, toolkit):
    """
    Create a trend analysis agent node for HFT. 
    Combines LLM visual analysis with rigid Scipy Linear Regression mathematics.
    """

    def trend_agent_node(state):
        # --- Tool definitions ---
        tools = [toolkit.generate_trend_image]
        time_frame = state["time_frame"]

        # --- Check for precomputed image in state ---
        trend_image_b64 = state.get("trend_image")

        messages = []

        # --- If no precomputed image, fall back to tool generation ---
        if not trend_image_b64:
            print("No precomputed trend image found in state, generating with tool...")

            system_prompt = (
                "You are a K-line trend pattern recognition assistant operating in a high-frequency trading context. "
                "You must first call the tool `generate_trend_image` using the provided `kline_data`. "
                "Once the chart is generated, analyze the image for support/resistance trendlines and known candlestick patterns. "
                "Only then should you proceed to make a prediction about the short-term trend (upward, downward, or sideways). "
                "Do not make any predictions before generating and analyzing the image."
            )

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(
                    content=f"Here is the recent kline data:\n{json.dumps(state['kline_data'], indent=2)}"
                ),
            ]

            chain = tool_llm.bind_tools(tools)

            # Step 1: Let LLM decide if it wants to call generate_trend_image
            ai_response = invoke_with_retry(chain.invoke, messages)
            messages.append(ai_response)

            # Step 2: Handle tool call
            if hasattr(ai_response, "tool_calls"):
                for call in ai_response.tool_calls:
                    tool_name = call["name"]
                    tool_args = call["args"]
                    tool_args["kline_data"] = copy.deepcopy(state["kline_data"])
                    tool_fn = next(t for t in tools if t.name == tool_name)
                    tool_result = tool_fn.invoke(tool_args)
                    trend_image_b64 = tool_result.get("trend_image")
                    messages.append(
                        ToolMessage(
                            tool_call_id=call["id"], content=json.dumps(tool_result)
                        )
                    )
        else:
            print("Using precomputed trend image from state")

        # --- Step 3: Vision analysis with image ---
        if trend_image_b64:
            image_prompt = [
                {
                    "type": "text",
                    "text": (
                        f"This candlestick ({time_frame} K-line) chart includes automated trendlines: the **blue line** is support, and the **red line** is resistance.\n\n"
                        "Analyze how price interacts with these lines — are candles bouncing off, breaking through, or compressing between them?\n\n"
                        "Based on trendline slope, spacing, and recent K-line behavior, predict the likely short-term trend: **upward**, **downward**, or **sideways**. "
                        "Support your prediction with reasoning."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{trend_image_b64}"},
                },
            ]

            human_msg = HumanMessage(content=image_prompt)
            
            if not human_msg.content or (isinstance(human_msg.content, list) and len(human_msg.content) == 0):
                raise ValueError("HumanMessage content is empty")
            
            messages = [
                SystemMessage(
                    content="You are a K-line trend pattern recognition assistant operating in a high-frequency trading context."
                ),
                human_msg,
            ]
            
            try:
                final_response = invoke_with_retry(graph_llm.invoke, messages)
            except Exception as e:
                error_str = str(e)
                if "at least one message" in error_str.lower():
                    final_response = invoke_with_retry(graph_llm.invoke, [human_msg])
                else:
                    raise
        else:
            final_response = invoke_with_retry(chain.invoke, messages)

        # ==========================================
        # --- NEW: STRICT MATHEMATICAL CALCULATOR ---
        # ==========================================
        kline_data = state.get("kline_data", {})
        math_metrics = {"trend_direction": "Unknown", "normalized_signal": 0.0}
        
        try:
            # Handle both dictionary formats (list of dicts vs dict of lists)
            if isinstance(kline_data, dict) and "Close" in kline_data:
                y = np.array(kline_data["Close"], dtype=float)
            else:
                y = np.array([float(k['Close']) for k in kline_data])

            if len(y) > 5:
                x = np.arange(len(y))
                slope, intercept, _, _, _ = linregress(x, y)
                regression_line = slope * x + intercept
                std_dev = np.std(y - regression_line)
                
                current_price = y[-1]
                current_upper = regression_line[-1] + (2 * std_dev)
                current_lower = regression_line[-1] - (2 * std_dev)
                
                channel_range = current_upper - current_lower
                position = (current_price - current_lower) / channel_range if channel_range > 0 else 0.5
                
                # Base Signal
                if position > 1.0: normalized_signal = 1.0
                elif position < 0.0: normalized_signal = -1.0
                else: normalized_signal = 1.0 - (position * 2.0)

                # Alignment Boost
                if slope > 0 and normalized_signal > 0:
                    normalized_signal = min(normalized_signal * 1.2, 1.0)
                elif slope < 0 and normalized_signal < 0:
                    normalized_signal = max(normalized_signal * 1.2, -1.0)

                math_metrics = {
                    "trend_direction": "Uptrend" if slope > 0 else "Downtrend",
                    "slope": round(slope, 4),
                    "channel_position": round(position, 3),
                    "normalized_signal": round(normalized_signal, 3)
                }
        except Exception as e:
            print(f"Mathematical trend evaluation failed: {e}")

        # Combine LLM narrative and Math into a single JSON payload
        combined_report = json.dumps({
            "llm_analysis": final_response.content,
            "quantitative_metrics": math_metrics
        }, indent=4)

        return {
            "messages": messages + [final_response],
            "trend_report": combined_report,
            "trend_image": trend_image_b64,
            "trend_image_filename": "trend_graph.png",
            "trend_image_description": (
                "Trend-enhanced candlestick chart with support/resistance lines"
                if trend_image_b64 else None
            ),
        }

    return trend_agent_node