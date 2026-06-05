<<<<<<< HEAD
"""
Agent for trend analysis in high-frequency trading (HFT) context.
Uses LLM and toolkit to generate and interpret trendline charts for short-term prediction.
"""

import json
import time

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from openai import RateLimitError
=======
import json
import time
import copy
import numpy as np
from scipy.stats import linregress

# NOTE: langchain_core / openai are imported lazily inside the LLM-only helpers
# (invoke_with_retry, create_trend_agent) so the pure-math quant functions in this
# module (quantify_trend_strength / quantify_trend_from_kline) can be imported and
# used with only numpy + scipy — no LLM SDK required.

>>>>>>> d25a181f7efa17f29ba9008ec6dd624f72b86e8c


# --- Retry wrapper for LLM invocation ---
def invoke_with_retry(call_fn, *args, retries=3, wait_sec=4):
    """
    Retry a function call with exponential backoff for rate limits or errors.
    """
    try:
        from openai import RateLimitError
    except ImportError:  # openai not installed — treat as a generic retryable error
        RateLimitError = ()

    for attempt in range(retries):
        try:
            result = call_fn(*args)
            return result
        except RateLimitError:
            print(
                f"Rate limit hit, retrying in {wait_sec}s (attempt {attempt + 1}/{retries})..."
            )
        except Exception as e:
            print(
                f"Other error: {e}, retrying in {wait_sec}s (attempt {attempt + 1}/{retries})..."
            )
        # Only sleep if not the last attempt
        if attempt < retries - 1:
            time.sleep(wait_sec)
    raise RuntimeError("Max retries exceeded")


def _close_prices_from_kline(kline_data):
    """Extract close prices from kline_data (dict-of-lists or list-of-dicts)."""
    if isinstance(kline_data, dict) and "Close" in kline_data:
        return np.array(kline_data["Close"], dtype=float)
    return np.array([float(k["Close"]) for k in kline_data], dtype=float)


def quantify_trend_strength(close_prices):
    """
    Calculates a Linear Regression Channel and scores the current price position.
    Returns a normalized vector [-1.0, 1.0] for the Decision Agent.
    """
    y = np.asarray(close_prices, dtype=float)
    if y.size == 0 or len(y) < 10:
        return {"trend_direction": "None", "slope": 0.0, "normalized_signal": 0.0}

    x = np.arange(len(y))

    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    regression_line = slope * x + intercept

    std_dev = np.std(y - regression_line)
    upper_channel = regression_line + (2 * std_dev)
    lower_channel = regression_line - (2 * std_dev)

    current_price = y[-1]
    current_upper = upper_channel[-1]
    current_lower = lower_channel[-1]

    direction_label = "Uptrend" if slope > 0 else "Downtrend"

    channel_range = current_upper - current_lower
    if channel_range == 0:
        return {"trend_direction": "Flat", "slope": 0.0, "normalized_signal": 0.0}

    position = (current_price - current_lower) / channel_range

    if position > 1.0:
        normalized_signal = 1.0
    elif position < 0.0:
        normalized_signal = -1.0
    else:
        normalized_signal = 1.0 - (position * 2.0)

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
        "normalized_signal": round(normalized_signal, 3),
    }


def quantify_trend_from_kline(kline_data):
    """Run trend quantification on kline_data (dict-of-lists or list-of-dicts)."""
    return quantify_trend_strength(_close_prices_from_kline(kline_data))


def create_trend_agent(tool_llm, graph_llm, toolkit):
    """
<<<<<<< HEAD
    Create a trend analysis agent node for HFT. The agent uses precomputed images from state or falls back to tool generation.
    """
=======
    Create a trend analysis agent node for HFT.
    Combines LLM visual analysis with rigid Scipy Linear Regression mathematics.
    """
    from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
>>>>>>> d25a181f7efa17f29ba9008ec6dd624f72b86e8c

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

<<<<<<< HEAD
            # --- System prompt for LLM ---
=======
>>>>>>> d25a181f7efa17f29ba9008ec6dd624f72b86e8c
            system_prompt = (
                "You are a K-line trend pattern recognition assistant operating in a high-frequency trading context. "
                "You must first call the tool `generate_trend_image` using the provided `kline_data`. "
                "Once the chart is generated, analyze the image for support/resistance trendlines and known candlestick patterns. "
                "Only then should you proceed to make a prediction about the short-term trend (upward, downward, or sideways). "
                "Do not make any predictions before generating and analyzing the image."
            )

<<<<<<< HEAD
            # --- Compose messages for the first round ---
=======
>>>>>>> d25a181f7efa17f29ba9008ec6dd624f72b86e8c
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(
                    content=f"Here is the recent kline data:\n{json.dumps(state['kline_data'], indent=2)}"
                ),
            ]

<<<<<<< HEAD
            # --- Prepare tool chain ---
            chain = tool_llm.bind_tools(tools)

            # --- Step 1: Let LLM decide if it wants to call generate_trend_image ---
            ai_response = invoke_with_retry(chain.invoke, messages)
            messages.append(ai_response)

            # --- Step 2: Handle tool call (generate_trend_image) ---
=======
            chain = tool_llm.bind_tools(tools)

            # Step 1: Let LLM decide if it wants to call generate_trend_image
            ai_response = invoke_with_retry(chain.invoke, messages)
            messages.append(ai_response)

            # Step 2: Handle tool call
>>>>>>> d25a181f7efa17f29ba9008ec6dd624f72b86e8c
            if hasattr(ai_response, "tool_calls"):
                for call in ai_response.tool_calls:
                    tool_name = call["name"]
                    tool_args = call["args"]
<<<<<<< HEAD
                    # Always provide kline_data
                    import copy

=======
>>>>>>> d25a181f7efa17f29ba9008ec6dd624f72b86e8c
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

<<<<<<< HEAD
        # --- Step 3: Vision analysis with image (precomputed or generated) ---
=======
        # --- Step 3: Vision analysis with image ---
>>>>>>> d25a181f7efa17f29ba9008ec6dd624f72b86e8c
        if trend_image_b64:
            image_prompt = [
                {
                    "type": "text",
                    "text": (
                        f"This candlestick ({time_frame} K-line) chart includes automated trendlines: the **blue line** is support, and the **red line** is resistance.\n\n"
                        "Analyze how price interacts with these lines — are candles bouncing off, breaking through, or compressing between them?\n\n"
                        "Based on trendline slope, spacing, and recent K-line behavior, predict the likely short-term trend: **upward**, **downward**, or **sideways**. "
<<<<<<< HEAD
                        "Support your prediction with respect to prediction, reasoning, signals."
=======
                        "Support your prediction with reasoning."
>>>>>>> d25a181f7efa17f29ba9008ec6dd624f72b86e8c
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{trend_image_b64}"},
                },
            ]

<<<<<<< HEAD
            # Create messages - ensure HumanMessage has valid content
            # For Anthropic, SystemMessage is extracted separately, but messages array must have at least one message
            human_msg = HumanMessage(content=image_prompt)
            
            # Verify HumanMessage content is valid
            if not human_msg.content:
                raise ValueError("HumanMessage content is empty")
            if isinstance(human_msg.content, list) and len(human_msg.content) == 0:
                raise ValueError("HumanMessage content list is empty")
            
            messages = [
                SystemMessage(
                    content="You are a K-line trend pattern recognition assistant operating in a high-frequency trading context. "
                    "Your task is to analyze candlestick charts annotated with support and resistance trendlines."
=======
            human_msg = HumanMessage(content=image_prompt)
            
            if not human_msg.content or (isinstance(human_msg.content, list) and len(human_msg.content) == 0):
                raise ValueError("HumanMessage content is empty")
            
            messages = [
                SystemMessage(
                    content="You are a K-line trend pattern recognition assistant operating in a high-frequency trading context."
>>>>>>> d25a181f7efa17f29ba9008ec6dd624f72b86e8c
                ),
                human_msg,
            ]
            
            try:
<<<<<<< HEAD
                final_response = invoke_with_retry(
                    graph_llm.invoke,
                    messages,
                )
            except Exception as e:
                error_str = str(e)
                # Handle Anthropic's "at least one message is required" error
                # This can happen when SystemMessage extraction leaves empty messages array
                if "at least one message" in error_str.lower():
                    # Retry with only HumanMessage (SystemMessage will be lost but Anthropic should work)
                    print("Retrying with HumanMessage only due to Anthropic message conversion issue...")
                    final_response = invoke_with_retry(
                        graph_llm.invoke,
                        [human_msg],
                    )
=======
                final_response = invoke_with_retry(graph_llm.invoke, messages)
            except Exception as e:
                error_str = str(e)
                if "at least one message" in error_str.lower():
                    final_response = invoke_with_retry(graph_llm.invoke, [human_msg])
>>>>>>> d25a181f7efa17f29ba9008ec6dd624f72b86e8c
                else:
                    raise
        else:
            final_response = invoke_with_retry(chain.invoke, messages)

        kline_data = state.get("kline_data", {})
        math_metrics = {"trend_direction": "Unknown", "normalized_signal": 0.0}

        try:
            math_metrics = quantify_trend_from_kline(kline_data)
        except Exception as e:
            print(f"Mathematical trend evaluation failed: {e}")

        # Combine LLM narrative and Math into a single JSON payload
        combined_report = json.dumps({
            "llm_analysis": final_response.content,
            "quantitative_metrics": math_metrics
        }, indent=4)

        return {
            "messages": messages + [final_response],
<<<<<<< HEAD
            "trend_report": final_response.content,
=======
            "trend_report": combined_report,
>>>>>>> d25a181f7efa17f29ba9008ec6dd624f72b86e8c
            "trend_image": trend_image_b64,
            "trend_image_filename": "trend_graph.png",
            "trend_image_description": (
                "Trend-enhanced candlestick chart with support/resistance lines"
<<<<<<< HEAD
                if trend_image_b64
                else None
=======
                if trend_image_b64 else None
>>>>>>> d25a181f7efa17f29ba9008ec6dd624f72b86e8c
            ),
        }

    return trend_agent_node
