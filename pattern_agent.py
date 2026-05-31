import copy
import json
import time

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from openai import RateLimitError


def invoke_tool_with_retry(tool_fn, tool_args, retries=3, wait_sec=4):
    """
    Invoke a tool function with retries if the result is missing an image.
    """
    for attempt in range(retries):
        result = tool_fn.invoke(tool_args)
        img_b64 = result.get("pattern_image")
        if img_b64:
            return result
        print(
            f"Tool returned no image, retrying in {wait_sec}s (attempt {attempt + 1}/{retries})..."
        )
        time.sleep(wait_sec)
    raise RuntimeError("Tool failed to generate image after multiple retries")


def create_pattern_agent(tool_llm, graph_llm, toolkit):
    """
    Create a pattern recognition agent node for candlestick pattern analysis.
    The agent uses precomputed images from state or falls back to tool generation.
    """

    def pattern_agent_node(state):
        tools = [toolkit.generate_kline_image]
        time_frame = state["time_frame"]
        stock_name = state.get("stock_name", "the asset")

        # Pattern prompt — kept as a plain string so the JSON schema braces are
        # never interpreted as template variables by LangChain's PromptTemplate.
        pattern_text = (
            f"You are a quantitative vision model analyzing a {time_frame} "
            f"candlestick chart for {stock_name}.\n"
            "Identify if any of these macro patterns are present:\n"
            "1. Inverse Head and Shoulders\n"
            "2. Double Bottom\n"
            "3. Rounded Bottom / Rounded Top\n"
            "4. Falling Wedge / Rising Wedge\n"
            "5. Ascending / Descending Triangle\n"
            "6. Bullish / Bearish Flag\n"
            "7. Rectangle\n"
            "8. Symmetrical Triangle\n\n"
            "OUTPUT STRICTLY AS A RAW JSON OBJECT. "
            "DO NOT wrap it in markdown block quotes (e.g., no ```json).\n"
            'Schema:\n{\n'
            '    "macro_pattern_name": "<Name of the pattern, or \'None\'>",\n'
            '    "direction": <1 for Bullish, -1 for Bearish, 0 for Neutral/None>,\n'
            '    "confidence_score": <Float between 0.0 and 1.0>,\n'
            '    "justification": "<One short sentence explaining why>"\n'
            "}"
        )

        # System content for the tool-calling step. Built as a plain string so
        # there is no template engine involved — avoids brace-interpolation errors.
        tool_system_content = (
            "You are a trading pattern recognition assistant. "
            "You have access to tool: generate_kline_image. "
            "Once generated, you MUST evaluate the chart using these rules:\n\n"
            + pattern_text
        )

        # Bind tools once; reused in both the tool-gen and text-fallback paths.
        llm_with_tools = tool_llm.bind_tools(tools)

        def invoke_with_retry(call_fn, *args, retries=3, wait_sec=8):
            for attempt in range(retries):
                try:
                    return call_fn(*args)
                except RateLimitError:
                    print(
                        f"Rate limit hit, retrying in {wait_sec}s "
                        f"(attempt {attempt + 1}/{retries})..."
                    )
                    time.sleep(wait_sec)
                except Exception as e:
                    print(
                        f"Other error: {e}, retrying in {wait_sec}s "
                        f"(attempt {attempt + 1}/{retries})..."
                    )
                    time.sleep(wait_sec)
            raise RuntimeError("Max retries exceeded")

        messages = state.get("messages", [])
        pattern_image_b64 = state.get("pattern_image")

        # ── Step 1: generate chart image via tool call (if not precomputed) ───
        if not pattern_image_b64:
            print("No precomputed pattern image found in state, generating with tool...")

            if not messages:
                messages = [HumanMessage(
                    content=(
                        "Please call generate_kline_image to produce a candlestick chart "
                        "so we can identify macro patterns."
                    )
                )]

            ai_response = invoke_with_retry(
                llm_with_tools.invoke,
                [SystemMessage(content=tool_system_content)] + messages,
            )
            messages.append(ai_response)

            if hasattr(ai_response, "tool_calls"):
                for call in ai_response.tool_calls:
                    tool_args = call["args"]
                    tool_args["kline_data"] = copy.deepcopy(state["kline_data"])
                    tool_fn = next(t for t in tools if t.name == call["name"])
                    tool_result = invoke_tool_with_retry(tool_fn, tool_args)
                    pattern_image_b64 = tool_result.get("pattern_image")
                    messages.append(
                        ToolMessage(
                            tool_call_id=call["id"],
                            content=json.dumps(tool_result),
                        )
                    )
        else:
            print("Using precomputed pattern image from state")

        # ── Step 2: vision analysis with the chart image ──────────────────────
        if pattern_image_b64:
            image_prompt = [
                {"type": "text", "text": pattern_text},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{pattern_image_b64}"},
                },
            ]
            human_msg = HumanMessage(content=image_prompt)

            if not human_msg.content or (
                isinstance(human_msg.content, list) and len(human_msg.content) == 0
            ):
                raise ValueError("HumanMessage content is empty")

            vision_messages = [
                SystemMessage(
                    content="You are a trading pattern recognition assistant "
                    "tasked with analyzing candlestick charts."
                ),
                human_msg,
            ]

            try:
                final_response = invoke_with_retry(graph_llm.invoke, vision_messages)
            except Exception as e:
                if "at least one message" in str(e).lower():
                    print("Retrying with HumanMessage only (Anthropic compatibility)...")
                    final_response = invoke_with_retry(graph_llm.invoke, [human_msg])
                else:
                    raise
        else:
            # Fallback: no image available — ask the LLM to reason from kline data text
            final_response = invoke_with_retry(
                llm_with_tools.invoke,
                [SystemMessage(content=tool_system_content)] + messages,
            )

        return {
            "messages": messages + [final_response],
            "pattern_report": final_response.content,
        }

    return pattern_agent_node
