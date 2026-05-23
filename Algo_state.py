'''
Defines the scheme of the shared memory dictionary passed between all AlgoEdge agents.
'''

from typing import TypedDict, Annotated, List, Any
import operator

class AlgoState(TypedDict):
    """
    The shared memory dictionary passed between all AlgoEdge agents.
    """
    symbol: str
    timeframe: str
    user_entry_price: float | None
    
    # Raw and processed data
    kline_data: dict
    technical_payload: dict
    
    # Agent Outputs
    indicator_report: dict | None
    pattern_report: str | None
    
    # LangGraph requires a messages list with an operator to append new messages
    messages: Annotated[list[str], operator.add]