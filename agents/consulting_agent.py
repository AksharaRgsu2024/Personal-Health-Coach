import os
from dotenv import load_dotenv
load_dotenv()

import json
from typing import TypedDict, List, Dict, Any

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph


class AgentState(TypedDict, total=False):
    user_query: str
    messages: List[BaseMessage]
    retrieved_docs: List[Dict[str, Any]]
    patient_profile: Dict[str, Any]
    turn_count: int
    need_more_info: bool
    red_flag: bool
    assistant_output: Dict[str, Any]
    last_question: str  # for loop prevention


consulting_system_prompt = """
You are the Consulting Agent in a multi-agent health coaching system.

Your job:
- Clarify the user's symptoms and situation.
- Ask at most ONE follow-up question at a time.
- Update a structured patient_profile.
- Decide if more information is needed or if it is time to hand off.
- Never give diagnoses, treatment plans, or strong medical recommendations.
- Always include a brief, empathetic explanation.
- Hand off when need_more_info = false.

You also receive:
- Original user query
- Chat history
- Retrieved MedlinePlus snippets
- Current patient_profile

You must return ONLY a JSON object with this structure:

{
  "followup_question": string or null,
  "explanation": string,
  "updated_patient_profile": object,
  "need_more_info": boolean,
  "red_flag": boolean,
  "handoff_reason": string
}
"""


def _make_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
    )


llm = _make_llm()


def consulting_agent_node(state: AgentState) -> AgentState:
    turn_count = state.get("turn_count", 0) + 1

    patient_profile = state.get("patient_profile") or {}
    retrieved_docs = state.get("retrieved_docs") or []
    messages: List[BaseMessage] = state.get("messages") or []
    user_query = state.get("user_query", "")

    context_blocks = []
    for doc in retrieved_docs[:3]:
        context_blocks.append(
            f"Title: {doc.get('title', '')}\n"
            f"Summary: {doc.get('summary', '')}\n"
            f"URL: {doc.get('url', '')}"
        )
    retrieved_context = "\n\n".join(context_blocks) if context_blocks else "No documents."

    model_messages: List[BaseMessage] = [
        SystemMessage(content=consulting_system_prompt),
        SystemMessage(content=f"Original user query:\n{user_query}"),
        SystemMessage(content=f"Patient profile (JSON):\n{json.dumps(patient_profile)}"),
        SystemMessage(content=f"Retrieved MedlinePlus context:\n{retrieved_context}"),
    ]
    model_messages.extend(messages)
    model_messages.append(
        HumanMessage(content="Produce ONLY the required JSON object, nothing else.")
    )

    result = llm.invoke(model_messages)
    raw_text = result.content

    try:
        parsed = json.loads(raw_text)
    except Exception:
        parsed = {
            "followup_question": None,
            "explanation": (
                "I'm sorry, I had trouble formatting my response correctly. "
                "Given what you've shared, it may be best to hand you over "
                "to the next part of the system so you can still get help."
            ),
            "updated_patient_profile": patient_profile,
            "need_more_info": False,
            "red_flag": False,
            "handoff_reason": "json_parse_error",
        }

    new_profile = parsed.get("updated_patient_profile") or {}
    if isinstance(new_profile, dict):
        patient_profile.update(new_profile)

    followup_q = parsed.get("followup_question")
    explanation = parsed.get("explanation", "")
    need_more_info = bool(parsed.get("need_more_info"))
    red_flag = bool(parsed.get("red_flag"))
    handoff_reason = parsed.get("handoff_reason", "")

    last_q = state.get("last_question")

    # 1) If model repeats EXACT same question → stop asking
    if last_q and followup_q and followup_q.strip() == last_q.strip():
        need_more_info = False

    # 2) Hard cap for safety
    if turn_count >= 3:
        need_more_info = False
        if not handoff_reason:
            handoff_reason = "max_turns_reached"

    state["last_question"] = followup_q or last_q

    # Build message text for history
    msg_text = explanation
    if followup_q:
        msg_text += f"\n\nFollow-up question: {followup_q}"

    messages.append(AIMessage(content=msg_text))

    state["turn_count"] = turn_count
    state["patient_profile"] = patient_profile
    state["messages"] = messages
    state["need_more_info"] = need_more_info
    state["red_flag"] = red_flag
    state["assistant_output"] = {
        "followup_question": followup_q,
        "explanation": explanation,
        "handoff_reason": handoff_reason,
    }

    return state


def build_consulting_agent_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("consulting_agent", consulting_agent_node)
    workflow.set_entry_point("consulting_agent")
    workflow.set_finish_point("consulting_agent")
    return workflow.compile()
