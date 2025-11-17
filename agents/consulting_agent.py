from typing import TypedDict, List, Dict, Any
import json

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph


class AgentState(TypedDict, total=False):
    # Original user query from the user
    user_query: str

    # Full chat history (Human/AI messages)
    messages: List[BaseMessage]

    # Retrieved docs from Retriever Agent
    # Each doc: {"id": ..., "title": ..., "url": ..., "summary": ...}
    retrieved_docs: List[Dict[str, Any]]

    # Structured info about the patient collected so far
    patient_profile: Dict[str, Any]

    # How many times Consulting Agent has been called
    turn_count: int

    # Whether the agent thinks it still needs more info
    need_more_info: bool

    # Emergency / crisis flag
    red_flag: bool

    # What we want to show the user from this agent
    assistant_output: Dict[str, Any]


consulting_system_prompt = """
You are the Consulting Agent in a multi-agent health coaching system.

YOUR JOB:
- Clarify the user's symptoms and situation.
- Ask at most ONE follow-up question at a time.
- Update a structured patient_profile.
- Decide if more information is needed or if it is time to hand off to the Personal Care Agent.
- Never give diagnoses, treatment plans, or strong medical recommendations. Those are handled by the Personal Care Agent.

INPUTS YOU GET:
- The original user query.
- The full chat history so far.
- A small set of retrieved MedlinePlus snippets (trusted health information).
- A patient_profile with details collected so far.

STYLE:
- Warm, empathetic, non-judgmental, and clear.
- You can briefly explain WHY you are asking something.
- Always include a clear disclaimer that you are not a doctor and cannot diagnose.

RED FLAGS:
- If the description suggests a medical emergency,
  such as chest pain, difficulty breathing, signs of stroke,
  severe allergic reaction, high fever in a baby, or suicidal thoughts:
    - Set "red_flag" to true.
    - Set "need_more_info" to false.
    - Your follow-up question should instead be a very strong recommendation to seek immediate or urgent help.
    - Still do NOT provide a diagnosis.

OUTPUT FORMAT (IMPORTANT):
Return ONLY a JSON object with these fields:

{
  "followup_question": string or null,
  "explanation": string,
  "updated_patient_profile": object,
  "need_more_info": boolean,
  "red_flag": boolean,
  "handoff_reason": string
}

Rules:
- "followup_question" should be null if you think you have enough info
  and want to hand off to the Personal Care Agent.
- "need_more_info" should be false when you are ready to hand off.
- Keep "explanation" to 2-4 sentences, empathetic and simple.
- Never include medications or treatment instructions.
"""


def _make_llm() -> ChatOpenAI:
    """
    Create the OpenAI chat model.
    Requires OPENAI_API_KEY to be set in the environment.
    """
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
    )


llm = _make_llm()


def consulting_agent_node(state: AgentState) -> AgentState:
    # ---- 1. Pull things from state & set defaults ----
    turn_count = state.get("turn_count", 0) + 1
    patient_profile = state.get("patient_profile") or {}
    retrieved_docs = state.get("retrieved_docs") or []
    messages = state.get("messages") or []
    user_query = state.get("user_query", "")

    # ---- 2. Build MedlinePlus context string (top 3 docs) ----
    context_parts = []
    for doc in retrieved_docs[:3]:
        title = doc.get("title", "")
        summary = doc.get("summary") or doc.get("meta-desc") or ""
        url = doc.get("url", "")
        context_parts.append(
            f"Title: {title}\nSummary: {summary}\nURL: {url}"
        )
    retrieved_context = (
        "\n\n".join(context_parts) if context_parts else "No retrieved docs provided."
    )

    # ---- 3. Build messages for the model ----
    model_messages: List[BaseMessage] = []

    model_messages.append(SystemMessage(content=consulting_system_prompt))
    model_messages.append(SystemMessage(content=f"Original user query:\n{user_query}"))
    model_messages.append(
        SystemMessage(
            content=f"Current patient_profile (JSON):\n{json.dumps(patient_profile)}"
        )
    )
    model_messages.append(
        SystemMessage(content=f"Retrieved MedlinePlus context:\n{retrieved_context}")
    )

    # Previous conversation (human + AI)
    model_messages.extend(messages)

    # Instruction to produce JSON
    model_messages.append(
        HumanMessage(
            content="Given everything above, produce the JSON response exactly as specified."
        )
    )

    # ---- 4. Call the LLM ----
    result = llm.invoke(model_messages)
    raw_text = result.content

    # ---- 5. Parse JSON safely ----
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        parsed = {
            "followup_question": None,
            "explanation": (
                "I'm sorry, I had trouble formatting my response correctly. "
                "Given what you've shared, it may be best to hand you over to the "
                "next part of the system so you can still get help."
            ),
            "updated_patient_profile": patient_profile,
            "need_more_info": False,
            "red_flag": False,
            "handoff_reason": "json_parse_error",
        }

    # ---- 6. Merge patient_profile ----
    new_profile = parsed.get("updated_patient_profile") or {}
    if isinstance(new_profile, dict):
        patient_profile.update(new_profile)

    followup_question = parsed.get("followup_question")
    explanation = parsed.get("explanation", "")
    need_more_info = bool(parsed.get("need_more_info"))
    red_flag = bool(parsed.get("red_flag"))
    handoff_reason = parsed.get("handoff_reason") or ""

    # Soft cap: if we've already asked 3 times, stop asking for more info
    if turn_count >= 3 and need_more_info and not red_flag:
        need_more_info = False
        if not handoff_reason:
            handoff_reason = "max_turns_reached"

    # ---- 7. Update message history for future turns ----
    history_text = explanation
    if followup_question:
        history_text += f"\n\nFollow-up question: {followup_question}"

    messages.append(AIMessage(content=history_text))

    # ---- 8. Build assistant_output (for UI / planner) ----
    assistant_output = {
        "followup_question": followup_question,
        "explanation": explanation,
        "handoff_reason": handoff_reason,
    }

    # ---- 9. Write back to state ----
    state["turn_count"] = turn_count
    state["patient_profile"] = patient_profile
    state["messages"] = messages
    state["need_more_info"] = need_more_info
    state["red_flag"] = red_flag
    state["assistant_output"] = assistant_output

    return state


def build_consulting_agent_graph():
    """
    Build a LangGraph graph with a single 'consulting_agent' node.
    The Planner can call this graph with an AgentState and inspect the updated state.
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("consulting_agent", consulting_agent_node)

    workflow.set_entry_point("consulting_agent")
    workflow.set_finish_point("consulting_agent")

    return workflow.compile()
