from typing_extensions import TypedDict
from typing import List, Dict, Any
from datetime import date
from pydantic import BaseModel, TypeAdapter, field_validator
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import InMemorySaver
import json


class SymptomDict(TypedDict):
    date_recorded: str  # Changed to str for easier LLM handling
    duration_days: int
    severity: str
    description: str


class PatientProfile(BaseModel):
    id: str
    name: str
    sex: str
    age: int
    symptoms: List[SymptomDict]=[]
    
    @field_validator('symptoms', mode='before')
    @classmethod
    def convert_symptoms(cls, v):
        """Convert string symptoms to proper SymptomDict format"""
        if not v:
            return []
        
        result = []
        for item in v:
            if isinstance(item, str):
                # Convert string to SymptomDict
                result.append({
                    'date_recorded': str(date.today()),
                    'duration_days': 0,
                    'severity': 'unknown',
                    'description': item
                })
            elif isinstance(item, dict):
                # Ensure all required fields exist
                symptom_dict = {
                    'date_recorded': item.get('date_recorded', str(date.today())),
                    'duration_days': item.get('duration_days', 0),
                    'severity': item.get('severity', 'unknown'),
                    'description': item.get('description', '')
                }
                result.append(symptom_dict)
        return result


class AgentState(TypedDict, total=False):
    user_query: str
    messages: List[BaseMessage]
    retrieved_docs: List[Dict[str, Any]]
    patient_profile: PatientProfile
    turn_count: int
    need_more_info: bool
    red_flag: bool
    assistant_output: Dict[str, Any]
    last_question: str


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

IMPORTANT: When recording symptoms, use this exact structure:
{
  "date_recorded": "YYYY-MM-DD",
  "duration_days": <number>,
  "severity": "mild|moderate|severe|unknown",
  "description": "<symptom description>"
}

You must return ONLY a JSON object with this structure:

{
  "followup_question": string or null,
  "explanation": string,
  "updated_patient_profile": {
    "id": string,
    "name": string,
    "sex": string,
    "symptoms": [
      {
        "date_recorded": "YYYY-MM-DD",
        "duration_days": number,
        "severity": "mild|moderate|severe|unknown",
        "description": "symptom description"
      }
    ]
  },
  "need_more_info": boolean,
  "red_flag": boolean,
  "handoff_reason": string
}

Example of properly formatted symptoms:
[
  {
    "date_recorded": "2024-01-15",
    "duration_days": 3,
    "severity": "moderate",
    "description": "runny nose with clear discharge"
  },
  {
    "date_recorded": "2024-01-15",
    "duration_days": 2,
    "severity": "mild",
    "description": "low-grade fever around 100°F"
  }
]
"""


def _make_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
    )


llm = ChatOllama(
    base_url="http://10.230.100.240:17020/",
    model="gpt-oss:20b",
    temperature=0.3
)


def consulting_agent_node(state: AgentState) -> AgentState:
    turn_count = state.get("turn_count", 0) + 1
    patient_profile_adapter = TypeAdapter(PatientProfile)

    raw_profile = state.get("patient_profile")
    if raw_profile:
        try:
            # Handle both dict and PatientProfile instances
            if isinstance(raw_profile, PatientProfile):
                patient_profile = raw_profile
            else:
                patient_profile = patient_profile_adapter.validate_python(raw_profile)
        except Exception as e:
            print(f"Profile validation error: {e}")
            # Create default profile if validation fails
            patient_profile = PatientProfile(id="", name="", sex="", age=0, symptoms=[])
    else:
        patient_profile = PatientProfile(id="", name="", sex="", age=0, symptoms=[])

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

    # Serialize patient_profile to dict for LLM
    profile_dict = patient_profile.model_dump()
    
    model_messages: List[BaseMessage] = [
        SystemMessage(content=consulting_system_prompt),
        SystemMessage(content=f"Current date: {date.today()}"),
        SystemMessage(content=f"Original user query:\n{user_query}"),
        SystemMessage(content=f"Current patient profile:\n{json.dumps(profile_dict, indent=2)}"),
        SystemMessage(content=f"Retrieved MedlinePlus context:\n{retrieved_context}"),
    ]
    model_messages.extend(messages)
    model_messages.append(
        HumanMessage(content="Produce ONLY the required JSON object with properly formatted symptoms. No markdown, no extra text.")
    )

    result = llm.invoke(model_messages)
    raw_text = result.content.strip()
    
    # Clean up potential markdown formatting
    if raw_text.startswith("```json"):
        raw_text = raw_text[7:]
    if raw_text.startswith("```"):
        raw_text = raw_text[3:]
    if raw_text.endswith("```"):
        raw_text = raw_text[:-3]
    raw_text = raw_text.strip()

    try:
        parsed = json.loads(raw_text)
    except Exception as e:
        print(f"JSON parse error: {e}\nRaw text: {raw_text}")
        parsed = {
            "followup_question": None,
            "explanation": (
                "I'm sorry, I had trouble formatting my response correctly. "
                "Given what you've shared, it may be best to hand you over "
                "to the next part of the system so you can still get help."
            ),
            "updated_patient_profile": profile_dict,
            "need_more_info": False,
            "red_flag": False,
            "handoff_reason": "json_parse_error",
        }

    # Parse updated patient profile
    updated_profile_data = parsed.get("updated_patient_profile")
    if updated_profile_data:
        try:
            patient_profile = patient_profile_adapter.validate_python(updated_profile_data)
        except Exception as e:
            print(f"Updated profile validation error: {e}\nData: {updated_profile_data}")
            # Keep the old profile if new one fails validation
            pass

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
    checkpointer = InMemorySaver()
    return workflow.compile(checkpointer=checkpointer)