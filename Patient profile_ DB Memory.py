import os
import sqlite3
import json
import re
import logging
from typing import TypedDict, List, Any, Literal, Optional
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel, Field

# --------------------------------------------------------------------
# ENV + GLOBALS
# --------------------------------------------------------------------
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = Path(__file__).with_name("patients.db")

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------
# LLM (Ollama) – used for BOTH consulting + personal care
# --------------------------------------------------------------------
llm = ChatOllama(
    base_url="http://10.230.100.240:17020/",
    model="gpt-oss:20b",
    temperature=0.3,
)

# Medline retriever (your existing module)
from agents.medline_retriever import retrieve_passages

# --------------------------------------------------------------------
# DIALOG CONTROL
# --------------------------------------------------------------------
MAX_FOLLOWUP_TURNS = 4  # max number of consulting follow-up turns

# --------------------------------------------------------------------
# GLOBAL MEMORY + DB HANDLES
# --------------------------------------------------------------------
GLOBAL_STORE: Optional[InMemoryStore] = None
GLOBAL_DB: Optional[sqlite3.Connection] = None


@dataclass
class Context:
    """Reserved for future extension (if you later want to pass context)."""
    user_id: str
    store: InMemoryStore


# --------------------------------------------------------------------
# PATIENT + HISTORY MODELS
# --------------------------------------------------------------------
class PatientSymptom(BaseModel):
    description: str
    onset: Optional[str] = None
    severity: Optional[str] = None


class PatientProfile(BaseModel):
    id: str
    name: str
    sex: Optional[str] = None
    age: Optional[int] = None
    symptoms: List[PatientSymptom] = Field(default_factory=list)


class Recommendation(BaseModel):
    date: str
    summary: str
    symptoms_at_time: List[str] = Field(default_factory=list)


class PatientHistory(BaseModel):
    profile: PatientProfile
    recommendations: List[Recommendation] = Field(default_factory=list)
    created_at: str
    last_updated: str


# --------------------------------------------------------------------
# STATE DEFINITIONS
# --------------------------------------------------------------------
class RetrievalState(TypedDict, total=False):
    passages: List[str]
    urls: List[str]


class ConsultState(TypedDict, total=False):
    questions: List[str]
    answers: List[str]
    needs_more: bool
    red_flag: bool


class CareState(TypedDict, total=False):
    plan: str
    warnings: List[str]
    care_details: dict


class PipelineState(RetrievalState, ConsultState, CareState):
    user_query: str
    history: List[str]
    final_output: str
    patient_profile: Optional[dict]
    patient_history: Optional[dict]
    patient_id: str
    messages: list
    turn_count: int
    last_question: str
    profile_complete: bool
    is_returning_patient: bool
    patient_lookup_attempted: bool


# --------------------------------------------------------------------
# DB UTILITIES
# --------------------------------------------------------------------
def init_db(db_path: Path) -> sqlite3.Connection:
    """Initialize SQLite DB and return connection."""
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS patients (
            id TEXT PRIMARY KEY,
            name TEXT,
            sex TEXT,
            age INTEGER,
            profile_json TEXT,
            history_json TEXT,
            created_at TEXT,
            last_updated TEXT
        )
        """
    )
    conn.commit()
    return conn


def db_get_patient(patient_id: str) -> Optional[PatientHistory]:
    global GLOBAL_DB
    if not GLOBAL_DB:
        return None

    try:
        cur = GLOBAL_DB.cursor()
        cur.execute("SELECT history_json FROM patients WHERE id = ?", (patient_id,))
        row = cur.fetchone()
        if not row:
            return None

        history_json = row[0]
        if not history_json:
            return None

        data = json.loads(history_json)
        history = PatientHistory.model_validate(data)
        return history
    except Exception as e:
        print(f"Error reading patient from DB: {e}")
        return None


def db_save_patient_history(history: PatientHistory):
    global GLOBAL_DB
    if not GLOBAL_DB:
        return

    try:
        profile = history.profile
        now = datetime.now().isoformat()
        history.last_updated = now

        profile_json = json.dumps(profile.model_dump())
        history_json = json.dumps(history.model_dump())

        GLOBAL_DB.execute(
            """
            INSERT INTO patients (id, name, sex, age, profile_json, history_json, created_at, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name=excluded.name,
                sex=excluded.sex,
                age=excluded.age,
                profile_json=excluded.profile_json,
                history_json=excluded.history_json,
                last_updated=excluded.last_updated
            """,
            (
                profile.id,
                profile.name,
                profile.sex,
                profile.age,
                profile_json,
                history_json,
                history.created_at,
                history.last_updated,
            ),
        )
        GLOBAL_DB.commit()
        print(f"✓ Patient {profile.id} saved/updated in DB.")
    except Exception as e:
        print(f"Error saving patient to DB: {e}")


# --------------------------------------------------------------------
# MEMORY HELPERS (Store + DB)
# --------------------------------------------------------------------
def lookup_patient_in_store(patient_id: str, store: InMemoryStore) -> Optional[PatientHistory]:
    if not patient_id:
        return None

    namespace = ("PatientDetails",)

    # 1) Try in-memory store
    try:
        patient_data = store.get(namespace, patient_id)
        if patient_data and patient_data.value:
            history = PatientHistory.model_validate(patient_data.value)
            return history
    except Exception as e:
        print(f"Error looking up patient {patient_id} in store: {e}")

    # 2) Try SQLite DB
    history = db_get_patient(patient_id)
    if history:
        try:
            store.put(namespace, patient_id, history.model_dump())
            print(f"✓ Patient {patient_id} loaded from DB into memory store.")
        except Exception as e:
            print(f"Error caching patient {patient_id} into store: {e}")
        return history

    return None


def save_new_patient_to_store(patient_profile: PatientProfile, store: InMemoryStore) -> bool:
    namespace = ("PatientDetails",)

    try:
        now = datetime.now().isoformat()
        history = PatientHistory(
            profile=patient_profile,
            recommendations=[],
            created_at=now,
            last_updated=now,
        )

        store.put(namespace, patient_profile.id, history.model_dump())
        print(f"✓ Saved new patient {patient_profile.id} to memory store")

        db_save_patient_history(history)
        return True
    except Exception as e:
        print(f"Error saving patient: {e}")
        return False


def update_patient_history_with_plan(state: PipelineState):
    """After generating care plan, append to PatientHistory and save."""
    global GLOBAL_STORE
    if not GLOBAL_STORE:
        return

    patient_id = state.get("patient_id", "").strip()
    profile_dict = state.get("patient_profile") or {}
    plan_text = state.get("plan") or ""
    if not patient_id or not plan_text:
        return

    namespace = ("PatientDetails",)

    # Load existing or create new history
    history = lookup_patient_in_store(patient_id, GLOBAL_STORE)
    if history is None:
        try:
            profile = PatientProfile(**profile_dict)
        except Exception:
            profile = PatientProfile(
                id=patient_id,
                name=profile_dict.get("name", "Unknown"),
                sex=profile_dict.get("sex"),
                age=profile_dict.get("age") or 0,
                symptoms=[],
            )
        now = datetime.now().isoformat()
        history = PatientHistory(
            profile=profile,
            recommendations=[],
            created_at=now,
            last_updated=now,
        )

    # Build snapshot of symptoms
    symptoms_list: List[str] = []
    for s in (profile_dict.get("symptoms") or []):
        desc = s.get("description") or ""
        sev = s.get("severity") or ""
        onset = s.get("onset") or ""
        text = desc
        parts = []
        if onset:
            parts.append(f"onset: {onset}")
        if sev:
            parts.append(f"severity: {sev}")
        if parts:
            text += " (" + ", ".join(parts) + ")"
        if text:
            symptoms_list.append(text)

    rec = Recommendation(
        date=datetime.now().isoformat(),
        summary=plan_text[:4000],
        symptoms_at_time=symptoms_list,
    )
    history.recommendations.append(rec)

    GLOBAL_STORE.put(namespace, patient_id, history.model_dump())
    db_save_patient_history(history)


def generate_patient_id(name: str) -> str:
    name_part = "".join(name.split()[:2])[:6].upper().replace(" ", "")
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    return f"P-{name_part}-{timestamp}"


# --------------------------------------------------------------------
# PARSE CARE PLAN SECTIONS
# --------------------------------------------------------------------
def _parse_care_output(text: str) -> dict:
    sections = {
        "possible_conditions": [],
        "lifestyle_recommendations": [],
        "treatments": [],
        "next_steps": [],
        "references": [],
    }

    if not text:
        return sections

    t = text.replace("\r", "")

    patterns = {
        "possible_conditions": r"POSSIBLE CONDITIONS:(.*?)(?=\n[A-Z ]+:|$)",
        "lifestyle_recommendations": r"LIFESTYLE RECOMMENDATIONS:(.*?)(?=\n[A-Z ]+:|$)",
        "treatments": r"POSSIBLE TREATMENTS:(.*?)(?=\n[A-Z ]+:|$)",
        "next_steps": r"NEXT STEPS:(.*?)(?=\n[A-Z ]+:|$)",
        "references": r"REFERENCES:(.*?)(?=\n[A-Z ]+:|$)",
    }

    for key, pat in patterns.items():
        m = re.search(pat, t, flags=re.IGNORECASE | re.DOTALL)
        if not m:
            continue
        block = m.group(1).strip()
        if not block:
            continue

        lines = [
            ln.strip("•- \t")
            for ln in block.splitlines()
            if ln.strip()
        ]
        sections[key] = lines

    return sections


# --------------------------------------------------------------------
# NODES
# --------------------------------------------------------------------
def patient_intake_node(state: PipelineState) -> PipelineState:
    logger.info("Patient Intake node running...")

    global GLOBAL_STORE
    if not GLOBAL_STORE:
        raise ValueError("Global store not initialized")

    if state.get("profile_complete"):
        logger.info("Profile already complete, skipping intake")
        return state

    patient_id = state.get("patient_id", "").strip()

    if patient_id:
        print(f"🔍 Looking up patient ID: {patient_id}...")
        patient_hist = lookup_patient_in_store(patient_id, GLOBAL_STORE)

        if patient_hist:
            print(f"\n✓ Welcome back, {patient_hist.profile.name}!")
            print(f"   Age: {patient_hist.profile.age}, Sex: {patient_hist.profile.sex}")
            print(f"   Previous visits: {len(patient_hist.recommendations)}")

            if patient_hist.recommendations:
                last_rec = patient_hist.recommendations[-1]
                print(f"   Last visit: {last_rec.date}")
                if last_rec.symptoms_at_time:
                    print(f"   Previous symptoms: {', '.join(last_rec.symptoms_at_time[:3])}")

            state["patient_profile"] = patient_hist.profile.model_dump()
            state["patient_history"] = patient_hist.model_dump()
            state["is_returning_patient"] = True
            state["profile_complete"] = True
            state["patient_lookup_attempted"] = True
            return state
        else:
            print(f"\n✗ Patient ID {patient_id} not found in system.")
            create_new = input("Would you like to create a new profile? (yes/no): ").strip().lower()
            if create_new != "yes":
                print("Cannot proceed without patient profile. Exiting.")
                state["profile_complete"] = False
                return state

    # Create new profile
    print("\n" + "=" * 60)
    print("CREATING NEW PATIENT PROFILE")
    print("=" * 60)

    patient_name = input("Enter your full name: ").strip()
    if not patient_name:
        patient_name = "Unknown Patient"

    sex = input("Enter your sex (Male/Female/Other): ").strip()
    age_str = input("Enter your age: ").strip()

    try:
        age = int(age_str)
    except ValueError:
        age = 0
        print("Invalid age, setting to 0")

    new_patient_id = generate_patient_id(patient_name)
    print(f"\n🆔 Generated Patient ID: {new_patient_id}")
    print("⚠️  Please save this ID for future visits!")

    new_profile = PatientProfile(
        id=new_patient_id,
        name=patient_name,
        sex=sex,
        age=age,
        symptoms=[],
    )

    save_success = save_new_patient_to_store(new_profile, GLOBAL_STORE)

    if save_success:
        print(f"\n✓ Profile created successfully for {patient_name}")
        state["patient_profile"] = new_profile.model_dump()
        state["patient_history"] = None
        state["patient_id"] = new_patient_id
        state["is_returning_patient"] = False
        state["profile_complete"] = True
        state["patient_lookup_attempted"] = True
    else:
        print("\n✗ Failed to save profile")
        state["profile_complete"] = False

    print("=" * 60 + "\n")
    return state


def route_from_intake(state: PipelineState) -> Literal["retrieve", END]:
    if state.get("profile_complete"):
        logger.info("✓ Profile complete, proceeding to retrieval")
        return "retrieve"
    else:
        logger.error("✗ Profile incomplete, ending workflow")
        return END


def retriever_agent(state: PipelineState) -> PipelineState:
    logger.info("Retriever agent running with MedlinePlus retrieval...")

    query = state["user_query"]
    profile_dict = state.get("patient_profile", {})

    if profile_dict and profile_dict.get("symptoms"):
        symptom_strings = [s.get("description", "") for s in profile_dict["symptoms"]]
        query += " " + " ".join(symptom_strings)

    results = retrieve_passages(query, top_k=6)

    state["passages"] = [r["text"] for r in results]
    state["urls"] = [r["url"] for r in results]

    return state


def consulting_agent(state: PipelineState) -> PipelineState:
    logger.info("Consulting agent running with Ollama...")

    user_query = state["user_query"]
    retrieved_docs = state.get("passages", [])
    urls = state.get("urls", [])
    profile_dict = state.get("patient_profile", {}) or {}
    last_question = state.get("last_question", "")
    turn_count = state.get("turn_count", 0)

    doc_summaries = []
    for i, (p, u) in enumerate(zip(retrieved_docs, urls)):
        if i >= 4:
            break
        doc_summaries.append(f"- [{i+1}] {p[:400]} (source: {u})")

    docs_text = "\n".join(doc_summaries) if doc_summaries else "No external references available."

    prompt = f"""
You are a careful, conservative healthcare intake assistant (NOT a doctor).

Your job:
1. Explain to the patient what might be going on, in simple language.
2. Decide if you need MORE information (ask ONE clear follow-up question).
3. Flag any red-flag / emergency signs if present.
4. Optionally refine/update the patient profile (e.g., add structured symptoms).

ALWAYS respond in *pure JSON* with exactly:

{{
  "explanation": "string",
  "followup_question": "string or null",
  "need_more_info": true or false,
  "red_flag": true or false,
  "updated_profile": {{
      "id": "string or same as before",
      "name": "string",
      "sex": "string or null",
      "age": integer or null,
      "symptoms": [
          {{
              "description": "string",
              "onset": "string or null",
              "severity": "string or null"
          }}
      ]
  }}
}}

If you do NOT need more info:
  "followup_question": null
  "need_more_info": false

-----------------------
PATIENT QUERY:
{user_query}

LAST QUESTION YOU ASKED (if any):
{last_question}

TURN COUNT:
{turn_count}

CURRENT PATIENT PROFILE (JSON):
{json.dumps(profile_dict, indent=2)}

RETRIEVED REFERENCE SNIPPETS (MedlinePlus-style):
{docs_text}
"""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        raw_text = response.content if isinstance(response, AIMessage) else str(response)
    except Exception as e:
        logger.error(f"Error calling Ollama in consulting_agent: {e}")
        explanation = (
            "I'm sorry, I'm having trouble connecting right now.\n"
            "Please monitor your symptoms and contact a healthcare professional, "
            "especially if they worsen or you develop new symptoms."
        )
        state["questions"] = []
        state["answers"] = [explanation]
        msgs = state.get("messages", [])
        msgs.append(AIMessage(content=explanation))
        state["messages"] = msgs
        state["turn_count"] = turn_count + 1
        state["needs_more"] = False
        state["last_question"] = last_question
        state["red_flag"] = False
        return state

    data: dict[str, Any] = {}
    try:
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
        else:
            data = json.loads(raw_text)
    except Exception:
        data = {
            "explanation": raw_text,
            "followup_question": None,
            "need_more_info": False,
            "red_flag": False,
            "updated_profile": profile_dict,
        }

    explanation = data.get("explanation", raw_text)
    followup_q = data.get("followup_question")
    need_more_info = bool(data.get("need_more_info", bool(followup_q)))
    red_flag = bool(data.get("red_flag", False))
    updated_profile_data = data.get("updated_profile") or profile_dict

    # Stop repeating same question
    if followup_q:
        normalized_last = (last_question or "").strip().lower()
        normalized_new = followup_q.strip().lower()
        if normalized_last and normalized_new and normalized_last == normalized_new:
            logger.info("Follow-up question is the same as last one → stop follow-ups.")
            followup_q = None
            need_more_info = False

    # Stop after MAX_FOLLOWUP_TURNS
    if turn_count >= MAX_FOLLOWUP_TURNS:
        logger.info(f"Reached MAX_FOLLOWUP_TURNS={MAX_FOLLOWUP_TURNS} → move to care plan.")
        followup_q = None
        need_more_info = False

    if not followup_q:
        need_more_info = False

    # Update profile
    try:
        patient_profile = PatientProfile(**updated_profile_data)
        state["patient_profile"] = patient_profile.model_dump()
    except Exception:
        state["patient_profile"] = updated_profile_data

    state["questions"] = [followup_q] if followup_q else []
    state["answers"] = [explanation] if explanation else []

    msgs = state.get("messages", [])
    msgs.append(AIMessage(content=explanation))
    state["messages"] = msgs

    state["turn_count"] = turn_count + 1
    state["needs_more"] = need_more_info
    state["last_question"] = followup_q or last_question
    state["red_flag"] = red_flag

    return state


def followup_node(state: PipelineState) -> PipelineState:
    logger.info("Follow-up node: waiting for user answer...")
    return state


def personal_care_agent(state: PipelineState) -> PipelineState:
    """
    Personal care plan generator using OLLAMA ONLY
    (no OpenAI, no create_personal_care_agent → no 429 errors).
    """
    logger.info("Personal Care agent (Ollama-only) running...")

    profile_dict = state.get("patient_profile", {}) or {}
    user_query = state.get("user_query", "")
    passages = state.get("passages", [])
    urls = state.get("urls", []) or []

    refs_text = ""
    if passages:
        refs_text = "\n".join(f"- {p}" for p in passages[:6])

    url_list_text = ""
    if urls:
        url_list_text = "\n".join(f"- {u}" for u in urls if u)

    prompt = f"""
You are a PERSONAL HEALTH COACH (NOT a doctor).

Using:
- The original concern
- The structured patient profile
- The Medline-like reference snippets
- The list of URLs

Create a CLEAR, STRUCTURED CARE PLAN with EXACTLY these SECTIONS
(in this order, with these headings in ALL CAPS):

POSSIBLE CONDITIONS:
- ...
- ...

LIFESTYLE RECOMMENDATIONS:
- ...

POSSIBLE TREATMENTS:
- ...

NEXT STEPS:
- ...

REFERENCES:
- ...

Rules:
- Use short bullet points starting with "- " under each section.
- Be conservative, advise to see a doctor for diagnosis.
- In REFERENCES, use the URLs provided below (if any) as bullets.
- Do NOT add markdown, just plain text sections as shown.

-----------------------
ORIGINAL CONCERN:
{user_query}

PATIENT PROFILE (JSON):
{json.dumps(profile_dict, indent=2)}

REFERENCE SNIPPETS:
{refs_text}

REFERENCE URLS:
{url_list_text}
"""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        care_text = response.content if isinstance(response, AIMessage) else str(response)
    except Exception as e:
        logger.error(f"Personal care (Ollama) failed: {e}")
        care_text = (
            "POSSIBLE CONDITIONS:\n"
            "- I had trouble generating a detailed analysis.\n\n"
            "LIFESTYLE RECOMMENDATIONS:\n"
            "- Maintain a healthy diet, gentle physical activity, and adequate sleep.\n\n"
            "POSSIBLE TREATMENTS:\n"
            "- Please consult a licensed healthcare provider for personalized advice.\n\n"
            "NEXT STEPS:\n"
            "- If symptoms worsen or you develop severe pain, chest pain, trouble breathing, "
            "or other alarming signs, seek urgent medical care.\n\n"
            "REFERENCES:\n"
            "- Local healthcare providers or clinics.\n"
        )

    care_details = _parse_care_output(care_text)

    state["plan"] = care_text
    state["warnings"] = [
        "This information is not a substitute for professional medical advice.",
        "Seek urgent help if you experience severe pain, difficulty breathing, chest pain, or persistent/worsening symptoms.",
    ]
    state["care_details"] = care_details

    return state


def planner_summary_node(state: PipelineState) -> PipelineState:
    logger.info("Planner summary node running...")
    summary = (
        f"Query: {state['user_query']}\n"
        f"Questions: {state.get('questions')}\n"
        f"Answers: {state.get('answers')}\n"
        f"Plan: {state.get('plan')}\n"
    )
    history = state.get("history", []) or []
    history.append(summary)
    state["history"] = history

    # Persist plan into patient history
    update_patient_history_with_plan(state)

    state["final_output"] = state.get("plan") or "[No care plan was generated.]"
    return state


def route_from_consult(state: PipelineState) -> Literal["followup", "care"]:
    if state.get("needs_more"):
        return "followup"
    else:
        return "care"


# --------------------------------------------------------------------
# BUILD GRAPH
# --------------------------------------------------------------------
def build_planner_graph():
    workflow = StateGraph(PipelineState)

    workflow.add_node("patient_intake", patient_intake_node)
    workflow.add_node("retrieve", retriever_agent)
    workflow.add_node("consult", consulting_agent)
    workflow.add_node("followup", followup_node)
    workflow.add_node("care", personal_care_agent)
    workflow.add_node("planner_summary", planner_summary_node)

    workflow.set_entry_point("patient_intake")

    workflow.add_conditional_edges(
        "patient_intake",
        route_from_intake,
        {
            "retrieve": "retrieve",
            END: END,
        },
    )

    workflow.add_edge("retrieve", "consult")

    workflow.add_conditional_edges(
        "consult",
        route_from_consult,
        {
            "followup": "followup",
            "care": "care",
        },
    )

    workflow.add_edge("followup", END)
    workflow.add_edge("care", "planner_summary")
    workflow.add_edge("planner_summary", END)

    checkpointer = InMemorySaver()
    return workflow.compile(checkpointer=checkpointer)


# --------------------------------------------------------------------
# CONSOLE WRAPPER
# --------------------------------------------------------------------
class PlannerAgent:
    def __init__(self, graph):
        self.graph = graph
        self.state: Optional[PipelineState] = None
        self.thread_config = {"configurable": {"thread_id": "main_session"}}

    def start(self, patient_id: str, query: str):
        self.state = {
            "user_query": query,
            "history": [],
            "final_output": "",
            "patient_profile": None,
            "patient_history": None,
            "patient_id": patient_id,
            "messages": [],
            "turn_count": 0,
            "last_question": "",
            "profile_complete": False,
            "is_returning_patient": False,
            "patient_lookup_attempted": False,
        }

        out = self.graph.invoke(self.state, self.thread_config)
        self.state = out
        return self._analyze(self.state)

    def continue_with_answer(self, answer: str):
        if not self.state:
            return {"type": "final", "output": "[No active conversation]"}

        msgs = self.state.get("messages", [])
        msgs.append(HumanMessage(content=answer))
        self.state["messages"] = msgs
        self.state["user_query"] = answer

        out = self.graph.invoke(self.state, self.thread_config)
        self.state = out

        return self._analyze(self.state)

    def _analyze(self, state: PipelineState):
        if state.get("final_output"):
            return {"type": "final", "output": state["final_output"]}

        questions = state.get("questions") or []
        answers = state.get("answers") or []

        if questions:
            explanation = answers[0] if answers else ""
            return {
                "type": "followup",
                "question": questions[0],
                "explanation": explanation,
            }

        return {"type": "final", "output": "[No further questions or output.]"}


# --------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🔵 PERSONAL HEALTH COACH — Multi-Agent System")
    print("=" * 70 + "\n")

    print("Initializing memory store (no embedding index)...")
    GLOBAL_STORE = InMemoryStore()
    print("✓ Memory store initialized (key–value only)")

    print("Initializing SQLite DB...")
    GLOBAL_DB = init_db(DB_PATH)
    print(f"✓ DB initialized at {DB_PATH}\n")

    planner_graph = build_planner_graph()
    planner = PlannerAgent(planner_graph)

    print("=" * 70)
    print("PATIENT IDENTIFICATION")
    print("=" * 70)
    patient_id = input("\nEnter your Patient ID (or press Enter if you're new): ").strip()

    print("\n" + "=" * 70)
    print("HEALTH CONCERN")
    print("=" * 70)
    user_input = input("\nWhat health concern would you like to discuss today?\nUser: ")

    step = planner.start(patient_id, user_input)

    while True:
        if step["type"] == "final":
            print("\n" + "=" * 70)
            print("HEALTH RECOMMENDATIONS")
            print("=" * 70)
            print(step["output"])
            print("=" * 70)
            break

        elif step["type"] == "followup":
            print("\n" + "-" * 70)
            print("CONSULTING AGENT:")
            print("-" * 70)
            if step.get("explanation"):
                print(f"\n{step['explanation']}")
            print(f"\nFollow-up question: {step['question']}")
            print("-" * 70)

            answer = input("\nYour answer: ")
            step = planner.continue_with_answer(answer)
            

    print("\n" + "=" * 70)
    print("Thank you for using the Personal Health Coach! Take care! 💙")
    print("=" * 70 + "\n")

    if GLOBAL_DB:
        GLOBAL_DB.close()

