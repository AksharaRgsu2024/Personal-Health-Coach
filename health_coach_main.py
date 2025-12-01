from pathlib import Path
import os
from vector_db import semantic_search
from agents.consulting_agent_with_memory import build_consulting_agent_graph, PatientProfile
from agents.personal_care_agent_lt_memory import PatientRecommendation, PatientHistory, create_personal_care_agent, PatientHistory, run_personal_care_agent
import sqlite3
import json
import re
import logging
from typing import TypedDict, List, Any, Literal, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from dataclasses import dataclass
from langgraph.store.memory import InMemoryStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
from datetime import datetime
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
import pickle

# --------------------------------------------------------------------
# ENV + GLOBALS
# --------------------------------------------------------------------
load_dotenv()

DB_PATH = "data/patients.db"

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Global store instance (not in state to avoid serialization issues)
GLOBAL_STORE: Optional[InMemoryStore] = None
GLOBAL_DB: Optional[sqlite3.Connection] = None

# Connect to an SQLite database (creates the file if it doesn't exist)
conn = sqlite3.connect("langgraph_memory.db")
memory = SqliteSaver(conn)

@dataclass
class Context:
    user_id: str
    store: InMemoryStore

# Configure LLM
llm = ChatOllama(
    base_url="http://10.230.100.240:17020/",
    model="gpt-oss:20b",
    temperature=0.3
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

# ----------------- State definitions -----------------
class RetrievalState(TypedDict, total=False):
    passages: List[str]
    urls: List[str]

class ConsultState(TypedDict, total=False):
    questions: List[str]
    answers: List[str]
    needs_more: bool

class CareState(TypedDict, total=False):
    plan: str
    warnings: List[str]
    care_details: dict

class PipelineState(RetrievalState, ConsultState, CareState):
    user_query: str
    history: List[str]
    final_output: str
    patient_profile: Optional[dict]  # Use dict for serialization
    patient_history: Optional[dict]  # Use dict for serialization
    patient_id: str  # Store patient ID separately
    messages: list
    turn_count: int
    last_question: str
    profile_complete: bool
    is_returning_patient: bool
    patient_lookup_attempted: bool

# --------------------------------------------------------------------
# DB UTILITIES WITH MEMORY STORE PERSISTENCE
# --------------------------------------------------------------------
def init_db(db_path: Path) -> sqlite3.Connection:
    """Initialize SQLite DB with memory_store table only."""
    conn = sqlite3.connect(db_path)
    
    # Single unified memory_store table for LangGraph store persistence
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_store (
            namespace TEXT NOT NULL,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (namespace, key)
        )
        """
    )
    
    # Index for faster lookups
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_memory_store_namespace 
        ON memory_store(namespace)
        """
    )
    
    conn.commit()
    return conn


def save_memory_store_to_db(store: InMemoryStore, conn: sqlite3.Connection):
    """
    Persist the entire memory store to SQLite.
    Optimized to handle patient history data structure from personal_care_agent.
    """
    try:
        cursor = conn.cursor()
        now = datetime.now().isoformat()
        
        # Define the namespace for patient details
        namespace = ("PatientDetails",)
        
        # Search for all items in the PatientDetails namespace
        try:
            results = store.search(namespace)
        except AttributeError:
            # If search method doesn't exist, try to access store data directly
            # This is a fallback for different InMemoryStore implementations
            logger.warning("Store.search() not available, attempting alternative method")
            results = []
            
            # Try to get all patient IDs from the namespace
            # This assumes you have a way to list all keys
            # If not, you may need to track patient IDs separately
        
        if not results:
            logger.info("No items found in memory store to persist")
            return
        
        saved_count = 0
        updated_count = 0
        
        for item in results:
            try:
                namespace_str = "/".join(item.namespace)
                patient_id = item.key
                
                # item.value should be a dict representing PatientHistory
                patient_data = item.value
                
                # Validate the data structure
                if not isinstance(patient_data, dict):
                    logger.warning(f"Invalid data type for patient {patient_id}: {type(patient_data)}")
                    continue
                
                # Ensure it has the expected structure
                if "profile" not in patient_data:
                    logger.warning(f"Missing profile for patient {patient_id}")
                    continue
                
                # Convert to JSON for storage
                value_json = json.dumps(patient_data)
                
                # Check if record exists
                cursor.execute(
                    "SELECT created_at FROM memory_store WHERE namespace = ? AND key = ?",
                    (namespace_str, patient_id)
                )
                existing = cursor.fetchone()
                
                if existing:
                    # Update existing record
                    cursor.execute(
                        """
                        UPDATE memory_store 
                        SET value = ?, updated_at = ?
                        WHERE namespace = ? AND key = ?
                        """,
                        (value_json, now, namespace_str, patient_id)
                    )
                    updated_count += 1
                    
                    # Log update details
                    profile = patient_data.get("profile", {})
                    recommendations = patient_data.get("recommendations", [])
                    logger.info(
                        f"  Updated: {profile.get('name', 'Unknown')} "
                        f"({patient_id}) - {len(recommendations)} recommendations"
                    )
                else:
                    # Insert new record
                    created_at = patient_data.get("created_at", now)
                    cursor.execute(
                        """
                        INSERT INTO memory_store (namespace, key, value, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (namespace_str, patient_id, value_json, created_at, now)
                    )
                    saved_count += 1
                    
                    # Log save details
                    profile = patient_data.get("profile", {})
                    logger.info(
                        f"  Saved: {profile.get('name', 'Unknown')} ({patient_id})"
                    )
                    
            except Exception as item_error:
                logger.error(f"Error processing item {getattr(item, 'key', 'unknown')}: {item_error}")
                continue
        
        conn.commit()
        
        total = saved_count + updated_count
        logger.info(
            f"✓ Memory store persisted to DB: "
            f"{saved_count} new, {updated_count} updated (total: {total})"
        )
        
    except Exception as e:
        logger.error(f"Error saving memory store to DB: {e}")
        import traceback
        traceback.print_exc()
        conn.rollback()


def load_memory_store_from_db(conn: sqlite3.Connection) -> InMemoryStore:
    """Load memory store from SQLite database."""
    try:
        # Initialize empty store
        store = InMemoryStore()
        
        cursor = conn.cursor()
        cursor.execute("SELECT namespace, key, value FROM memory_store")
        rows = cursor.fetchall()
        
        for namespace_str, key, value_json in rows:
            namespace = tuple(namespace_str.split("/"))
            value = json.loads(value_json)
            
            # Put the item back into the store
            store.put(namespace, key, value)
        
        logger.info(f"✓ Memory store loaded from DB ({len(rows)} items)")
        return store
        
    except Exception as e:
        logger.error(f"Error loading memory store from DB: {e}")
        return InMemoryStore()


def db_get_patient(patient_id: str) -> Optional[PatientHistory]:
    """Get patient from memory_store table via LangGraph store."""
    global GLOBAL_STORE
    if not GLOBAL_STORE:
        return None

    try:
        # Use the store lookup instead of direct DB query
        return lookup_patient_in_store(patient_id, GLOBAL_STORE)
    except Exception as e:
        logger.error(f"Error reading patient from store: {e}")
        return None


def db_save_patient_history(history: PatientHistory):
    """Save patient to memory_store table via LangGraph store."""
    global GLOBAL_STORE, GLOBAL_DB
    if not GLOBAL_STORE or not GLOBAL_DB:
        return

    try:
        namespace = ("PatientDetails",)
        profile = history.profile
        now = datetime.now().isoformat()
        history.last_updated = now

        # Save to memory store
        GLOBAL_STORE.put(namespace, profile.id, history.model_dump())
        
        # Persist to DB
        save_memory_store_to_db(GLOBAL_STORE, GLOBAL_DB)
        
        logger.info(f"✓ Patient {profile.id} saved/updated in memory_store.")
    except Exception as e:
        logger.error(f"Error saving patient to memory_store: {e}")

# --------------------------------------------------------------------
# MEMORY HELPERS (Store + DB)
# --------------------------------------------------------------------

def lookup_patient_in_store(patient_id: str, store: InMemoryStore) -> Optional[PatientHistory]:
    """Look up patient in the store by patient ID."""
    if not patient_id:
        return None
    
    namespace = ("PatientDetails",)
    
    try:
        patient_data = store.get(namespace, patient_id)
        if patient_data and patient_data.value:
            history = PatientHistory.model_validate(patient_data.value)
            return history
    except Exception as e:
        logger.error(f"Error looking up patient {patient_id}: {e}")
    
    return None


def save_new_patient_to_store(patient_profile: PatientProfile, store: InMemoryStore) -> bool:
    """Save a new patient profile to the store and DB."""
    global GLOBAL_DB
    namespace = ("PatientDetails",)
    
    try:
        now = datetime.now().isoformat()
        history = PatientHistory(
            profile=patient_profile,
            recommendations=[],
            created_at=now,
            last_updated=now
        )
        
        # Save to memory store
        store.put(namespace, patient_profile.id, history.model_dump())
        
        # Persist memory store to DB (single table)
        if GLOBAL_DB:
            save_memory_store_to_db(store, GLOBAL_DB)
        
        logger.info(f"✓ Saved new patient {patient_profile.id} to memory_store")
        return True
    except Exception as e:
        logger.error(f"Error saving patient: {e}")
        return False


def update_patient_history_with_plan(state: PipelineState):
    """
    Retrieve PatientHistory and PatientRecommendation from LangGraph store
    (created by personal_care_agent) and save to memory_store table.
    """
    global GLOBAL_STORE, GLOBAL_DB
    if not GLOBAL_STORE or not GLOBAL_DB:
        logger.error("Global store or DB not initialized")
        return

    patient_id = state.get("patient_id", "").strip()
    
    if not patient_id:
        logger.warning("No patient ID provided, cannot update history")
        return

    namespace = ("PatientDetails",)

    try:
        # Step 1: Retrieve patient history from LangGraph store
        # This should already contain the recommendation added by personal_care_agent
        logger.info(f"Retrieving patient {patient_id} from LangGraph store...")
        history = lookup_patient_in_store(patient_id, GLOBAL_STORE)
        
        if history is None:
            logger.error(f"Patient {patient_id} not found in store. personal_care_agent should have saved it.")
            return
        
        logger.info(f"✓ Retrieved patient {history.profile.name} with {len(history.recommendations)} recommendations")
        
        # Step 2: Verify the latest recommendation exists
        if not history.recommendations:
            logger.warning("No recommendations found in patient history")
            return
        
        latest_rec = history.recommendations[-1]
        logger.info(f"Latest recommendation date: {latest_rec.date}")
        logger.info(f"Symptoms at visit: {', '.join(latest_rec.symptoms_at_time[:3])}{'...' if len(latest_rec.symptoms_at_time) > 3 else ''}")
        
        # Step 3: Update the last_updated timestamp
        history.last_updated = datetime.now().isoformat()
        
        # Step 4: Save updated history back to LangGraph memory store
        logger.info("Saving to LangGraph memory store...")
        GLOBAL_STORE.put(namespace, patient_id, history.model_dump())
        logger.info("✓ Saved to memory store")
        
        # Step 5: Persist entire memory store to memory_store table
        logger.info("Persisting memory store to DB...")
        save_memory_store_to_db(GLOBAL_STORE, GLOBAL_DB)
        logger.info("✓ Memory store persisted to DB")
        
        # Step 6: Verify the save by reading back
        verification = lookup_patient_in_store(patient_id, GLOBAL_STORE)
        if verification and len(verification.recommendations) == len(history.recommendations):
            logger.info(f"✓ Verification successful: {len(verification.recommendations)} recommendations stored")
        else:
            logger.warning("⚠ Verification failed: data may not have been saved correctly")
        
        print(f"\n{'='*60}")
        print(f"✓ Patient history saved to database")
        print(f"  Patient: {history.profile.name} ({patient_id})")
        print(f"  Total visits: {len(history.recommendations)}")
        print(f"  Latest visit: {latest_rec.date}")
        print(f"{'='*60}\n")
        
    except Exception as e:
        logger.error(f"Error updating patient history: {e}")
        import traceback
        traceback.print_exc()
        
        # Attempt to rollback DB changes if error occurred
        if GLOBAL_DB:
            try:
                GLOBAL_DB.rollback()
                logger.info("Database rollback completed")
            except Exception as rollback_error:
                logger.error(f"Rollback failed: {rollback_error}")


def generate_patient_id(name: str) -> str:
    """Generate a unique patient ID based on name and timestamp."""
    name_part = ''.join(name.split()[:2])[:6].upper().replace(" ", "")
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    return f"P-{name_part}-{timestamp}"


def _parse_care_output(text: str) -> dict:
    """Parse the personal care agent output into sections."""
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


# ----------------- Nodes -----------------
def patient_intake_node(state: PipelineState) -> PipelineState:
    """Handle patient intake - lookup or create new profile."""
    logging.info("Patient Intake node running...")
    
    global GLOBAL_STORE
    if not GLOBAL_STORE:
        raise ValueError("Global store not initialized")
    
    # Check if patient is already loaded (for continuation)
    if state.get("profile_complete"):
        logging.info("Profile already complete, skipping intake")
        return state
    
    # Get patient_id from state (set during initialization)
    patient_id = state.get("patient_id", "").strip()
    
    if patient_id:
        # Try to look up existing patient
        print(f"🔍 Looking up patient ID: {patient_id}...")
        patient_hist = lookup_patient_in_store(patient_id, GLOBAL_STORE)
        
        if patient_hist:
            print(f"\n✓ Welcome back, {patient_hist.profile.name}!")
            print(f"   Age: {patient_hist.profile.age}, Sex: {patient_hist.profile.sex}")
            print(f"   Previous visits: {len(patient_hist.recommendations)}")
            
            if patient_hist.recommendations:
                last_rec = patient_hist.recommendations[-1]
                print(f"   Last visit: {last_rec.date}")
                print(f"   Previous symptoms: {', '.join(last_rec.symptoms_at_time[:3])}")
            
            # Convert to dict for state serialization
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
    
    # Create new patient profile
    print("\n" + "="*60)
    print("CREATING NEW PATIENT PROFILE")
    print("="*60)
    
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
    
    # Generate patient ID
    new_patient_id = generate_patient_id(patient_name)
    print(f"\n🆔 Generated Patient ID: {new_patient_id}")
    print("⚠️  Please save this ID for future visits!")
    
    # Create and save new profile
    new_profile = PatientProfile(
        id=new_patient_id,
        name=patient_name,
        sex=sex,
        age=age,
        symptoms=[]
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
    
    print("="*60 + "\n")
    return state


def route_from_intake(state: PipelineState) -> Literal["retrieve", END]:
    """Route after intake based on whether profile is complete."""
    if state.get("profile_complete"):
        logging.info("✓ Profile complete, proceeding to retrieval")
        return "retrieve"
    else:
        logging.error("✗ Profile incomplete, ending workflow")
        return END


def retriever_agent(state: PipelineState) -> PipelineState:
    logging.info("Retriever agent running with MedlinePlus embeddings...")

    query = state["user_query"]
    profile_dict = state.get("patient_profile", {})

    # Enrich query with symptoms if available
    if profile_dict and profile_dict.get("symptoms"):
        symptom_strings = [
            s.get('description', '') for s in profile_dict["symptoms"]
        ]
        query += " " + " ".join(symptom_strings)

    results = semantic_search(query)

    state["passages"] = [r.payload["text"] for r in results]
    state["urls"] = [r.payload["url"] for r in results]

    return state


def consulting_agent(state: PipelineState) -> PipelineState:
    logging.info("Consulting agent running...")

    # Build consulting agent graph (with memory)
    consult_graph = build_consulting_agent_graph()
    
    retrieved_docs = [
        {
            "title": p[:60],
            "summary": p,
            "url": u,
        }
        for p, u in zip(state.get("passages", []), state.get("urls", []))
    ]

    # Convert dict back to PatientProfile for consulting agent
    profile_dict = state.get("patient_profile", {})
    if profile_dict:
        # Ensure age field exists before creating PatientProfile
        if "age" not in profile_dict or profile_dict["age"] is None:
            logger.warning("Age missing from profile, setting default to 0")
            profile_dict["age"] = 0
        
        try:
            patient_profile = PatientProfile(**profile_dict)
        except Exception as e:
            logger.error(f"Error creating PatientProfile: {e}")
            logger.error(f"Profile data: {profile_dict}")
            # Create minimal valid profile
            patient_profile = PatientProfile(
                id=profile_dict.get("id", "unknown"),
                name=profile_dict.get("name", "Unknown"),
                sex=profile_dict.get("sex"),
                age=profile_dict.get("age", 0),
                symptoms=profile_dict.get("symptoms", [])
            )
    else:
        patient_profile = None

    agent_state = {
        "user_query": state["user_query"],
        "messages": state.get("messages", []),
        "retrieved_docs": retrieved_docs,
        "patient_profile": patient_profile,
        "turn_count": state.get("turn_count", 0),
        "need_more_info": True,
        "red_flag": False,
        "last_question": state.get("last_question"),
    }

    updated = consult_graph.invoke(agent_state)

    print("\n===== DEBUG CONSULT OUTPUT =====")
    print(f"Need more info: {updated.get('need_more_info')}")
    print(f"Turn count: {updated.get('turn_count')}")
    print("================================\n")

    assistant_out = updated.get("assistant_output", {}) or {}
    followup_q = assistant_out.get("followup_question")
    explanation = assistant_out.get("explanation", "")

    # Convert updated profile back to dict, preserving all fields including age
    updated_profile = updated.get("patient_profile")
    if updated_profile:
        if isinstance(updated_profile, PatientProfile):
            # Use model_dump to get all fields
            updated_dict = updated_profile.model_dump()
            
            # Preserve age from original if missing in update
            if ("age" not in updated_dict or updated_dict["age"] is None) and profile_dict:
                original_age = profile_dict.get("age", 0)
                logger.warning(f"Age missing in updated profile, preserving original: {original_age}")
                updated_dict["age"] = original_age
            
            state["patient_profile"] = updated_dict
        else:
            # It's already a dict
            # Ensure age exists
            if "age" not in updated_profile or updated_profile["age"] is None:
                if profile_dict:
                    updated_profile["age"] = profile_dict.get("age", 0)
                else:
                    updated_profile["age"] = 0
                logger.warning(f"Age missing in updated profile dict, set to: {updated_profile['age']}")
            
            state["patient_profile"] = updated_profile

    state["questions"] = [followup_q] if followup_q else []
    state["answers"] = [explanation] if explanation else []
    state["messages"] = updated.get("messages", [])
    state["turn_count"] = updated.get("turn_count", 1)
    state["needs_more"] = bool(updated.get("need_more_info"))
    state["last_question"] = updated.get("last_question")

    return state


def followup_node(state: PipelineState) -> PipelineState:
    logging.info("Follow-up node: waiting for user answer...")
    return state


def personal_care_agent(state: PipelineState) -> PipelineState:
    logging.info("Personal Care agent running with KG + MedlinePlus...")

    global GLOBAL_STORE
    
    profile_dict = state.get("patient_profile", {})
    user_query = state.get("user_query", "")
    passages = state.get("passages", [])

    try:
        # Use the new run function
        care_text = run_personal_care_agent(
            patient_profile=profile_dict,
            user_query=user_query,
            passages=passages,
            store=GLOBAL_STORE
        )
        
    except Exception as e:
        logging.error(f"Personal care agent failed: {e}")
        import traceback
        traceback.print_exc()
        
        care_text = (
            "Sorry, I had trouble generating a detailed care plan right now. "
            "Please consider speaking directly with a healthcare professional."
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
    logging.info("Planner summary node running...")
    summary = (
        f"Query: {state['user_query']}\n"
        f"Questions: {state.get('questions')}\n"
        f"Answers: {state.get('answers')}\n"
        f"Plan: {state.get('plan')}\n"
    )
    history = state.get("history", [])
    history.append(summary)
    state["history"] = history

    state["final_output"] = state.get("plan") or "[No care plan was generated.]"
    
    # Save memory store after generating plan
    update_patient_history_with_plan(state)
    
    return state


def route_from_consult(state: PipelineState) -> Literal["followup", "care"]:
    if state.get("needs_more"):
        return "followup"
    else:
        return "care"


# ----------------- Build the graph -----------------
def build_planner_graph():
    """Build the complete multi-agent workflow."""
    workflow = StateGraph(PipelineState)
    
    workflow.add_node("patient_intake", patient_intake_node)
    workflow.add_node("retrieve", retriever_agent)
    workflow.add_node("consult", consulting_agent)
    workflow.add_node("followup", followup_node)
    workflow.add_node("care", personal_care_agent)
    workflow.add_node("planner_summary", planner_summary_node)

    workflow.set_entry_point("patient_intake")
    
    # Route from intake based on profile completion
    workflow.add_conditional_edges(
        "patient_intake",
        route_from_intake,
        {
            "retrieve": "retrieve",
            END: END
        }
    )
    
    workflow.add_edge("retrieve", "consult")

    workflow.add_conditional_edges(
        "consult",
        route_from_consult,
        {
            "followup": "followup",
            "care": "care",
        }
    )

    workflow.add_edge("followup", END)
    workflow.add_edge("care", "planner_summary")
    workflow.add_edge("planner_summary", END)

    checkpointer = InMemorySaver()
    return workflow.compile(checkpointer=checkpointer)


# ----------------- Console wrapper -----------------
class PlannerAgent:
    def __init__(self, graph):
        self.graph = graph
        self.state: Optional[PipelineState] = None
        self.thread_config = {"configurable": {"thread_id": "main_session"}}
                            
    def start(self, patient_id: str, query: str):
        """Start a new conversation with patient ID."""
        self.state = {
            "user_query": query,
            "history": [],
            "final_output": "",
            "patient_profile": None,
            "patient_history": None,
            "patient_id": patient_id,  # Set patient ID from input
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
        """Continue conversation with user's answer to follow-up question."""
        if not self.state:
            return {"type": "final", "output": "[No active conversation]"}

        # Update state with user's answer
        msgs = self.state.get("messages", [])
        msgs.append(HumanMessage(content=answer))
        self.state["messages"] = msgs
        self.state["user_query"] = answer  # Update query with the answer

        # Re-run from retrieval
        out = self.graph.invoke(self.state, self.thread_config)
        self.state = out

        return self._analyze(self.state)

    def _analyze(self, state: PipelineState):
        """Analyze state to determine next step."""
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


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🔵 PERSONAL HEALTH COACH — Multi-Agent System")
    print("="*70 + "\n")
    
    # Initialize SQLite DB first
    print("Initializing SQLite DB...")
    GLOBAL_DB = init_db(DB_PATH)
    print(f"✓ DB initialized at {DB_PATH}\n")
    
    # Initialize memory store - load from DB if exists
    print("Initializing memory store...")
    try:
        GLOBAL_STORE = load_memory_store_from_db(GLOBAL_DB)
        print("✓ Memory store loaded from DB\n")
    except Exception as e:
        logger.warning(f"Could not load from DB, creating new store: {e}")
        GLOBAL_STORE = InMemoryStore()
        print("✓ New memory store initialized\n")

    # Build graph
    planner_graph = build_planner_graph()
    planner = PlannerAgent(planner_graph)

    # Ask for patient ID first
    print("="*70)
    print("PATIENT IDENTIFICATION")
    print("="*70)
    patient_id = input("\nEnter your Patient ID (or press Enter if you're new): ").strip()
    
    # Get health concern
    print("\n" + "="*70)
    print("HEALTH CONCERN")
    print("="*70)
    user_input = input("\nWhat health concern would you like to discuss today?\nUser: ")
    
    # Start conversation with patient ID
    step = planner.start(patient_id, user_input)

    while True:
        if step["type"] == "final":
            print("\n" + "="*70)
            print("HEALTH RECOMMENDATIONS")
            print("="*70)
            print(step["output"])
            print("="*70)
            break

        elif step["type"] == "followup":
            print("\n" + "-"*70)
            print("CONSULTING AGENT:")
            print("-"*70)
            if step.get("explanation"):
                print(f"\n{step['explanation']}")
            print(f"\nFollow-up question: {step['question']}")
            print("-"*70)
            
            answer = input("\nYour answer: ")
            step = planner.continue_with_answer(answer)
    
    # Final save of memory store
    if GLOBAL_STORE and GLOBAL_DB:
        print("\nSaving memory store to database...")
        save_memory_store_to_db(GLOBAL_STORE, GLOBAL_DB)
    
    print("\n" + "="*70)
    print("Thank you for using the Personal Health Coach! Take care! 💙")
    print("="*70 + "\n")
    
    # Close database connection
    if GLOBAL_DB:
        GLOBAL_DB.close()