from pathlib import Path
import os
from vector_db import semantic_search, load_embedding_model
from agents.consulting_agent_with_memory import build_consulting_agent_graph, PatientProfile
from agents.personal_care_agent_lt_memory import PatientRecommendation, PatientHistory, create_personal_care_agent, PatientHistory, run_personal_care_agent
import sqlite3
import json
import re
import logging
from typing_extensions import TypedDict
from typing import List, Any, Literal, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from dataclasses import dataclass
from langgraph.store.memory import InMemoryStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
from datetime import datetime
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
import sqlite3
import pickle
from langchain_core.runnables.graph import MermaidDrawMethod
from IPython.display import Image

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


@dataclass
class Context:
    user_id: str
    store: InMemoryStore


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
TRACKED_PATIENT_IDS = set()  # Track all patient IDs we've created/seen

def track_patient_id(patient_id: str):
    """Track a patient ID for later retrieval."""
    global TRACKED_PATIENT_IDS
    TRACKED_PATIENT_IDS.add(patient_id)
    logger.info(f"✓ Tracking patient ID: {patient_id}")

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

def get_connection():
    return sqlite3.connect(DB_PATH)

def save_memory_store_to_db():
    """
    Persist the entire memory store to SQLite.
    Uses manual tracking instead of store.search() to avoid missing new patients.
    """
    global GLOBAL_STORE, GLOBAL_DB, TRACKED_PATIENT_IDS
    
    if not GLOBAL_STORE:
        logger.error("Memory store not initialized")
        return
    
    if not GLOBAL_DB:
        logger.error("DB connection not initialized")
        return
    
    try:
        # conn=get_connection()
        # cursor = conn.cursor()
        cursor = GLOBAL_DB.cursor()
        now = datetime.now().isoformat()
        namespace = ("PatientDetails",)
        namespace_str = "/".join(namespace)
        
        # CRITICAL FIX: Don't use search() - use our tracked IDs instead
        logger.info(f"Processing {len(TRACKED_PATIENT_IDS)} tracked patient IDs...")
        
        if not TRACKED_PATIENT_IDS:
            logger.warning("⚠️  No patient IDs tracked! This shouldn't happen.")
            # Fallback: try to get IDs from database
            cursor.execute("SELECT key FROM memory_store WHERE namespace = ?", (namespace_str,))
            db_rows = cursor.fetchall()
            for row in db_rows:
                TRACKED_PATIENT_IDS.add(row[0])
            logger.info(f"Loaded {len(TRACKED_PATIENT_IDS)} patient IDs from database")
        
        saved_count = 0
        updated_count = 0
        error_count = 0
        not_found_count = 0
        
        for patient_id in TRACKED_PATIENT_IDS:
            try:
                # Get patient data using get() instead of search()
                patient_item = GLOBAL_STORE.get(namespace, patient_id)
                
                if not patient_item or not patient_item.value:
                    logger.warning(f"Patient {patient_id} not found in store")
                    not_found_count += 1
                    continue
                
                patient_data = patient_item.value
                
                # Validate the data structure
                if not isinstance(patient_data, dict):
                    logger.warning(f"Invalid data type for {patient_id}: {type(patient_data)}")
                    error_count += 1
                    continue
                
                if "profile" not in patient_data:
                    logger.warning(f"Missing profile for {patient_id}")
                    error_count += 1
                    continue
                
                recommendations = patient_data.get("recommendations", [])
                if not isinstance(recommendations, list):
                    logger.warning(f"Invalid recommendations for {patient_id}")
                    error_count += 1
                    continue
                
                # Convert to JSON
                try:
                    value_json = json.dumps(patient_data, ensure_ascii=False)
                except Exception as json_err:
                    logger.error(f"JSON serialization failed for {patient_id}: {json_err}")
                    error_count += 1
                    continue
                
                # Check if record exists in database
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
                    
                    profile = patient_data.get("profile", {})
                    logger.info(
                        f"  ✓ Updated: {profile.get('name', 'Unknown')} "
                        f"({patient_id}) - {len(recommendations)} rec(s)"
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
                    
                    profile = patient_data.get("profile", {})
                    logger.info(
                        f"  ✅ NEW: {profile.get('name', 'Unknown')} "
                        f"({patient_id}) - {len(recommendations)} rec(s)"
                    )
                    
            except Exception as item_error:
                logger.error(f"Error processing {patient_id}: {item_error}")
                import traceback
                traceback.print_exc()
                error_count += 1
                continue
        
        # Commit changes
        # conn.commit()
        GLOBAL_DB.commit()
        
        total = saved_count + updated_count
        logger.info(
            f"✓ Memory store persisted: "
            f"{saved_count} new, {updated_count} updated, "
            f"{error_count} errors, {not_found_count} not found (total: {total})"
        )
        
        # Verify database count
        cursor.execute(
            "SELECT COUNT(*) FROM memory_store WHERE namespace = ?",
            (namespace_str,)
        )
        db_count = cursor.fetchone()[0]
        logger.info(f"✓ Database contains {db_count} patient record(s)")
        
    except Exception as e:
        logger.error(f"Error saving memory store to DB: {e}")
        import traceback
        traceback.print_exc()
        try:
            GLOBAL_DB.rollback()
            # conn.rollback()
        except:
            pass



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
        save_memory_store_to_db()
        
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
    global GLOBAL_DB, GLOBAL_STORE
    namespace = ("PatientDetails",)
    
    try:
        now = datetime.now().isoformat()
        history = PatientHistory(
            profile=patient_profile,
            recommendations=[],
            created_at=now,
            last_updated=now
        )
        
        patient_id = patient_profile.id
        
        # Save to memory store
        GLOBAL_STORE.put(namespace, patient_id, history.model_dump())
        logger.info(f"✓ Put patient {patient_id} into memory store")
        
        # CRITICAL: Track this patient ID
        track_patient_id(patient_id)
        
        # Verify it's in the store
        verification = GLOBAL_STORE.get(namespace, patient_id)
        if not verification or not verification.value:
            logger.error(f"❌ Patient {patient_id} not retrievable from store after put()!")
            return False
        
        logger.info(f"✓ Verified patient {patient_id} is in memory store")
        
        # Immediately persist to DB
        if GLOBAL_DB:
            logger.info("Immediately persisting new patient to DB...")
            save_memory_store_to_db()
            
            # Verify it's in the database
            cursor = GLOBAL_DB.cursor()
            # conn=get_connection()
            # cursor = conn.cursor()
            cursor.execute(
                "SELECT key FROM memory_store WHERE namespace = ? AND key = ?",
                ("/".join(namespace), patient_id)
            )
            result = cursor.fetchone()
            
            if result:
                logger.info(f"✓ VERIFIED: Patient {patient_id} is in DATABASE")
                return True
            else:
                logger.error(f"❌ Patient {patient_id} NOT in database after save!")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error saving patient: {e}")
        import traceback
        traceback.print_exc()
        return False



def update_patient_history_with_plan(state: PipelineState):
    """
    Ensure patient history with recommendations is saved to SQLite DB.
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
    plan_text = state.get("plan", "")
    
    if not plan_text:
        logger.warning("No care plan generated, skipping history update")
        return

    try:
        logger.info(f"Retrieving patient {patient_id} from LangGraph store...")
        history = lookup_patient_in_store(patient_id, GLOBAL_STORE)
        
        if history is None:
            logger.warning(f"Patient {patient_id} not in store. Creating from state...")
            
            profile_dict = state.get("patient_profile", {})
            if not profile_dict:
                logger.error("No patient profile in state, cannot save")
                return
            
            try:
                profile = PatientProfile(**profile_dict)
                history = PatientHistory(
                    profile=profile,
                    recommendations=[],
                    created_at=datetime.now().isoformat(),
                    last_updated=datetime.now().isoformat()
                )
                # Immediately save this new history to store
                GLOBAL_STORE.put(namespace, patient_id, history.model_dump())
                logger.info(f"✓ Created and saved new history for {patient_id}")
            except Exception as e:
                logger.error(f"Failed to create patient history: {e}")
                return
        
        logger.info(f"✓ Patient {history.profile.name} loaded with {len(history.recommendations)} existing recommendations")
        
        # Check if already saved (prevent duplicates)
        today = datetime.now().strftime("%Y-%m-%d")
        already_saved = False
        
        if history.recommendations:
            latest_rec = history.recommendations[-1]
            if latest_rec.date == today and plan_text[:100] in latest_rec.recommendations:
                logger.info("Recommendation already saved")
                already_saved = True
        
        # Create and add recommendation if not already saved
        if not already_saved:
            logger.info("Creating new recommendation...")
            
            profile_dict = state.get("patient_profile", {})
            symptoms = profile_dict.get("symptoms", [])
            symptoms_list = [s.get("description", "") for s in symptoms if s.get("description")]
            
            care_details = state.get("care_details", {})
            conditions = care_details.get("possible_conditions", [])
            
            if not conditions and "POSSIBLE CONDITIONS:" in plan_text:
                conditions_section = plan_text.split("POSSIBLE CONDITIONS:")[1].split("\n\n")[0]
                conditions = [line.strip("•- ") for line in conditions_section.split("\n") if line.strip()][:5]
            
            new_recommendation = PatientRecommendation(
                date=today,
                possible_conditions=conditions[:5] if conditions else ["Consultation completed"],
                recommendations=plan_text,
                symptoms_at_time=symptoms_list if symptoms_list else ["General health concern"]
            )
            
            history.recommendations.append(new_recommendation)
            logger.info(f"✓ Added new recommendation. Total: {len(history.recommendations)}")
        
        # Update timestamps
        history.last_updated = datetime.now().isoformat()
        
        # Save to memory store
        logger.info("Saving to LangGraph memory store...")
        GLOBAL_STORE.put(namespace, patient_id, history.model_dump())
        logger.info("✓ Saved to memory store")
        
        # CRITICAL: Persist to SQLite DB using globals
        logger.info("Persisting memory store to SQLite DB...")
        save_memory_store_to_db()  # No params - uses globals
        logger.info("✓ Memory store persisted to DB")
        
        # Verify in database
        cursor = GLOBAL_DB.cursor()
        # conn=get_connection()
        # cursor = conn.cursor()
        cursor.execute(
            "SELECT value FROM memory_store WHERE namespace = ? AND key = ?",
            ("/".join(namespace), patient_id)
        )
        db_row = cursor.fetchone()
        
        if db_row:
            db_data = json.loads(db_row[0])
            db_rec_count = len(db_data.get("recommendations", []))
            logger.info(f"✓ Database verification SUCCESS: {db_rec_count} recommendations for {patient_id}")
            
            print(f"\n{'='*60}")
            print(f"✓ Patient history saved to database")
            print(f"  Patient: {history.profile.name} ({patient_id})")
            print(f"  Total visits: {len(history.recommendations)}")
            if history.recommendations:
                latest_rec = history.recommendations[-1]
                print(f"  Latest visit: {latest_rec.date}")
            print(f"{'='*60}\n")
        else:
            logger.error(f"❌ Patient {patient_id} not found in database after save!")
            logger.error("This indicates save_memory_store_to_db() is not working correctly")
            
            # Debug: Check what's actually in the store
            test_get = GLOBAL_STORE.get(namespace, patient_id)
            if test_get and test_get.value:
                logger.info(f"✓ Patient IS in memory store (has {len(test_get.value.get('recommendations', []))} recs)")
            else:
                logger.error("❌ Patient NOT in memory store either!")
        
    except Exception as e:
        logger.error(f"Error updating patient history: {e}")
        import traceback
        traceback.print_exc()
        if GLOBAL_DB:
            try:
                GLOBAL_DB.rollback()
            except:
                pass

def debug_memory_store():
    """Debug function to check what's in the memory store."""
    global GLOBAL_STORE
    
    if not GLOBAL_STORE:
        print("❌ GLOBAL_STORE is None")
        return
    
    namespace = ("PatientDetails",)
    
    print("\n" + "="*60)
    print("DEBUGGING MEMORY STORE")
    print("="*60)
    
    # Try search method
    try:
        results = list(GLOBAL_STORE.search(namespace))
        print(f"✓ store.search() found {len(results)} items")
        for item in results:
            print(f"  - {item.key}: {item.value.get('profile', {}).get('name', 'Unknown')}")
    except Exception as e:
        print(f"❌ store.search() failed: {e}")
    
    # Try direct access if available
    if hasattr(GLOBAL_STORE, '_data'):
        print(f"\n✓ Store has _data attribute")
        if namespace in GLOBAL_STORE._data:
            print(f"✓ Namespace exists in _data")
            print(f"✓ Contains {len(GLOBAL_STORE._data[namespace])} items")
            for key in GLOBAL_STORE._data[namespace].keys():
                print(f"  - {key}")
        else:
            print(f"❌ Namespace NOT in _data")
            print(f"Available namespaces: {list(GLOBAL_STORE._data.keys())}")
    else:
        print("❌ Store has no _data attribute")
    
    print("="*60 + "\n")


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
    
    if state.get("profile_complete"):
        logging.info("Profile already complete, skipping intake")
        return state
    
    patient_id = state.get("patient_id", "").strip()
    
    if patient_id:
        print(f"🔍 Looking up patient ID: {patient_id}...")
        patient_hist = lookup_patient_in_store(patient_id, GLOBAL_STORE)
        
        if patient_hist:
            print(f"\n✓ Welcome back, {patient_hist.profile.name}!")
            print(f"   Age: {patient_hist.profile.age}, Sex: {patient_hist.profile.sex}")
            print(f"   Previous visits: {len(patient_hist.recommendations)}")
            
            # CRITICAL: Track this returning patient
            track_patient_id(patient_id)
            
            if patient_hist.recommendations:
                last_rec = patient_hist.recommendations[-1]
                print(f"   Last visit: {last_rec.date}")
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
    
    new_patient_id = generate_patient_id(patient_name)
    print(f"\n🆔 Generated Patient ID: {new_patient_id}")
    print("⚠️  Please save this ID for future visits!")
    
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


def route_from_intake(state: PipelineState) -> Literal["retriever_agent", END]:
    """Route after intake based on whether profile is complete."""
    if state.get("profile_complete"):
        logging.info("✓ Profile complete, proceeding to retrieval")
        return "retriever_agent"  # ✅ Changed from "retrieve"
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
    MAX_QUESTIONS=3
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

    current_turn = state.get("turn_count", 0)

    agent_state = {
        "user_query": state["user_query"],
        "messages": state.get("messages", []),
        "retrieved_docs": retrieved_docs,
        "patient_profile": patient_profile,
        "turn_count":  current_turn,
        "need_more_info": True,
        "red_flag": False,
        "last_question": state.get("last_question"),
    }

    updated = consult_graph.invoke(agent_state)

    print("\n===== DEBUG CONSULT OUTPUT =====")
    print(f"Need more info: {updated.get('need_more_info')}")
    print(f"Turn count: {updated.get('turn_count')} / {MAX_QUESTIONS}")
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
    if (bool(followup_q)==False):
        state["needs_more"] = False
    state["last_question"] = updated.get("last_question")
    
    logger.info(
        f"Consulting agent decision: needs_more={state['needs_more']}, "
        f"followup_q={bool(followup_q)}, turn={state['turn_count']}/{MAX_QUESTIONS}"
    )

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


def route_from_consult(state: PipelineState) -> Literal["followup", "personal_care_agent"]:
    if state.get("needs_more"):
        return "followup"
    else:
        return "personal_care_agent"

# ----------------- Build the graph -----------------
def build_planner_graph():
    """Build the complete multi-agent workflow."""
    workflow = StateGraph(PipelineState)
    
    workflow.add_node("patient_intake", patient_intake_node)
    workflow.add_node("retriever_agent", retriever_agent)
    workflow.add_node("consulting_agent", consulting_agent)
    workflow.add_node("followup", followup_node)
    workflow.add_node("personal_care_agent", personal_care_agent)
    workflow.add_node("planner_summary", planner_summary_node)

    workflow.set_entry_point("patient_intake")
    
    # Route from intake based on profile completion
    workflow.add_conditional_edges(
        "patient_intake",
        route_from_intake,
        {
            "retriever_agent": "retriever_agent",
            END: END
        }
    )
    
    workflow.add_edge("retriever_agent", "consulting_agent")

    workflow.add_conditional_edges(
        "consulting_agent",
        route_from_consult,
        {
            "followup": "followup",
            "personal_care_agent": "personal_care_agent",
        }
    )

    
    workflow.add_edge("personal_care_agent", "planner_summary")
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

def save_graph(planner_graph):
    # Save graph image
    graph = planner_graph.get_graph()
    png_bytes = graph.draw_mermaid_png(draw_method=MermaidDrawMethod.API)

    with open("langgraph_diagram.png", "wb") as f:
        f.write(png_bytes)
        

def init_memory_backend():
    global GLOBAL_STORE, GLOBAL_DB, TRACKED_PATIENT_IDS
    
    # Initialize SQLite DB first
    print("Initializing SQLite DB...")
    GLOBAL_DB = init_db(DB_PATH)
    print(f"✓ DB initialized at {DB_PATH}\n")
    
    # Initialize memory store - load from DB if exists
    print("Initializing memory store...")
    try:
        GLOBAL_STORE = load_memory_store_from_db(GLOBAL_DB)
        print("✓ Memory store loaded from DB\n")
        
        # CRITICAL: Load all patient IDs from database into tracking set
        cursor = GLOBAL_DB.cursor()
        cursor.execute(
            "SELECT key FROM memory_store WHERE namespace = ?",
            ("PatientDetails",)
        )
        rows = cursor.fetchall()
        for row in rows:
            TRACKED_PATIENT_IDS.add(row[0])
        
        print(f"✓ Tracking {len(TRACKED_PATIENT_IDS)} existing patient IDs\n")
        
    except Exception as e:
        logger.warning(f"Could not load from DB, creating new store: {e}")
        GLOBAL_STORE = InMemoryStore()
        print("✓ New memory store initialized\n")
    
    # Load embedding model
    load_embedding_model()
    print(f"✓ Embedding model loaded for semantic search\n")


def final_memory_save():
    global GLOBAL_STORE, GLOBAL_DB
        # Final save of memory store
    if GLOBAL_STORE and GLOBAL_DB:
        print("\nSaving memory store to database...")
        save_memory_store_to_db()
        print("✓ Memory store saved to database.\n")
    
def close_database():
    global GLOBAL_DB
    # Close database connection
    if GLOBAL_DB:
        GLOBAL_DB.close()

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🔵 PERSONAL HEALTH COACH — Multi-Agent System")
    print("="*70 + "\n")
    
    init_memory_backend()
    # Build graph
    planner_graph = build_planner_graph()
    # save_graph(planner_graph)
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
    final_memory_save()
    close_database()
    print("\n" + "="*70)
    print("Thank you for using the Personal Health Coach! Take care! 💙")
    print("="*70 + "\n")
    
    