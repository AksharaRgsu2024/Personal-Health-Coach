"""
Memory Manager for Patient Data
Handles unified SQLite DB and LangGraph InMemoryStore operations
"""

import sqlite3
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Set, List
from langgraph.store.memory import InMemoryStore

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Unified manager for patient data persistence using SQLite and LangGraph InMemoryStore.
    Handles all database operations and memory store synchronization.
    """
    
    def __init__(self, db_path: str):
        """
        Initialize the MemoryManager.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = Path(db_path)
        self._store: Optional[InMemoryStore] = None
        self._tracked_patient_ids: Optional[Set[str]] = None
        self.namespace = ("PatientDetails",)
        
    def init(self):
        """
        Initialize the database and load existing data.
        Call once per session before any other operations.
        """
        # Initialize database schema
        self._init_db()
        
        # Load memory store from database
        conn = self._get_connection()
        try:
            self._store = self._load_memory_store_from_db(conn)
            
            # Load tracked patient IDs
            self._tracked_patient_ids = set()
            cursor = conn.cursor()
            namespace_str = "/".join(self.namespace)
            cursor.execute(
                "SELECT key FROM memory_store WHERE namespace = ?", 
                (namespace_str,)
            )
            self._tracked_patient_ids.update(row[0] for row in cursor.fetchall())
            
            logger.info(
                f"✓ MemoryManager initialized: {len(self._tracked_patient_ids)} "
                f"patient(s) loaded"
            )
        finally:
            conn.close()
    
    @property
    def store(self) -> InMemoryStore:
        """Get the InMemoryStore instance."""
        if self._store is None:
            raise RuntimeError("MemoryManager not initialized. Call init() first.")
        return self._store
    
    @property
    def tracked_patient_ids(self) -> Set[str]:
        """Get the set of tracked patient IDs."""
        if self._tracked_patient_ids is None:
            raise RuntimeError("MemoryManager not initialized. Call init() first.")
        return self._tracked_patient_ids
    
    def _get_connection(self) -> sqlite3.Connection:
        """Create a new database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_db(self):
        """Initialize SQLite database with required schema."""
        conn = self._get_connection()
        try:
            # Create memory_store table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_store (
                    namespace TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (namespace, key)
                )
            """)
            
            # Create index for faster lookups
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memory_store_namespace 
                ON memory_store(namespace)
            """)
            
            conn.commit()
            logger.info(f"✓ Database initialized: {self.db_path}")
        finally:
            conn.close()
    
    def _load_memory_store_from_db(self, conn: sqlite3.Connection) -> InMemoryStore:
        """Load memory store from SQLite database."""
        try:
            store = InMemoryStore()
            cursor = conn.cursor()
            cursor.execute("SELECT namespace, key, value FROM memory_store")
            rows = cursor.fetchall()
            
            for namespace_str, key, value_json in rows:
                namespace = tuple(namespace_str.split("/"))
                value = json.loads(value_json)
                store.put(namespace, key, value)
            
            logger.info(f"✓ Memory store loaded from DB ({len(rows)} items)")
            return store
        except Exception as e:
            logger.error(f"Error loading memory store from DB: {e}")
            return InMemoryStore()
    
    def track_patient_id(self, patient_id: str):
        """
        Track a patient ID for later retrieval and persistence.
        
        Args:
            patient_id: The patient ID to track
        """
        if self._tracked_patient_ids is None:
            raise RuntimeError("MemoryManager not initialized. Call init() first.")
        
        self._tracked_patient_ids.add(patient_id)
        logger.info(f"✓ Tracking patient ID: {patient_id}")
    
    def save_to_db(self):
        """
        Persist the entire memory store to SQLite.
        Uses tracked patient IDs to ensure all patients are saved.
        """
        if self._store is None or self._tracked_patient_ids is None:
            raise RuntimeError("MemoryManager not initialized. Call init() first.")
        
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            now = datetime.now().isoformat()
            namespace_str = "/".join(self.namespace)
            
            # Check for tracked IDs, fallback to DB if empty
            if not self._tracked_patient_ids:
                logger.warning("⚠️  No patient IDs tracked! Loading from database...")
                cursor.execute(
                    "SELECT key FROM memory_store WHERE namespace = ?", 
                    (namespace_str,)
                )
                self._tracked_patient_ids.update(row[0] for row in cursor.fetchall())
                logger.info(f"Loaded {len(self._tracked_patient_ids)} patient IDs from DB")
            
            saved_count = 0
            updated_count = 0
            error_count = 0
            not_found_count = 0
            
            logger.info(f"Processing {len(self._tracked_patient_ids)} tracked patient IDs...")
            
            for patient_id in self._tracked_patient_ids:
                try:
                    # Get patient data from store
                    patient_item = self._store.get(self.namespace, patient_id)
                    
                    if not patient_item or not patient_item.value:
                        logger.warning(f"Patient {patient_id} not found in store")
                        not_found_count += 1
                        continue
                    
                    patient_data = patient_item.value
                    
                    # Validate data structure
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
                    
                    # Serialize to JSON
                    try:
                        value_json = json.dumps(patient_data, ensure_ascii=False)
                    except Exception as json_err:
                        logger.error(f"JSON serialization failed for {patient_id}: {json_err}")
                        error_count += 1
                        continue
                    
                    # Check if record exists
                    cursor.execute(
                        "SELECT created_at FROM memory_store WHERE namespace = ? AND key = ?",
                        (namespace_str, patient_id)
                    )
                    existing = cursor.fetchone()
                    
                    if existing:
                        # Update existing record
                        cursor.execute("""
                            UPDATE memory_store 
                            SET value = ?, updated_at = ?
                            WHERE namespace = ? AND key = ?
                        """, (value_json, now, namespace_str, patient_id))
                        updated_count += 1
                        
                        profile = patient_data.get("profile", {})
                        logger.info(
                            f"  ✓ Updated: {profile.get('name', 'Unknown')} "
                            f"({patient_id}) - {len(recommendations)} rec(s)"
                        )
                    else:
                        # Insert new record
                        created_at = patient_data.get("created_at", now)
                        cursor.execute("""
                            INSERT INTO memory_store (namespace, key, value, created_at, updated_at)
                            VALUES (?, ?, ?, ?, ?)
                        """, (namespace_str, patient_id, value_json, created_at, now))
                        saved_count += 1
                        
                        profile = patient_data.get("profile", {})
                        logger.info(
                            f"  ✅ NEW: {profile.get('name', 'Unknown')} "
                            f"({patient_id}) - {len(recommendations)} rec(s)"
                        )
                
                except Exception as item_error:
                    logger.error(f"Error processing {patient_id}: {item_error}")
                    error_count += 1
                    continue
            
            # Commit changes
            conn.commit()
            
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
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def get_patient(self, patient_id: str) -> Optional[dict]:
        """
        Retrieve patient data from the memory store.
        
        Args:
            patient_id: The patient ID to retrieve
            
        Returns:
            Patient data dictionary or None if not found
        """
        if self._store is None:
            raise RuntimeError("MemoryManager not initialized. Call init() first.")
        
        try:
            patient_item = self._store.get(self.namespace, patient_id)
            if patient_item and patient_item.value:
                return patient_item.value
        except Exception as e:
            logger.error(f"Error retrieving patient {patient_id}: {e}")
        
        return None
    
    def save_patient(self, patient_id: str, patient_data: dict, persist_immediately: bool = True):
        """
        Save patient data to the memory store and optionally persist to DB.
        
        Args:
            patient_id: The patient ID
            patient_data: Patient data dictionary
            persist_immediately: If True, immediately persist to database
        """
        if self._store is None:
            raise RuntimeError("MemoryManager not initialized. Call init() first.")
        
        try:
            # Update timestamps
            now = datetime.now().isoformat()
            if "last_updated" not in patient_data:
                patient_data["last_updated"] = now
            if "created_at" not in patient_data:
                patient_data["created_at"] = now
            
            # Save to memory store
            self._store.put(self.namespace, patient_id, patient_data)
            logger.info(f"✓ Patient {patient_id} saved to memory store")
            
            # Track the patient ID
            self.track_patient_id(patient_id)
            
            # Verify it's in the store
            verification = self._store.get(self.namespace, patient_id)
            if not verification or not verification.value:
                logger.error(f"❌ Patient {patient_id} not retrievable after save!")
                return False
            
            # Persist to database if requested
            if persist_immediately:
                logger.info("Immediately persisting to database...")
                self.save_to_db()
                
                # Verify in database
                conn = self._get_connection()
                try:
                    cursor = conn.cursor()
                    namespace_str = "/".join(self.namespace)
                    cursor.execute(
                        "SELECT key FROM memory_store WHERE namespace = ? AND key = ?",
                        (namespace_str, patient_id)
                    )
                    result = cursor.fetchone()
                    
                    if result:
                        logger.info(f"✓ VERIFIED: Patient {patient_id} is in DATABASE")
                        return True
                    else:
                        logger.error(f"❌ Patient {patient_id} NOT in database after save!")
                        return False
                finally:
                    conn.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving patient {patient_id}: {e}")
            return False
    
    def search_patients(self, prefix: str = "") -> List[dict]:
        """
        Search for patients in the memory store.
        
        Args:
            prefix: Optional prefix to filter patient IDs
            
        Returns:
            List of patient data dictionaries
        """
        if self._store is None:
            raise RuntimeError("MemoryManager not initialized. Call init() first.")
        
        try:
            results = list(self._store.search(self.namespace))
            patients = []
            
            for item in results:
                if prefix and not item.key.startswith(prefix):
                    continue
                if item.value:
                    patients.append(item.value)
            
            return patients
        except Exception as e:
            logger.error(f"Error searching patients: {e}")
            return []
    
    def debug_store(self):
        """Debug function to inspect the memory store contents."""
        if self._store is None:
            print("❌ Store not initialized")
            return
        
        print("\n" + "="*60)
        print("MEMORY STORE DEBUG INFO")
        print("="*60)
        
        # Search method
        try:
            results = list(self._store.search(self.namespace))
            print(f"✓ store.search() found {len(results)} items")
            for item in results:
                profile = item.value.get('profile', {})
                name = profile.get('name', 'Unknown')
                rec_count = len(item.value.get('recommendations', []))
                print(f"  - {item.key}: {name} ({rec_count} recommendations)")
        except Exception as e:
            print(f"❌ store.search() failed: {e}")
        
        # Internal data structure
        if hasattr(self._store, '_data'):
            print(f"\n✓ Store has _data attribute")
            if self.namespace in self._store._data:
                print(f"✓ Namespace exists with {len(self._store._data[self.namespace])} items")
            else:
                print(f"❌ Namespace not found. Available: {list(self._store._data.keys())}")
        
        print(f"\n✓ Tracked patient IDs: {len(self._tracked_patient_ids)}")
        for pid in self._tracked_patient_ids:
            print(f"  - {pid}")
        
        print("="*60 + "\n")
    
    def lookup_patient(self, patient_id: str):
        """
        Look up patient and return PatientHistory object.
        
        Args:
            patient_id: The patient ID to look up
            
        Returns:
            PatientHistory object or None if not found
        """
        from agents.personal_care_agent_lt_memory import PatientHistory
        
        patient_data = self.get_patient(patient_id)
        if patient_data:
            try:
                return PatientHistory.model_validate(patient_data)
            except Exception as e:
                logger.error(f"Error validating patient data: {e}")
        return None
    
    def save_patient_history(self, history, persist_immediately: bool = True):
        """
        Save PatientHistory object to store and DB.
        
        Args:
            history: PatientHistory object
            persist_immediately: If True, immediately persist to database
        """
        patient_id = history.profile.id
        patient_data = history.model_dump()
        return self.save_patient(patient_id, patient_data, persist_immediately)
    
    def save_new_patient_profile(self, patient_profile, persist_immediately: bool = True) -> bool:
        """
        Save a new patient profile to the store and DB.
        
        Args:
            patient_profile: PatientProfile object
            persist_immediately: If True, immediately persist to database
            
        Returns:
            True if successful, False otherwise
        """
        from agents.personal_care_agent_lt_memory import PatientHistory
        
        try:
            now = datetime.now().isoformat()
            history = PatientHistory(
                profile=patient_profile,
                recommendations=[],
                created_at=now,
                last_updated=now
            )
            
            patient_id = patient_profile.id
            return self.save_patient(patient_id, history.model_dump(), persist_immediately)
            
        except Exception as e:
            logger.error(f"Error saving new patient profile: {e}")
            return False
    
    def update_patient_with_recommendation(self, patient_id: str, plan_text: str, 
                                          patient_profile_dict: dict, 
                                          care_details: dict) -> bool:
        """
        Update patient history with a new recommendation.
        
        Args:
            patient_id: Patient ID
            plan_text: Care plan text
            patient_profile_dict: Patient profile dictionary
            care_details: Care details dictionary with possible conditions
            
        Returns:
            True if successful, False otherwise
        """
        from agents.personal_care_agent_lt_memory import PatientHistory, PatientProfile, PatientRecommendation
        
        try:
            logger.info(f"Retrieving patient {patient_id} from memory store...")
            history = self.lookup_patient(patient_id)
            
            if history is None:
                logger.warning(f"Patient {patient_id} not in store. Creating from profile...")
                
                if not patient_profile_dict:
                    logger.error("No patient profile provided, cannot save")
                    return False
                
                try:
                    profile = PatientProfile(**patient_profile_dict)
                    history = PatientHistory(
                        profile=profile,
                        recommendations=[],
                        created_at=datetime.now().isoformat(),
                        last_updated=datetime.now().isoformat()
                    )
                    self.save_patient(patient_id, history.model_dump(), persist_immediately=False)
                    logger.info(f"✓ Created and saved new history for {patient_id}")
                except Exception as e:
                    logger.error(f"Failed to create patient history: {e}")
                    return False
            
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
                
                symptoms = patient_profile_dict.get("symptoms", [])
                symptoms_list = [s.get("description", "") for s in symptoms if s.get("description")]
                
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
            
            # Save to memory store and DB
            success = self.save_patient_history(history, persist_immediately=True)
            
            if success:
                print(f"\n{'='*60}")
                print(f"✓ Patient history saved to database")
                print(f"  Patient: {history.profile.name} ({patient_id})")
                print(f"  Total visits: {len(history.recommendations)}")
                if history.recommendations:
                    latest_rec = history.recommendations[-1]
                    print(f"  Latest visit: {latest_rec.date}")
                print(f"{'='*60}\n")
            
            return success
            
        except Exception as e:
            logger.error(f"Error updating patient history: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    @staticmethod
    def generate_patient_id(name: str) -> str:
        """
        Generate a unique patient ID based on name and timestamp.
        
        Args:
            name: Patient name
            
        Returns:
            Unique patient ID string
        """
        name_part = ''.join(name.split()[:2])[:6].upper().replace(" ", "")
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        return f"P-{name_part}-{timestamp}"