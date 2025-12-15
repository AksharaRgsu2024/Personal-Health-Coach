from neo4j import GraphDatabase
from neo4j.exceptions import CypherSyntaxError
from openai import OpenAI
from typing import TypedDict, Annotated, Literal, Dict, Any, List, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import json
from langchain.agents import create_agent
from .consulting_agent_with_memory import PatientProfile
from langchain.agents.middleware import wrap_tool_call
from langgraph.prebuilt import ToolNode
from langchain_ollama import ChatOllama
from langgraph.store.memory import InMemoryStore
from langchain_ollama import OllamaEmbeddings
from dataclasses import dataclass
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from dotenv import load_dotenv
import os

load_dotenv()
URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Global store reference
_global_store = None
_global_user_id = None


_neo4j_schema_cache = None
_neo4j_driver_instance = None

def get_cached_neo4j_driver():
    """Reuse Neo4j driver connection"""
    global _neo4j_driver_instance
    if _neo4j_driver_instance is None:
        _neo4j_driver_instance = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    return _neo4j_driver_instance

def schema_text(node_props, rel_props, rels):
    return f"""
  This is the schema representation of the Neo4j database.
  Node properties are the following:
  {node_props}
  Relationship properties are the following:
  {rel_props}
  Relationship point from source to target nodes
  {rels}
  Make sure to respect relationship types and directions
  """

class RetrievedTopic(BaseModel):
    condition: str
    associated_symptoms: list[str]
    recommended_tests: list[str]
    resources: list[str]
    description: str

class RetrievedList(BaseModel):
    items: List[RetrievedTopic]

llm_model = ChatOllama(
    base_url=os.getenv("OLLAMA_SERVER_URL", "http://localhost:11434"),
    model=os.getenv("LLM_MODEL", "gpt-oss:20b"),
    temperature=0.3,  # Slightly higher for faster sampling
    num_predict=2000,  # Limit output tokens
)

class Neo4jGPTQuery:
    node_properties_query = """
    CALL apoc.meta.data()
    YIELD label, other, elementType, type, property
    WHERE NOT type = "RELATIONSHIP" AND elementType = "node"
    WITH label AS nodeLabels, collect(property) AS properties
    RETURN {labels: nodeLabels, properties: properties} AS output
    """

    rel_properties_query = """
    CALL apoc.meta.data()
    YIELD label, other, elementType, type, property
    WHERE NOT type = "RELATIONSHIP" AND elementType = "relationship"
    WITH label AS nodeLabels, collect(property) AS properties
    RETURN {type: nodeLabels, properties: properties} AS output
    """

    rel_query = """
    CALL apoc.meta.data()
    YIELD label, other, elementType, type, property
    WHERE type = "RELATIONSHIP" AND elementType = "node"
    RETURN {source: label, relationship: property, target: other} AS output
    """

    def __init__(self, url, user, password, openai_api_key):
        global _neo4j_schema_cache
        
        self.driver = get_cached_neo4j_driver()
        self.openai_client = OpenAI(api_key=openai_api_key)
        
        # Use cached schema if available
        if _neo4j_schema_cache is None:
            print("⏱️  Generating Neo4j schema (first time only)...")
            _neo4j_schema_cache = self.generate_schema()
            print("✓ Schema cached for future use")
        
        self.schema = _neo4j_schema_cache
        # Skip connection check to save time
        # self.check_connection()

    def generate_schema(self):
        node_props = self.query_database(self.node_properties_query)
        rel_props = self.query_database(self.rel_properties_query)
        rels = self.query_database(self.rel_query)
        return schema_text(node_props, rel_props, rels)

    def get_system_message(self):
        return f"""
        Task: Generate Cypher queries to query a Neo4j graph database based on the provided schema definition.
        Instructions:
        Use only the provided relationship types and properties.
        Do not use any other relationship types or properties that are not provided.
        If you cannot generate a Cypher statement based on the provided schema, explain the reason to the user.
        Schema:
        {self.schema}

        Note: Do not include any explanations or apologies in your responses.
        """

    def query_database(self, neo4j_query, params={}):
        with self.driver.session() as session:
            result = session.run(neo4j_query, params)
            output = result.data()
            return output

    def construct_cypher(self, question, history=None):
        messages = [
            {"role": "system", "content": self.get_system_message()},
            {"role": "user", "content": question},
        ]
        if history:
            messages.extend(history)

        # Use faster model for cypher generation
        completion = self.openai_client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=messages,
            temperature=0.0,
            max_tokens=500,  # Reduced from 1000
        )
        return completion.choices[0].message.content

    def run(self, question, history=None, retry=True):
        query_result = {'cypher query': '', 'result': ''}
        cypher = self.construct_cypher(question, history)
        query_result['cypher query'] = cypher
        try:
            query_result['result'] = self.query_database(cypher)
        except CypherSyntaxError as e:
            if not retry:
                query_result['result'] = []
                query_result['error'] = f"Invalid Cypher syntax: {str(e)}"
                return query_result
            print("⚠️  Retrying Cypher query...")
            retry_result = self.run(
                question,
                [
                    {"role": "assistant", "content": cypher},
                    {
                        "role": "user",
                        "content": f"""This query returns an error: {str(e)} 
                        Give me a improved query that works without any explanations or apologies""",
                    },
                ],
                retry=False
            )
            return retry_result
        except Exception as e:
            query_result['result'] = []
            query_result['error'] = str(e)
        return query_result


@tool
def query_medical_knowledge_graph(symptoms_dict: str) -> list[dict]:
    """
    Query the Neo4j medical knowledge graph to find conditions associated with patient symptoms.
    
    Args:
        symptoms_dict: Description of patient symptoms (e.g., "fever, cough, headache")
        
    Returns:
        List of dictionaries containing conditions and related medical information
    """
    
    # Simplified, more direct query
    query = f"""
    Find up to 5 most relevant medical conditions for symptoms: {symptoms_dict}
    
    Return as list of dictionaries with: condition name, associated symptoms, recommended tests, resources, description.
    Use partial text matching. Be concise.
    """
    
    try:
        start_time = time.time()
        neo4j_gpt = Neo4jGPTQuery(URI, USER, PASSWORD, OPENAI_API_KEY)
        result = neo4j_gpt.run(query)
        elapsed = time.time() - start_time
        
        if not isinstance(result, dict):
            print(f"⚠️  Unexpected result type: {type(result)}")
            return []
        
        retrieved_nodes = result.get('result', [])
        cypher_query = result.get('cypher query', '')
        error = result.get('error', None)
        
        if error:
            print(f"❌ Neo4j query error: {error}")
            return []
        
        print(f"\n{'='*60}")
        print("Generated Cypher Query:")
        print(f"{'='*60}")
        print(cypher_query)
        print(f"{'='*60}\n")
        print(f"Retrieved {len(retrieved_nodes)} nodes from knowledge graph")
        
        if isinstance(retrieved_nodes, list):
            return retrieved_nodes
        else:
            print(f"⚠️  retrieved_nodes is not a list: {type(retrieved_nodes)}")
            return []
    
    except Exception as e:
        print(f"❌ Error querying Neo4j: {str(e)}")
        return []


class PatientRecommendation(BaseModel):
    """Single recommendation record"""
    date: str
    possible_conditions: List[str]
    recommendations: str
    symptoms_at_time: List[str]


class PatientHistory(BaseModel):
    """Complete patient history including profile and all recommendations"""
    profile: PatientProfile
    recommendations: List[PatientRecommendation] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = Field(default_factory=lambda: datetime.now().isoformat())


class Context(TypedDict):
    """Context schema for the agent"""
    user_id: str


def initialize_memory():
    """Initialize the in-memory store"""
    in_memory_store = InMemoryStore()
    return in_memory_store


# 
medical_agent_system_prompt = """
You are a Personal Healthcare Coach with access to long-term patient memory.

STRICTLY FOLLOW THIS WORKFLOW:
1. ALWAYS start by using the get_patient_history tool to check if this patient has visited before.
   - Review past symptoms, conditions, and recommendations
   - Consider how current symptoms relate to past issues
   - Note any recurring patterns or chronic conditions

2. Extract symptoms from the patient's profile and context provided.

3. Use the query_medical_knowledge_graph tool to retrieve relevant medical information based on current symptoms.

4. Generate comprehensive recommendations using ONLY:
   - Current symptoms 
   - Retrieved knowledge graph results
   - Past medical history and recommendations (if available)
   - Patient demographics (age, sex, etc.)
   - Medical reference materials provided

5. Structure your response with these sections:
   - PATIENT CONTEXT: Brief overview noting if returning patient and relevant history
   - CURRENT SYMPTOMS ANALYSIS: What the patient is experiencing now
   - POSSIBLE CONDITIONS: List likely conditions with brief explanations
   - LIFESTYLE RECOMMENDATIONS: Actionable lifestyle modifications
   - POSSIBLE TREATMENTS: Treatment options (tests, medications, therapies)
   - COMPARISON TO PAST: If returning patient, note changes or similarities
   - NEXT STEPS: When to seek medical care
   - REFERENCES: URL links from knowledge graph only

6. AFTER providing recommendations, always use the save_patient_history tool to store:
   - Updated patient profile
   - New recommendation with date, conditions, symptoms, and advice

 USE THIS FORMAT:
save_patient_history(
    patient_profile={
        "id": "<patient_id>",
        "name": "<patient_name>",
        "sex": "<patient_sex>",
        "age": <patient_age_as_number>,
        "symptoms": [list of symptom dicts]
    },
    recommendation={
        "date": "YYYY-MM-DD",
        "possible_conditions": ["condition1", "condition2"],
        "recommendations": "Your recommendations as a single string",
        "symptoms_at_time": ["symptom1", "symptom2"]
    }
)

IMPORTANT RULES:
- Use information from the knowledge graph and provided medical references
- Always emphasize this is informational, not professional medical advice
- Be especially attentive to recurring or worsening symptoms in returning patients
- Consider medication interactions or conflicts with past treatments
- Format responses clearly with section headers
- Be compassionate and clear in your recommendations

MEMORY USAGE:
- First action: get_patient_history
- Last action: save_patient_history
- Use history to provide continuity of care
"""



# Global store reference - will be set by create_personal_care_agent
_global_store = None
_global_user_id = None

#MEMORY TOOLS
@tool
def save_patient_history(
    patient_profile: dict,
    recommendation: dict
) -> str:
    """
    Save or update patient history with new recommendation.
    
    Args:
        patient_profile: Current patient profile dictionary
        recommendation: New recommendation to add
    
    Returns:
        Success message
    """
    try:
        store = _global_store
        user_id = _global_user_id or patient_profile.get("id", "unknown")
        
        if not store:
            return "Error: Memory store not available"
        
        namespace = ("PatientDetails",)
        
        # Try to get existing history
        existing = store.get(namespace, user_id)
        
        if existing and existing.value:
            try:
                history = PatientHistory.model_validate(existing.value)
            except Exception as e:
                print(f"⚠️  Error loading existing history: {e}, creating new")
                history = PatientHistory(profile=PatientProfile(**patient_profile))
        else:
            history = PatientHistory(profile=PatientProfile(**patient_profile))
        
        # Update profile
        history.profile = PatientProfile(**patient_profile)
        
        # Convert list to string if needed
        recommendation_copy = recommendation.copy()
        if "recommendations" in recommendation_copy and isinstance(recommendation_copy["recommendations"], list):
            recommendation_copy["recommendations"] = "\n".join(recommendation_copy["recommendations"])
        
        # Add new recommendation
        new_recommendation = PatientRecommendation(**recommendation_copy)
        history.recommendations.append(new_recommendation)
        history.last_updated = datetime.now().isoformat()
        
        # Save to store
        store.put(namespace, user_id, history.model_dump())

        print(f"✓ Saved patient history for {user_id}. Total: {len(history.recommendations)}")
        return f"Successfully saved. Total consultations: {len(history.recommendations)}"
    
    except Exception as e:
        print(f"❌ Error saving patient history: {str(e)}")
        return f"Error saving: {str(e)}"


@tool
def get_patient_history() -> str:
    """
    Retrieve patient history including past symptoms and recommendations.
    
    Returns:
        Formatted patient history or message if not found
    """
    try:
        store = _global_store
        user_id = _global_user_id
        
        if not store or not user_id:
            return "No previous history available."
        
        namespace = ("PatientDetails",)
        patient_data = store.get(namespace, user_id)
        
        if not patient_data or not patient_data.value:
            return "New patient - no previous history."
        
        history = PatientHistory.model_validate(patient_data.value)
        
        # Compact format for faster processing
        formatted = f"""PATIENT: {history.profile.name} | ID: {history.profile.id} | Age: {history.profile.age} | Sex: {history.profile.sex}
CURRENT SYMPTOMS: {json.dumps([s for s in history.profile.symptoms])}
PAST VISITS: {len(history.recommendations)}
"""
        
        # Include only last 2 visits for speed
        for i, rec in enumerate(list(history.recommendations)[-2:], 1):
            formatted += f"""
Visit {i} ({rec.date}): {', '.join(rec.symptoms_at_time[:3])}
Conditions: {', '.join(rec.possible_conditions[:3])}
"""
        
        return formatted
    
    except Exception as e:
        print(f"❌ Error retrieving history: {str(e)}")
        return "Error retrieving history."


def create_personal_care_agent(store: InMemoryStore, context_schema=Context):
    """
    Create the personal care agent with memory capabilities.
    
    Args:
        store: InMemoryStore instance
        context_schema: Schema for context
    
    Returns:
        Compiled agent graph
    """
    global _global_store
    _global_store = store
    
    agent = create_agent(
        model=llm_model,
        tools=[query_medical_knowledge_graph, get_patient_history, save_patient_history],
        system_prompt=medical_agent_system_prompt,
        context_schema=context_schema
    )
    
    return agent


def run_personal_care_agent(
    patient_profile: dict,
    user_query: str,
    passages: List[str],
    store: InMemoryStore
) -> str:
    """
    Run the personal care agent with patient information.
    
    Args:
        patient_profile: Patient profile dictionary
        user_query: User's health query
        passages: Retrieved medical reference passages
        store: Memory store instance
    
    Returns:
        Care recommendations as string
    """
    # Set global user_id so tools can access it
    global _global_user_id
    patient_id = patient_profile.get("id", "unknown")
    _global_user_id = patient_id
    
    # Create agent
    agent = create_personal_care_agent(store=store, context_schema=Context)
    
    # Build context
    refs_text = ""
    if passages:
        refs_text = "\n".join(f"- {p[:200]}..." for p in passages[:6])
    
    user_content = f"""
Here is the patient's situation.

Original concern:
{user_query}

Patient Profile:
{json.dumps(patient_profile, indent=2)}

Relevant reference snippets from MedlinePlus:
{refs_text}

Please provide comprehensive health recommendations following your instructions as a Personal Healthcare Coach.
Use the knowledge graph to find relevant medical conditions and provide evidence-based advice.
"""
    
    try:
        # Pass context in the initial state
        result = agent.invoke(
            {
                "messages": [{"role": "user", "content": user_content}],
                "context": {
                    "user_id": patient_id
                }
            },
            config={"configurable": {}}
        )
        
        # Extract the final response
        messages = result.get("messages", [])
        
        # Get the last AI message
        care_text = ""
        for msg in reversed(messages):
            if hasattr(msg, 'type') and msg.type == 'ai':
                care_text = msg.content
                break
            elif hasattr(msg, 'role') and msg.role == 'assistant':
                care_text = msg.content
                break
        
        if not care_text:
            care_text = """
I apologize, but I'm having trouble generating recommendations at this time.
Based on your symptoms, I recommend:
1. Monitor your condition closely
2. Stay hydrated and rest
3. Seek medical attention if symptoms worsen or persist
4. Contact a healthcare provider for a proper evaluation

Please remember this is not professional medical advice. Always consult with a healthcare provider for proper diagnosis and treatment.
"""
        
        return care_text
    
    except Exception as e:
        print(f"Error running personal care agent: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return f"""
I encountered an error while generating recommendations: {str(e)}

However, based on your symptoms, I recommend:
1. Monitor your condition closely
2. Stay hydrated and rest
3. Seek medical attention if symptoms worsen or persist
4. Contact a healthcare provider for a proper evaluation

Please remember this is not professional medical advice. Always consult with a healthcare provider for proper diagnosis and treatment.
"""
    
    finally:
        # Clear global state after invocation
        _global_user_id = None

def generate_fallback_response(patient_profile: dict, user_query: str) -> str:
    """Generate a basic fallback response if agent fails or times out."""
    symptoms = patient_profile.get("symptoms", [])
    symptom_list = ", ".join([s.get("description", "") for s in symptoms[:3]])
    
    return f"""
CURRENT SYMPTOMS ANALYSIS:
You are experiencing: {symptom_list}

GENERAL RECOMMENDATIONS:
1. Monitor your symptoms closely and track any changes
2. Stay hydrated and get adequate rest
3. Maintain a balanced diet and avoid triggers
4. Keep a symptom diary to share with healthcare providers

NEXT STEPS:
- If symptoms worsen or persist beyond 3-5 days, consult a healthcare provider
- Seek immediate medical attention if you experience:
  • Severe pain or discomfort
  • Difficulty breathing
  • High fever (>103°F/39.4°C)
  • Chest pain or pressure
  • Confusion or altered mental state

IMPORTANT: This is general information only, not professional medical advice. 
Always consult with a qualified healthcare provider for proper diagnosis and treatment.
"""


if __name__ == "__main__":
    # Initialize memory store
    memory_store = initialize_memory()
    
    print("="*80)
    print("TESTING PERSONAL CARE AGENT")
    print("="*80)
    
    test_profile = {
        "id": "test_001",
        "name": "Test Patient",
        "sex": "Male",
        "age": 35,
        "symptoms": [
            {"description": "fever", "severity": "moderate"},
            {"description": "cough", "severity": "mild"}
        ]
    }
    
    test_query = "I have fever and cough for 3 days"
    test_passages = ["Fever is a common symptom of infection..."]
    
    start = time.time()
    response = run_personal_care_agent(
        test_profile,
        test_query,
        test_passages,
        memory_store,
        timeout=45  # 45 second timeout
    )
    elapsed = time.time() - start
    
    print("\n--- Response ---")
    print(response)
    print("="*80)
    print(f"Total execution time: {elapsed:.2f}s")