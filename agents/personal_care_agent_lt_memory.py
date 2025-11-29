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
from . consulting_agent_with_memory import PatientProfile
from langchain.agents.middleware import wrap_tool_call
from langgraph.prebuilt import ToolNode
from langchain_ollama import ChatOllama
from langgraph.store.memory import InMemoryStore
from langchain_ollama import OllamaEmbeddings
from dataclasses import dataclass
from datetime import datetime
# Demo database credentials
from dotenv import load_dotenv
import os
load_dotenv()
URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Global store reference - will be set by create_personal_care_agent
_global_store = None
_global_user_id = None

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
    base_url="http://10.230.100.240:17020/",
    model="gpt-oss:20b",
    temperature=0.3
)

# Neo4j Query class
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
        self.driver = GraphDatabase.driver(url, auth=(user, password))
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.schema = self.generate_schema()
        self.check_connection()

    def check_connection(self):
        try:
            self.driver.verify_connectivity()
            print("✓ Neo4j connection successful!")
        except Exception as e:
            print(f"✗ Failed to connect to Neo4j: {e}")

    def generate_schema(self):
        node_props = self.query_database(self.node_properties_query)
        rel_props = self.query_database(self.rel_properties_query)
        rels = self.query_database(self.rel_query)
        return schema_text(node_props, rel_props, rels)

    def refresh_schema(self):
        self.schema = self.generate_schema()

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

        completion = self.openai_client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=messages,
            temperature=0.0,
            max_tokens=1000,
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
            print("Retrying Cypher query...")
            # Recursively retry
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
            # Return the retry result (which is already a dict)
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
    
    query = f"""
    Find all medical conditions (MeshDescriptor nodes) that are associated with the symptoms and related context given in this dictionary: {symptoms_dict}
    
    For each condition found, provide:
    - Condition name
    - Associated symptoms
    - Recommended diagnostic tests
    - Available health topics or resources
    - Any summary or description information
    
    Match symptoms flexibly using partial text matching.
    Return result as list of dictionaries
    """
    
    try:
        neo4j_gpt = Neo4jGPTQuery(URI, USER, PASSWORD, OPENAI_API_KEY)
        result = neo4j_gpt.run(query)
        
        # Check if result is a dict (should be)
        if not isinstance(result, dict):
            print(f"Unexpected result type: {type(result)}")
            return []
        
        retrieved_nodes = result.get('result', [])
        cypher_query = result.get('cypher query', '')
        error = result.get('error', None)
        
        if error:
            print(f"Neo4j query error: {error}")
            return []
        
        print(f"\n{'='*60}")
        print("Generated Cypher Query:")
        print(f"{'='*60}")
        print(cypher_query)
        print(f"{'='*60}\n")
        print(f"Retrieved {len(retrieved_nodes)} nodes from knowledge graph")
        
        # Ensure we return a list
        if isinstance(retrieved_nodes, list):
            return retrieved_nodes
        else:
            print(f"Warning: retrieved_nodes is not a list: {type(retrieved_nodes)}")
            return []
    
    except Exception as e:
        print(f"Error querying Neo4j: {str(e)}")
        import traceback
        traceback.print_exc()
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
    # created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = Field(default_factory=lambda: datetime.now().isoformat())


class Context(TypedDict):
    """Context schema for the agent"""
    user_id: str


def initialize_memory():
    """Initialize the in-memory store"""
    in_memory_store = InMemoryStore()
    return in_memory_store

medical_agent_system_prompt = """
You are a Personal Healthcare Coach with access to long-term patient memory.

WORKFLOW:
1. ALWAYS start by using the get_patient_history tool to check if this patient has visited before.
   - Review past symptoms, conditions, and recommendations
   - Consider how current symptoms relate to past issues
   - Note any recurring patterns or chronic conditions

2. Extract symptoms from the patient's profile and context provided.

3. Use the query_medical_knowledge_graph tool to retrieve relevant medical information based on current symptoms.

4. Generate comprehensive recommendations considering:
   - Current symptoms and retrieved medical knowledge
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

6. AFTER providing recommendations, use the save_patient_history tool to store:
   - Updated patient profile
   - New recommendation with date, conditions, symptoms, and advice

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

#TOOLS
@tool
def save_patient_history(
    patient_profile: dict,
    recommendation: dict
) -> str:
    """
    Save or update patient history with new recommendation.
    
    Args:
        patient_profile: Current patient profile dictionary
        recommendation: New recommendation to add (dict with date, conditions, recommendations, symptoms)
    
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
                print(f"Error loading existing history: {e}, creating new")
                history = PatientHistory(profile=PatientProfile(**patient_profile))
        else:
            history = PatientHistory(profile=PatientProfile(**patient_profile))
        
        # Update profile with latest information
        history.profile = PatientProfile(**patient_profile)
        
        # Add new recommendation
        new_recommendation = PatientRecommendation(**recommendation)
        history.recommendations.append(new_recommendation)
        history.last_updated = datetime.now().isoformat()
        
        # Save to store
        store.put(namespace, user_id, history.model_dump())
        
        print(f"✓ Saved patient history for {user_id}. Total consultations: {len(history.recommendations)}")
        return f"Successfully saved patient history. Total consultations: {len(history.recommendations)}"
    
    except Exception as e:
        print(f"Error saving patient history: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error saving patient history: {str(e)}"


@tool
def get_patient_history() -> str:
    """
    Retrieve complete patient history including past symptoms and recommendations.
    
    Returns:
        Formatted patient history or message if not found
    """
    try:
        store = _global_store
        user_id = _global_user_id
        
        if not store:
            return "Error: Memory store not available"
        
        if not user_id:
            return "Error: User ID not provided in context"
        
        namespace = ("PatientDetails",)
        
        # Retrieve data from store
        patient_data = store.get(namespace, user_id)
        
        if not patient_data or not patient_data.value:
            return "No previous patient history found. This appears to be a new patient."
        
        history = PatientHistory.model_validate(patient_data.value)
        
        # Format the history nicely
        formatted = f"""
PATIENT HISTORY SUMMARY
========================
Patient ID: {history.profile.id}
Name: {history.profile.name}
Sex: {history.profile.sex}
Age: {history.profile.age}
Last Updated: {history.last_updated}

CURRENT SYMPTOMS:
{json.dumps([s for s in history.profile.symptoms], indent=2)}

PAST CONSULTATIONS ({len(history.recommendations)}):
"""
        for i, rec in enumerate(history.recommendations, 1):
            formatted += f"""
--- Consultation {i} ({rec.date}) ---
Symptoms at time: {', '.join(rec.symptoms_at_time)}
Possible conditions: {', '.join(rec.possible_conditions)}
Recommendations: {rec.recommendations[:200]}...
"""
        
        return formatted
    
    except Exception as e:
        print(f"Error retrieving patient history: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error retrieving patient history: {str(e)}"


def create_personal_care_agent(store: InMemoryStore, context_schema=Context):
    """
    Create the personal care agent with memory capabilities using create_agent.
    
    Args:
        store: InMemoryStore instance for patient data persistence
        context_schema: Schema for context (includes user_id)
    
    Returns:
        Compiled agent graph with store
    """
    # Set global store reference so tools can access it
    global _global_store
    _global_store = store
    
    # create_agent returns a compiled graph
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


if __name__ == "__main__":
    # Initialize memory store
    memory_store = initialize_memory()
    
    # Test the agent
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
    
    response = run_personal_care_agent(
        test_profile,
        test_query,
        test_passages,
        memory_store
    )
    
    print("\n--- Response ---")
    print(response)
    print("="*80)