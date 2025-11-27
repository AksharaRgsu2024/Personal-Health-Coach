from neo4j import GraphDatabase
from neo4j.exceptions import CypherSyntaxError
from openai import OpenAI
from typing import TypedDict, Annotated, Literal, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import json
import openai
from langgraph.prebuilt import ToolNode
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from pydantic import BaseModel
from langchain.tools import ToolRuntime
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage
from langchain_ollama import ChatOllama

from langgraph.prebuilt import InjectedState
from typing import Deque, List, Optional, Tuple
from langgraph.store.memory import InMemoryStore
from langchain.embeddings import init_embeddings
from langchain_ollama import OllamaEmbeddings
from dataclasses import dataclass
from .consulting_agent_with_memory import PatientProfile
from datetime import datetime

# Demo database credentials
from dotenv import load_dotenv
import os
load_dotenv()
URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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

class Neo4jOllamaQuery:
    """
    Neo4j Cypher query generator using Ollama LLM.
    Replaces OpenAI with local Ollama models for query generation.
    """
    
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

    def __init__(
        self, 
        url: str, 
        user: str, 
        password: str,
        ollama_base_url: str = "http://localhost:11434",
        ollama_model: str = "llama3.1:latest"
    ):
        """
        Initialize Neo4j connection and Ollama LLM.
        
        Args:
            url: Neo4j database URL
            user: Neo4j username
            password: Neo4j password
            ollama_base_url: Ollama server URL (default: http://localhost:11434)
            ollama_model: Ollama model name (default: llama3.1:latest)
        """
        self.driver = GraphDatabase.driver(url, auth=(user, password))
        
        # Initialize Ollama LLM
        self.llm = ChatOllama(
            base_url=ollama_base_url,
            model=ollama_model,
            temperature=0.0,
        )
        
        self.schema = self.generate_schema()
        self.check_connection()

    def check_connection(self):
        """Verify Neo4j database connection."""
        try:
            self.driver.verify_connectivity()
            print("✓ Neo4j connection successful!")
        except Exception as e:
            print(f"✗ Failed to connect to Neo4j: {e}")

    def generate_schema(self) -> str:
        """Generate database schema from Neo4j metadata."""
        node_props = self.query_database(self.node_properties_query)
        rel_props = self.query_database(self.rel_properties_query)
        rels = self.query_database(self.rel_query)
        return schema_text(node_props, rel_props, rels)

    def refresh_schema(self):
        """Refresh the database schema."""
        self.schema = self.generate_schema()

    def get_system_message(self) -> str:
        """Get the system prompt for Cypher query generation."""
        return f"""
Task: Generate Cypher queries to query a Neo4j graph database based on the provided schema definition.

Instructions:
- Use only the provided relationship types and properties
- Do not use any other relationship types or properties that are not provided
- If you cannot generate a Cypher statement based on the provided schema, explain the reason
- Return ONLY the Cypher query, no explanations, no markdown, no code blocks

Schema:
{self.schema}

IMPORTANT: 
- Return only valid Cypher syntax
- Do not include ```cypher or ``` markers
- Do not include any explanations or apologies
- Just the raw Cypher query
"""

    def query_database(self, neo4j_query: str, params: Dict = None) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query on Neo4j database.
        
        Args:
            neo4j_query: Cypher query string
            params: Query parameters
            
        Returns:
            List of result dictionaries
        """
        if params is None:
            params = {}
            
        with self.driver.session() as session:
            result = session.run(neo4j_query, params)
            output = result.data()
            return output

    def construct_cypher(self, question: str, history: Optional[List[Dict]] = None) -> str:
        """
        Generate a Cypher query from natural language question using Ollama.
        
        Args:
            question: Natural language question
            history: Conversation history for retry attempts
            
        Returns:
            Generated Cypher query string
        """
        # Build message list for LangChain
        messages = [
            SystemMessage(content=self.get_system_message()),
            HumanMessage(content=question),
        ]
        
        # Add history if provided (for retry attempts)
        if history:
            for msg in history:
                if msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))
                elif msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
        
        # Invoke Ollama LLM
        response = self.llm.invoke(messages)
        
        # Extract content
        cypher = response.content if hasattr(response, 'content') else str(response)
        
        # Clean up the response - remove markdown code blocks if present
        cypher = cypher.strip()
        if cypher.startswith("```cypher"):
            cypher = cypher[9:]
        elif cypher.startswith("```"):
            cypher = cypher[3:]
        if cypher.endswith("```"):
            cypher = cypher[:-3]
        cypher = cypher.strip()
        
        return cypher

    def run(self, question: str, history: Optional[List[Dict]] = None, retry: bool = True) -> Dict[str, Any]:
        """
        Run the complete workflow: generate Cypher query and execute it.
        
        Args:
            question: Natural language question
            history: Conversation history for retry attempts
            retry: Whether to retry on syntax errors
            
        Returns:
            Dictionary with 'cypher query' and 'result' keys
        """
        query_result = {'cypher query': '', 'result': ''}
        
        # Generate Cypher query
        cypher = self.construct_cypher(question, history)
        query_result['cypher query'] = cypher
        
        print(f"\n{'='*60}")
        print("Generated Cypher Query:")
        print(f"{'='*60}")
        print(cypher)
        print(f"{'='*60}\n")
        
        try:
            # Execute query
            query_result['result'] = self.query_database(cypher)
            
        except CypherSyntaxError as e:
            # Self-healing flow
            if not retry:
                print(f"✗ Cypher syntax error (no retry): {str(e)}")
                return {
                    'cypher query': cypher,
                    'result': [],
                    'error': f"Invalid Cypher syntax: {str(e)}"
                }
            
            print(f"⚠️  Cypher syntax error, retrying with correction...")
            print(f"Error: {str(e)}")
            
            # Retry with error feedback
            retry_result = self.run(
                question,
                history=[
                    {"role": "assistant", "content": cypher},
                    {
                        "role": "user",
                        "content": f"""This query returns an error: {str(e)} 

Give me an improved query that works without any explanations or apologies.
Return ONLY the corrected Cypher query."""
                    },
                ],
                retry=False
            )
            
            query_result['cypher query'] = retry_result.get('cypher query', cypher)
            query_result['result'] = retry_result.get('result', [])
            if 'error' in retry_result:
                query_result['error'] = retry_result['error']
        
        except Exception as e:
            print(f"✗ Unexpected error: {str(e)}")
            query_result['error'] = str(e)
            query_result['result'] = []
        
        return query_result
    
    def close(self):
        """Close Neo4j driver connection."""
        self.driver.close()
        print("✓ Neo4j connection closed")


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
        # neo4j_gpt = Neo4jGPTQuery(URI, USER, PASSWORD, OPENAI_API_KEY)
        neo4j_ollama = Neo4jOllamaQuery(
            url=URI,
            user=USER,
            password=PASSWORD,
            ollama_base_url="http://10.230.100.240:17020/",  # Your server
            ollama_model="gpt-oss:20b"  # Your model
        )

        result = neo4j_ollama.run(query)
        retrieved_nodes = result.get('result', '')
        cypher_query = result.get('cypher query', '')
        
        print(f"Generated Cypher Query:\n{cypher_query}\n")
        print(f"Retrieved {len(retrieved_nodes)} nodes")
        print(f"Debug output:\n{retrieved_nodes}")
        
        return retrieved_nodes
    
    except Exception as e:
        print(f"Error querying Neo4j: {str(e)}")
        return {"messages": "Error"}


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
    last_updated: str = Field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class Context:
    user_id: str


def initialize_memory():
    """Initialize the in-memory store with embeddings"""
    embeddings = OllamaEmbeddings(
        model="embeddinggemma:300m",
        base_url="http://127.0.0.1:11434"
    )

    in_memory_store = InMemoryStore(
        index={
            "embed": embeddings.embed_documents,
            "dims": 768,
        }
    )
    return in_memory_store


@tool
def save_patient_history(
    patient_profile: dict,
    recommendation: dict,
    runtime: ToolRuntime[Context]
) -> str:
    """
    Save or update patient history with new recommendation.
    
    Args:
        patient_profile: Current patient profile dictionary
        recommendation: New recommendation to add (dict with date, conditions, recommendations, symptoms)
        runtime: Tool runtime with store and context
    
    Returns:
        Success message
    """
    store = runtime.store
    
    # Handle case where context might be None
    if runtime.context is None:
        return "Error: User context not provided. Cannot save patient history."
    
    user_id = runtime.context.user_id
    namespace = ("PatientDetails",)
    
    # Try to get existing history
    existing = store.get(namespace, user_id)
    
    if existing and existing.value:
        # Update existing history
        try:
            history = PatientHistory.model_validate(existing.value)
        except Exception as e:
            print(f"Error loading existing history: {e}, creating new")
            history = PatientHistory(profile=PatientProfile(**patient_profile))
    else:
        # Create new history
        history = PatientHistory(profile=PatientProfile(**patient_profile))
    
    # Update profile with latest information
    history.profile = PatientProfile(**patient_profile)
    
    # Add new recommendation
    new_recommendation = PatientRecommendation(**recommendation)
    history.recommendations.append(new_recommendation)
    history.last_updated = datetime.now().isoformat()
    
    # Save to store
    store.put(namespace, user_id, history.model_dump())
    
    return f"Successfully saved patient history. Total consultations: {len(history.recommendations)}"


@tool
def get_patient_history(runtime: ToolRuntime[Context]) -> str:
    """
    Retrieve complete patient history including past symptoms and recommendations.
    
    Args:
        runtime: Tool runtime with store and context
    
    Returns:
        Formatted patient history or message if not found
    """
    store = runtime.store
    
    # Handle case where context might be None
    if runtime.context is None:
        return "Error: User context not provided. Cannot retrieve patient history."
    
    user_id = runtime.context.user_id
    namespace = ("PatientDetails",)
    
    # Retrieve data from store
    patient_data = store.get(namespace, user_id)
    
    if not patient_data or not patient_data.value:
        return "No previous patient history found. This appears to be a new patient."
    
    try:
        history = PatientHistory.model_validate(patient_data.value)
        
        # Format the history nicely
        formatted = f"""
PATIENT HISTORY SUMMARY
========================
Patient ID: {history.profile.id}
Name: {history.profile.name}
Sex: {history.profile.sex}
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
        return f"Error retrieving patient history: {str(e)}"


medical_agent_system_prompt = """
You are a Personal Healthcare Coach with access to long-term patient memory.

WORKFLOW:
1. ALWAYS start by using the get_patient_history tool to check if this patient has visited before.
   - Review past symptoms, conditions, and recommendations
   - Consider how current symptoms relate to past issues
   - Note any recurring patterns or chronic conditions

2. From the patient's current profile, extract and structure symptoms:
   - Primary symptoms with duration and severity
   - Patient demographics (age, sex, etc.)
   - Compare with historical symptoms if available

3. Use the query_medical_knowledge_graph tool to retrieve relevant medical information based on current symptoms.

4. Generate comprehensive recommendations considering:
   - Current symptoms and retrieved medical knowledge
   - Past medical history and recommendations
   - Any progression or changes in condition over time

5. Structure your response with these sections:
   - PATIENT CONTEXT: Briefly note if returning patient and relevant history
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
- Only use information from the knowledge graph - no external sources
- Always emphasize this is informational, not professional medical advice
- Be especially attentive to recurring or worsening symptoms in returning patients
- Consider medication interactions or conflicts with past treatments
- Format responses clearly with section headers

MEMORY USAGE:
- First action: get_patient_history
- Last action: save_patient_history
- Use history to provide continuity of care
"""


def create_personal_care_agent(store: InMemoryStore, context_schema=Context):
    """Create the personal care agent with memory capabilities"""
    agent = create_agent(
        model=llm_model,
        tools=[query_medical_knowledge_graph, get_patient_history, save_patient_history],
        system_prompt=medical_agent_system_prompt,
        store=store,
        context_schema=context_schema
    )
    return agent


if __name__ == "__main__":
    # Initialize memory store
    memory_store = initialize_memory()
    
    # Create agent with memory
    agent = create_personal_care_agent(store=memory_store, context_schema=Context)
    
    # Simulate patient visits
    user_id = "patient_001"
    
    # Create context for this user
    context = Context(user_id=user_id)
    
    # First visit
    print("="*80)
    print("FIRST VISIT")
    print("="*80)
    
    patient_input_1 = """
    Patient Profile:
    - Name: John Doe
    - ID: patient_001
    - Sex: Male
    - Symptoms:
      * Weight loss (15 lbs over 3 months)
      * Increased thirst
      * Frequent urination
      * Fatigue
    """
    
    response = agent.invoke(
        {"messages": [{"role": "user", "content": patient_input_1}]},
        config={"configurable": {"context": context}}
    )
    
    print("\n--- First Visit Response ---")
    for m in response["messages"]:
        m.pretty_print()
    
    # Second visit - same patient returns
    print("\n" + "="*80)
    print("FOLLOW-UP VISIT (2 weeks later)")
    print("="*80)
    
    patient_input_2 = """
    Patient Profile:
    - Name: John Doe
    - ID: patient_001
    - Sex: Male
    - Symptoms:
      * Still experiencing frequent urination
      * Thirst has improved slightly
      * New symptom: Blurred vision
      * Weight stable now
    """
    
    response = agent.invoke(
        {"messages": [{"role": "user", "content": patient_input_2}]},
        config={"configurable": {"context": context}}
    )
    
    print("\n--- Follow-up Visit Response ---")
    for m in response["messages"]:
        m.pretty_print()