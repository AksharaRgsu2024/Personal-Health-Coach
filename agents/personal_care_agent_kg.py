from neo4j import GraphDatabase
from neo4j.exceptions import CypherSyntaxError
from openai import OpenAI
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
# from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import json
import openai
from langgraph.prebuilt import ToolNode
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from pydantic import BaseModel
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage

from langgraph.prebuilt import InjectedState
from typing import Deque, List, Optional, Tuple
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
        # construct schema
        self.schema = self.generate_schema()
        self.check_connection()

    def check_connection(self):
        try:
            self.driver.verify_connectivity()
            print("Connection successful!")
        except Exception as e:
            print(f"Failed to connect to Neo4j: {e}")

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
            # output = [r.values() for r in result]
            # output.insert(0, result.keys())
            output=result.data()
            return output

    def construct_cypher(self, question, history=None):
        messages = [
            {"role": "system", "content": self.get_system_message()},
            {"role": "user", "content": question},
        ]
        # Used for Cypher healing flows
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
        query_result={'cypher query':'', 'result':''}
        # Construct Cypher statement
        cypher = self.construct_cypher(question, history)
        query_result['cypher query']=cypher
        # print(cypher)
        try:
            query_result['result']=self.query_database(cypher)
            
        # Self-healing flow
        except CypherSyntaxError as e:
            # If out of retries
            if not retry:
              return "Invalid Cypher syntax"
        # Self-healing Cypher flow by
        # providing specific error to GPT-4
            print("Retrying")
            query_result['result']= self.run(
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
        return query_result
    
class MedicalAgentState(TypedDict):
    """State for the medical recommendation agent."""
    patient_info: str  # Patient's symptoms and information
    messages: list  # Conversation history
    symptom_analysis: dict  # Extracted symptoms
    conditions: list  # Possible conditions from knowledge graph
    recommendations: dict  # Final recommendations
    kg_query_result: str  # Raw result from knowledge graph

@tool
def query_medical_knowledge_graph(symptoms_dict: str) -> list[dict]:
    """
    Query the Neo4j medical knowledge graph to find conditions associated with patient symptoms.
    
    Args:
        patient_symptoms: Description of patient symptoms (e.g., "fever, cough, headache")
        neo4j_chain: Configured GraphCypherQAChain instance
        
    Returns:
        List of dictionaries containing conditions and related medical information
    """
    
    # Enhanced query with clearer instructions
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
        # result = neo4j_chain.invoke({"query": query})
        neo4j_gpt = Neo4jGPTQuery(URI, USER, PASSWORD, OPENAI_API_KEY)
        result = neo4j_gpt.run(query)
        # Extract intermediate steps for debugging
        # intermediate_steps = result.get("intermediate_steps", [])
        retrieved_nodes=result.get('result','')
        cypher_query=result.get('cypher query','')
        # database_results = []
        
        # if intermediate_steps:
        #     if len(intermediate_steps) > 0 and "query" in intermediate_steps[0]:
        #         cypher_query = intermediate_steps[0]["query"]
        #     if len(intermediate_steps) > 1 and "context" in intermediate_steps[1]:
        #         database_results = intermediate_steps[1]["context"]
        
        print(f"Generated Cypher Query:\n{cypher_query}\n")
        # print(f"Database Results: {database_results}\n")
        # print(f"Final Answer: {result.get('result', '')}\n")
        print(f"Retrieved {len(retrieved_nodes)} nodes")
        # return {
        #     "success": True,
        #     "result": retrieved_nodes,
        #     "cypher_query": cypher_query,
        #     "raw_data": result
        # }
        # return {"messages": retrieved_nodes}
        return retrieved_nodes
    
    except Exception as e:
        print(f"Error querying Neo4j: {str(e)}")
        return {"messages":"Error"}



medical_agent_system_prompt="""
    You are a Personal Healthcare Coach. 
    Instructions:
    1.From the patient's input, Extract and structure the key symptoms and medical information. Identify:
    - Primary symptoms and relevant terms or synonyms
    - Duration and severity
    - Additional context

    2. Use the query_medical_knowledge_graph tool to retrieve relevant topics from the patient's symptoms with their details.
    3. Based on the retrieved information from the medical knowledge graph, generate comprehensive recommendations:
    - POSSIBLE CONDITIONS: List the most likely conditions with brief explanations
    - LIFESTYLE RECOMMENDATIONS: Provide actionable lifestyle modifications
    - POSSIBLE TREATMENTS: Suggest treatment options (mention diagnostic tests, medications, therapies)
    - NEXT STEPS: Recommend when to seek medical care
    - REFERENCES: URL links for the relevant health topics from the knowledge graph for further information
    
    Important: Always emphasize that this is informational and not a substitute for professional medical advice.
    Format your response in clear, structured sections.
"""

def create_personal_care_agent():
    basic_model = ChatOpenAI(model="gpt-4o-mini")
    agent=create_agent(model=basic_model, tools=[query_medical_knowledge_graph], system_prompt=medical_agent_system_prompt)
    return agent

if __name__=="__main__":
    # neo4j_gpt = Neo4jGPTQuery(URI, USER, PASSWORD, OPENAI_API_KEY)
    # response = neo4j_gpt.run("List the conditions associated with losing weight")
    # print(response)
    # response=query_medical_knowledge_graph("I have dizziness and trouble balancing after I wake up in the morning, and headache")
    # for res in response['result']:
    #     print(res)
    agent = create_personal_care_agent()
    patient_input = """
    I have been losing weight and have sugar cravings 
    """

    # Run the agent
    print("Starting Medical Recommendation Agent...")
    print(f"Patient Input: {patient_input.strip()}\n")

    response = agent.invoke( {"messages": [{"role": "user", "content": patient_input}]})
    for m in response["messages"]:
        m.pretty_print()