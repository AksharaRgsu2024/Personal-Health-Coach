# Personal-Health-Coach

Personalized Health Coach application created using a multi-agent system for conversing with patients on their conditions, and provides lifestyle and health recommendations based on medical information from MedlinePlus.

Inspiration:
Patients can use this app to describe their symptoms, find their possible conditions and get information on treatments, and recommendations for lifestyle changes and doctor consultations. 

## Dataset
This application uses evidence-based health information from MedlinePlus, an open source database of wide range of medical topics from National Library of Medicine.
https://medlineplus.gov/healthtopics.html

## Tech Stack
- Agent Orchestration: Python, LangGraph
- Vector database and Graph RAG: Qdrant + Neo4j Knowledge Graph
- Memory store for long-term patient histories: Sqlite3
- LLM models: Ollama and OpenAI
- Frontend: Streamlit

## Vector database and Knowledge graph Setup
1. **Setup Qdrant instance**
Login to Qdrant Cloud at: https://cloud.qdrant.io/
Create a new cluster, named 'medline_topics'

2. **Setup a Neo4j instance for the knowledge graph**
- Create a Neo4j Aura account if you don't have one, at : https://console.neo4j.io/
- Login to the console, and create a new free instance.
- When the instance is created, Aura will show a generated password (username is typically neo4j); copy or download these credentials as a .txt file
- The instance will move from “creating” to “running” after a short provisioning period

3. **Save environment credentials** 
Create a .env file with the parameters in .env.example file, and replace with your qdrant and Neo4j credentials, and OpenAI API key.
Note: By default, the Application uses local Ollama server for LLM models for the agents. if you are using a different server, specify the server url in the .env file. 

## Instructions to run application
1. Clone the repo

2. Install requirements :
`pip install -r requirements.txt`

3. Populate Vector database (One time only)
Execute :
`python vector_db.py`

4. Populate Neo4j Knowledge graph (One time only)
Execute:
`python Knowledge_graph_creation.py`

5. Execute application in Console mode:
`python health_coach_main.py`

6. To run Streamlit application:
`streamlit run app.py`

### References
https://pmc.ncbi.nlm.nih.gov/articles/PMC8075483/  
   
https://github.com/anusha1219/cypher_query_bot/tree/main
