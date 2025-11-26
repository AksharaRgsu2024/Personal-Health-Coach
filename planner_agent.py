from agents.medline_retriever import retrieve_passages
from agents.consulting_agent_with_memory import build_consulting_agent_graph, PatientProfile
from agents.personal_care_agent_kg import create_personal_care_agent
import json
import re
import logging
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# llm = ChatOpenAI(model="gpt-4o-mini")
llm = ChatOllama(
    base_url="http://10.230.100.240:17020/", #http://localhost:11434", #"http://10.230.100.240:17020/"
    model="gpt-oss:20b",#"llama3.1:latest",
    temperature=0.3
)
consult_graph = build_consulting_agent_graph()

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
    care_details: dict  # structured parsed plan sections

class PipelineState(RetrievalState, ConsultState, CareState):
    user_query: str
    history: List[str]
    final_output: str
    patient_profile: PatientProfile
    messages: list
    turn_count: int
    last_question: str


# Create the Personal Care LLM agent once (re-used across calls)
care_llm_agent = create_personal_care_agent()


def _parse_care_output(text: str) -> dict:
    """
    Naively parse the personal care agent output into sections.
    We look for headings like:
      - POSSIBLE CONDITIONS:
      - LIFESTYLE RECOMMENDATIONS:
      - POSSIBLE TREATMENTS:
      - NEXT STEPS:
      - REFERENCES:
    and split their content into bullet-style lists.
    """
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
def retriever_agent(state: PipelineState) -> PipelineState:
    logging.info("Retriever agent running with MedlinePlus embeddings...")

    query = state["user_query"]

    # If consulting already added symptoms, we can enrich the query
    profile = state.get("patient_profile") or {}

    if type(profile)==dict:
        symptoms = profile.get("symptoms") or []
    elif type(profile)==PatientProfile:
        symptoms=profile.symptoms
    # if symptoms and type(symptoms)==list:
    
    #     query += " " + " ".join(symptoms)
    # elif type(symptoms)==dict:
    #     symptoms_list=list(symptoms.items())
    #     query+=" "+ ', '.join(f"({key},{value})" for key, value in symptoms_list)
    query+=" "+ ', '.join(f"({key}:{value})" for symptom in symptoms for key, value in symptom.items())

    results = retrieve_passages(query, top_k=6)

    state["passages"] = [r["text"] for r in results]
    state["urls"] = [r["url"] for r in results]

    return state


def consulting_agent(state: PipelineState) -> PipelineState:
    logging.info("Consulting agent (teammate) running via planner...")

    retrieved_docs = [
        {
            "title": p[:60],
            "summary": p,
            "url": u,
        }
        for p, u in zip(state.get("passages", []), state.get("urls", []))
    ]

    agent_state = {
        "user_query": state["user_query"],
        "messages": state.get("messages", []),
        "retrieved_docs": retrieved_docs,
        "patient_profile": state.get("patient_profile", {}),
        "turn_count": state.get("turn_count", 0),
        "need_more_info": True,
        "red_flag": False,
        "last_question": state.get("last_question"),
    }

    updated = consult_graph.invoke(agent_state)

    print("\n===== DEBUG CONSULT OUTPUT =====")
    print(updated)
    print("================================\n")

    assistant_out = updated.get("assistant_output", {}) or {}
    followup_q = assistant_out.get("followup_question")
    explanation = assistant_out.get("explanation", "")

    state["questions"] = [followup_q] if followup_q else []
    state["answers"] = [explanation] if explanation else []
    state["patient_profile"] = updated.get("patient_profile", {})
    state["messages"] = updated.get("messages", [])
    state["turn_count"] = updated.get("turn_count", 1)
    state["needs_more"] = bool(updated.get("need_more_info"))
    state["last_question"] = updated.get("last_question")

    return state


def followup_node(state: PipelineState) -> PipelineState:
    logging.info("Follow-up node: stopping after consulting to wait for user answer.")
    return state


def personal_care_agent(state: PipelineState) -> PipelineState:
    logging.info("Personal Care agent running with KG + MedlinePlus...")

    # 1. Gather context for the personal care agent
    patient_profile = state.get("patient_profile", {}) or {}
    user_query = state.get("user_query", "")
    passages = state.get("passages", [])

    refs_text = ""
    if passages:
        refs_text = "\n".join(f"- {p}" for p in passages[:6])

    user_content = (
        "Here is the patient's situation.\n\n"
        f"Original concern:\n{user_query}\n\n"
        "Structured symptom info from earlier consulting:\n"
        f"{json.dumps(patient_profile.model_dump_json(), indent=2)}\n\n"
        "Relevant reference snippets from MedlinePlus:\n"
        f"{refs_text}\n\n"
        "Please follow your instructions as a Personal Healthcare Coach, "
        "using the knowledge graph and this context."
    )

    # 2. Call the personal care LLM + KG agent
    try:
        result = care_llm_agent.invoke(
            {"messages": [{"role": "user", "content": user_content}]}
        )
    except Exception as e:
        logging.error(f"Personal care agent failed: {e}")
        state["plan"] = (
            "Sorry, I had trouble generating a detailed care plan right now. "
            "Please consider speaking directly with a healthcare professional."
        )
        state["warnings"] = ["System error during care planning step."]
        state["care_details"] = {}
        return state

    # 3. Extract the assistant's final message text
    messages = result.get("messages", []) if isinstance(result, dict) else []

    care_text = ""
    for m in reversed(messages):
        role = getattr(m, "type", None) or getattr(m, "role", None)
        if role in ("ai", "assistant"):
            care_text = m.content if isinstance(m.content, str) else str(m.content)
            break

    if not care_text and messages:
        m = messages[-1]
        care_text = m.content if isinstance(m.content, str) else str(m.content)

    if not care_text:
        care_text = (
            "I could not generate a detailed plan, but based on your symptoms, "
            "please monitor your condition and seek medical help if things worsen."
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

    # Final output: just use the care plan text (could be extended later)
    state["final_output"] = state.get("plan") or "[No care plan was generated.]"
    return state


def route_from_consult(state: PipelineState):
    if state.get("needs_more"):
        return "followup"
    else:
        return "care"


# ----------------- Build the graph -----------------
workflow = StateGraph(PipelineState)

workflow.add_node("retrieve", retriever_agent)
workflow.add_node("consult", consulting_agent)
workflow.add_node("followup", followup_node)
workflow.add_node("care", personal_care_agent)
workflow.add_node("planner_summary", planner_summary_node)

workflow.set_entry_point("retrieve")
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

#Add short-term memory
# checkpointer = InMemorySaver()
planner_graph = workflow.compile()


# ----------------- Console wrapper -----------------
class PlannerAgent:
    def __init__(self, graph):
        self.graph = graph
        self.state: PipelineState | None = None
        self.thread_config={"configurable": {"thread_id": "1"}}
                            
    def start(self, query: str):
        self.state = {
            "user_query": query,
            "history": [],
            "final_output": "",
            "patient_profile": {},
            "messages": [],
            "turn_count": 0,
            "last_question": "",
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


if __name__ == "__main__":
    print("\n🔵 Personal Health Coach — Console Mode\n")
    planner = PlannerAgent(planner_graph)

    user_input = input("User: ")
    step = planner.start(user_input)

    while True:
        if step["type"] == "final":
            print("\nFINAL OUTPUT:")
            print(step["output"])
            break

        elif step["type"] == "followup":
            print("\nConsulting Agent:")
            print("Explanation:", step["explanation"])
            print("Follow-up question:", step["question"])
            answer = input("\nYour answer: ")
            step = planner.continue_with_answer(answer)