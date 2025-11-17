import logging
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END

import os
os.environ["OPENAI_API_KEY"] = "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"

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini")

# ---------------------------------------------------
# Logging Configuration (shows what each agent does)
# ---------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


# ---------------------------------------------------
# Modular Typed State Definitions (cleaner + safer)
# ---------------------------------------------------
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

class SafetyState(TypedDict, total=False):
    validated_output: str
    citations: List[str]

class PipelineState(RetrievalState, ConsultState, CareState, SafetyState):
    user_query: str
    final_output: str
    history: List[str]


# ---------------------------------------------------
# AGENT NODE PLACEHOLDERS (to replace later)
# ---------------------------------------------------

def retriever_agent(state: PipelineState) -> PipelineState:
    logging.info("Retriever agent running...")
    try:
        # --- replace with real retriever code later ---
        state["passages"] = [f"[Stub] Retrieved info for: {state['user_query']}"]
        state["urls"] = ["https://medlineplus.gov/"]
    except Exception as e:
        logging.error(f"Retriever error: {e}")
    return state


def consulting_agent(state: PipelineState) -> PipelineState:
    logging.info("Consulting agent running...")
    try:
        # --- replace with real question-asking logic ---
        state["questions"] = ["When did symptoms begin?", "How severe are they?"]
        state["answers"] = ["Symptoms started 2 days ago", "Severity moderate"]

        # Simple dynamic branching example:
        state["needs_more"] = False   # change to True to loop back to consult
    except Exception as e:
        logging.error(f"Consulting error: {e}")
        state["needs_more"] = False
    return state


def personal_care_agent(state: PipelineState) -> PipelineState:
    logging.info("Personal Care agent running...")
    try:
        # --- replace with real summarization logic ---
        state["plan"] = f"[Stub] Care plan based on symptoms + {state['answers']}"
        state["warnings"] = ["Seek urgent help if symptoms worsen."]
    except Exception as e:
        logging.error(f"Care agent error: {e}")
    return state


def safety_agent(state: PipelineState) -> PipelineState:
    logging.info("Safety Critic agent running...")
    try:
        # --- replace with hallucination checks---
        validated = f"SAFE OUTPUT: {state['plan']}"
        state["validated_output"] = validated
        state["citations"] = state.get("urls", [])
        state["final_output"] = validated
    except Exception as e:
        logging.error(f"Safety agent error: {e}")
        state["final_output"] = "[Error in safety agent]"
    return state


def planner_summary_node(state: PipelineState) -> PipelineState:
    logging.info("Planner summary node running...")
    try:
        # ---- Structured Agent-by-Agent Summary ----
        summary_template = f"""
### Pipeline Summary

Below is a structured overview of the information produced by each agent in the multi-agent medical pipeline.

---

## 1. Retriever Agent Output
**Retrieved Passages:**  
{state.get('passages', 'None')}

**Citations / URLs:**  
{state.get('urls', 'None')}

---

## 2. Consulting Agent Output
**Follow-up Questions Asked:**  
{state.get('questions', 'None')}

**Patient Answers:**  
{state.get('answers', 'None')}

**Needs More Questions?**  
{state.get('needs_more', False)}

---

## 3. Personal Care Agent Output
**Generated Care Plan:**  
{state.get('plan', 'None')}

**Warnings:**  
{state.get('warnings', [])}

---

## 4. Safety Critic Agent Output
**Validated (Safe) Output:**  
{state.get('validated_output', 'None')}

**Citations Used:**  
{state.get('citations', [])}

---

## Final Output Delivered to User
{state.get('final_output', 'None')}

---

### Query Processed:
**"{state['user_query']}"**

        """

        #llm here
        polished_summary = llm.invoke(
            f"Format this technical pipeline summary into a clean, readable patient-facing explanation and debugging report:\n\n{summary_template}"
        ).content

        # Store into history
        state["history"].append(polished_summary)

    except Exception as e:
        logging.error(f"Planner summary error: {e}")

    return state


# ---------------------------------------------------
# BRANCHING LOGIC
# ---------------------------------------------------
def need_more_questions(state: PipelineState):
    return "consult" if state.get("needs_more") else "care"


# ---------------------------------------------------
# BUILD THE LANGGRAPH PIPELINE
# ---------------------------------------------------
workflow = StateGraph(PipelineState)

workflow.add_node("retrieve", retriever_agent)
workflow.add_node("consult", consulting_agent)
workflow.add_node("care", personal_care_agent)
workflow.add_node("safety", safety_agent)
workflow.add_node("planner_summary", planner_summary_node)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "consult")

workflow.add_conditional_edges("consult",need_more_questions,{"consult": "consult","care": "care"})

workflow.add_edge("care", "safety")
workflow.add_edge("safety", "planner_summary")
workflow.add_edge("planner_summary", END)

planner_graph = workflow.compile()


# ---------------------------------------------------
# RUN WRAPPER CLASS
# ---------------------------------------------------
class PlannerAgent:
    def __init__(self, graph):
        self.graph = graph

    def run(self, query: str):
        init_state: PipelineState = {
            "user_query": query,
            "final_output": "",
            "history": []
        }
        return self.graph.invoke(init_state)


# ---------------------------------------------------
# EXAMPLE RUN
# ---------------------------------------------------
planner = PlannerAgent(planner_graph)
result = planner.run("I have been feeling dizzy and tired for two days.")

print("\n================ FINAL OUTPUT ================\n")
print(result["final_output"])

print("\n================ DEBUG HISTORY ================\n")
for item in result["history"]:
    print(item)
