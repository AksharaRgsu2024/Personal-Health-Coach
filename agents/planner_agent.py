from consulting_agent import build_consulting_agent_graph
import logging
from typing import TypedDict, List
from langgraph.graph import StateGraph, END

import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage


llm = ChatOpenAI(model="gpt-4o-mini")
consult_graph = build_consulting_agent_graph()

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


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
    history: List[str]
    final_output: str
    patient_profile: dict
    messages: list
    turn_count: int
    last_question: str


def retriever_agent(state: PipelineState) -> PipelineState:
    logging.info("Retriever agent running...")
    state["passages"] = [f"[Stub] Retrieved info for: {state['user_query']}"]
    state["urls"] = ["https://medlineplus.gov/"]
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
    logging.info("Personal Care agent running...")
    answers = state.get("answers") or []
    state["plan"] = f"[Stub] Care plan based on symptoms + {answers}"
    state["warnings"] = ["Seek urgent help if symptoms worsen."]
    return state


def safety_agent(state: PipelineState) -> PipelineState:
    logging.info("Safety Critic agent running...")
    validated = f"SAFE OUTPUT: {state['plan']}"
    state["validated_output"] = validated
    state["citations"] = state.get("urls", [])
    state["final_output"] = validated
    return state


def planner_summary_node(state: PipelineState) -> PipelineState:
    logging.info("Planner summary node running...")
    summary = (
        f"Query: {state['user_query']}\n"
        f"Questions: {state.get('questions')}\n"
        f"Answers: {state.get('answers')}\n"
        f"Plan: {state.get('plan')}\n"
        f"Final output: {state.get('final_output')}\n"
    )
    history = state.get("history", [])
    history.append(summary)
    state["history"] = history
    return state


def route_from_consult(state: PipelineState):
    if state.get("needs_more"):
        return "followup"
    else:
        return "care"


workflow = StateGraph(PipelineState)

workflow.add_node("retrieve", retriever_agent)
workflow.add_node("consult", consulting_agent)
workflow.add_node("followup", followup_node)
workflow.add_node("care", personal_care_agent)
workflow.add_node("safety", safety_agent)
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
workflow.add_edge("care", "safety")
workflow.add_edge("safety", "planner_summary")
workflow.add_edge("planner_summary", END)

planner_graph = workflow.compile()


class PlannerAgent:
    def __init__(self, graph):
        self.graph = graph
        self.state: PipelineState | None = None

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
        out = self.graph.invoke(self.state)
        self.state = out
        return self._analyze(self.state)

    def continue_with_answer(self, answer: str):
        if not self.state:
           return {"type": "final", "output": "[No active conversation]"}

    # 1. Add the user's answer into message history
        msgs = self.state.get("messages", [])
        msgs.append(HumanMessage(content=answer))
        self.state["messages"] = msgs

    # 2. DO NOT replace user_query. It remains the original problem.
    # 3. Re-invoke graph
        out = self.graph.invoke(self.state)
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
