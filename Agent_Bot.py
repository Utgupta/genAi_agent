"""
A simple conversational AI agent built with LangGraph.

Based on the freeCodeCamp LangGraph course example:
https://github.com/iamvaibhavmehra/LangGraph-Course-freeCodeCamp/blob/main/Agents/Agent_Bot.py

Improvements over the original:
- Full conversation memory (the original forgot every previous turn).
- Provider switch: run locally with Ollama (free, no key) or with OpenAI.
- Graceful exit on 'exit' / 'quit' / Ctrl+C, plus error handling.

Configuration (via .env or shell env):
    LLM_PROVIDER   = "ollama" (default) | "openai" | "gemini"
    OLLAMA_MODEL   = "llama3"             (provider=ollama)
    OPENAI_MODEL   = "gpt-4o"             (provider=openai)
    OPENAI_API_KEY = required             (provider=openai)
    GEMINI_MODEL   = "gemini-2.5-flash"   (provider=gemini)
    GOOGLE_API_KEY = required             (provider=gemini)
"""

import os
from typing import TypedDict, List, Union

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END

load_dotenv()


class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]


def _build_llm():
    """Return a chat LLM based on LLM_PROVIDER (defaults to Ollama)."""
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        model = os.getenv("OPENAI_MODEL", "gpt-4o")
        return ChatOpenAI(model=model)

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        model = os.getenv("OLLAMA_MODEL", "llama3")
        return ChatOllama(model=model)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        return ChatGoogleGenerativeAI(model=model)

    raise ValueError(
        f"Unknown LLM_PROVIDER={provider!r}. Use 'ollama', 'openai', or 'gemini'."
    )


llm = _build_llm()


def process(state: AgentState) -> AgentState:
    """Send the running message history to the LLM and append its reply."""
    response = llm.invoke(state["messages"])
    state["messages"].append(AIMessage(content=response.content))
    print(f"\nAI: {response.content}")
    return state


graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()


def run_chat() -> None:
    """Interactive REPL loop. Type 'exit' or 'quit' to leave."""
    conversation: List[BaseMessage] = []

    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    defaults = {
        "ollama": ("OLLAMA_MODEL", "llama3"),
        "openai": ("OPENAI_MODEL", "gpt-4o"),
        "gemini": ("GEMINI_MODEL", "gemini-2.5-flash"),
    }
    model_env, default_model = defaults.get(provider, ("OLLAMA_MODEL", "llama3"))
    model_name = os.getenv(model_env, default_model)
    print(f"AI Agent ready (provider={provider}, model={model_name}).")
    print("Type 'exit' or 'quit' to end the chat.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        conversation.append(HumanMessage(content=user_input))

        try:
            result = agent.invoke({"messages": conversation})
            conversation = result["messages"]
        except Exception as err:
            print(f"\n[Error] {err}\n")


if __name__ == "__main__":
    run_chat()
