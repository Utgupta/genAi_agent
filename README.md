# AI Agent (LangGraph)

A minimal conversational AI agent built with [LangGraph](https://github.com/langchain-ai/langgraph), inspired by the freeCodeCamp LangGraph course [`Agent_Bot.py`](https://github.com/iamvaibhavmehra/LangGraph-Course-freeCodeCamp/blob/main/Agents/Agent_Bot.py).

Improvements over the reference:
- **Conversation memory** — the original sent only the latest message and forgot prior turns. This version keeps the full message history in state.
- **Provider switch** — three backends, swap with one env var: [Ollama](https://ollama.com) (free, local, default), [Google Gemini](https://aistudio.google.com) (free cloud), or OpenAI (paid cloud).
- **Graceful exit & error handling** — `exit`, `quit`, or Ctrl+C cleanly leaves; API errors don't crash the loop.

## Architecture

A single-node LangGraph:

```
START -> process -> END
```

- **State**: a list of `HumanMessage` / `AIMessage` objects.
- **`process` node**: sends the full message list to the LLM and appends the reply.
- The REPL loop persists `state["messages"]` across turns to give the agent memory.

## Setup

1. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Pick a provider (see below).

## Running with Ollama (default, free, local)

1. Install Ollama: <https://ollama.com/download>
2. Pull a model:

   ```bash
   ollama pull llama3
   ```

3. Run:

   ```bash
   python Agent_Bot.py
   ```

You can choose a different local model:

```bash
OLLAMA_MODEL=llama3.2 python Agent_Bot.py
```

## Running with Google Gemini (free cloud)

Google AI Studio offers a generous free tier — no billing required, no credit card.

1. Go to <https://aistudio.google.com/apikey> and click **Create API Key**.
2. Copy the key.
3. Edit `.env` and set:

   ```env
   LLM_PROVIDER=gemini
   GEMINI_MODEL=gemini-2.0-flash
   GOOGLE_API_KEY=your-key-here
   ```

4. Run:

   ```bash
   python Agent_Bot.py
   ```

Other free-tier model options for `GEMINI_MODEL`: `gemini-2.5-flash`, `gemini-1.5-flash`, `gemini-2.0-flash-lite`. Free tier currently allows ~15 requests/minute and ~1500 requests/day per model.

## Running with OpenAI

1. Add billing & credits at <https://platform.openai.com/settings/organization/billing/overview>.
2. Get an API key at <https://platform.openai.com/api-keys>.
3. Configure: edit `.env` and set:

   ```env
   LLM_PROVIDER=openai
   OPENAI_MODEL=gpt-4o
   OPENAI_API_KEY=sk-your-real-key
   ```

4. Run:

   ```bash
   python Agent_Bot.py
   ```

## Configuration reference

| Variable | Default | Used when |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | always |
| `OLLAMA_MODEL` | `llama3` | provider = `ollama` |
| `GEMINI_MODEL` | `gemini-2.0-flash` | provider = `gemini` |
| `GOOGLE_API_KEY` | _(none)_ | provider = `gemini` |
| `OPENAI_MODEL` | `gpt-4o` | provider = `openai` |
| `OPENAI_API_KEY` | _(none)_ | provider = `openai` |

## Example session

```
AI Agent ready (provider=ollama, model=llama3).
Type 'exit' or 'quit' to end the chat.

You: My name is Utkarsh and I like LangGraph.

AI: Nice to meet you, Utkarsh! It's great that you're interested in LangGraph...

You: What is my name and what do I like?

AI: According to our conversation, your name is Utkarsh, and you like LangGraph!
```

## Files

- `Agent_Bot.py` – the agent and chat loop
- `requirements.txt` – Python dependencies
- `.env` – local config (provider, model, API keys)
- `.gitignore` – keeps `.env` and `.venv` out of git
