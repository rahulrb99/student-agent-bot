# Simple SE Tutor Chatbot

A local web UI for undergrad software-engineering tutoring.

## Topics

1. Use Case Diagrams
2. Software Requirements
3. UML Class Diagrams
4. Agile & Scrum Basics

## Flow

1. Pick model (Groq or OpenAI) and topic
2. Tutor sends W3Schools / GeeksforGeeks links — you read and come back
3. Tutor asks a few simple questions (with examples)
4. Guided practice (e.g. build a DoorDash use case diagram step by step)
5. Final summary when you're confident or have no doubts

## Setup

From the repo root, make sure `env` has your API keys:

```
GROQ_API_KEY=...
OPENAI_API_KEY=...
```

Install deps (from repo root):

```bash
uv sync
```

## Run

```bash
cd simple_chat
uv run uvicorn app:app --reload --port 8080
```

Open **http://127.0.0.1:8080**

## Features

- **Model**: Groq or OpenAI — chosen once at startup
- **Manual mode**: type your own answers
- **Simulate student**: button generates a student reply (confused / missing context / overconfident)
- **Switch personality** anytime from the chat toolbar

## Session lifecycle (Option 3 — soft end)

| State | What happens |
|-------|----------------|
| **Active** | Read → quiz → guided practice |
| **Complete** | Lesson done; up to 3 follow-up questions; say "no doubts" to close |
| **Ended** | Chat locked; download transcript or reset |

## Plug-and-play topics

- **4 built-in topics** in `data/topics/builtin/` (read-only)
- **Add a topic:** enter name only → Firecrawl finds W3Schools + GeeksforGeeks → LLM drafts lesson → professor reviews/edits → Save
- Needs `FIRECRAWL_API_KEY` + `GROQ_API_KEY` or `OPENAI_API_KEY` in `env`
- Custom topics saved to `data/topics/custom/` (deletable)
