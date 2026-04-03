# rag-china-travel-ai-agent

rag-china-travel-ai-agent

## Tech Stack

- python
- uv
- Pydantic AI
- ollama

## Requirement

- install python (v3.14)

```zsh
// install dependencies
$ uv sync

// ollama pull model
$ ollama pull qwen3.5:2b

// run in local
$ uv run main.py

// run in web
$ uvicorn main:app --host 127.0.0.1 --port 7932
```

```zsh
// copy .env file
$ cp .env.sample .env
```
