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
$ ollama pull minimax-m2.7:cloud
$ ollama pull kimi-k2-thinking:cloud
$ ollama pull qwen3.5:2b
$ ollama pull deepseek-v3.2:cloud

// run in local
$ uv run main.py

// run in web
$ uvicorn main:app --host 127.0.0.1 --port 7932
```

```zsh
// copy .env file
$ cp .env.sample .env
```
