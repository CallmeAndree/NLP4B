# Configuration Files

This directory contains configuration templates for the NLP4B system.

## Files

| File | Description |
|------|-------------|
| `.env.example` | Environment variable template. Copy to `backend/.env` or `data-processing/.env` and fill in your credentials. |

## Per-module configs

- `frontend/.streamlit/config.toml` — Streamlit theme and server settings
- `azure-ai-provider/docker-compose.yml` — Docker Compose for the embedding service
