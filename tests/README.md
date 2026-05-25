# Tests

Tests are organized per module:

| Path | Description |
|------|-------------|
| `backend/test/` | Backend API benchmarks and demo scripts |
| `data-processing/test/` | Qdrant payload validation tests |

## Running tests

```bash
# Backend demo
cd backend
python test/run_agentic_demo.py --query "a person in red shirt cooking"

# Data-processing unit test
cd data-processing
python -m pytest test/
```
