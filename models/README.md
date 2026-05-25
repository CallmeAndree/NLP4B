# Models

Pre-trained models used in this project are downloaded at runtime from HuggingFace Hub. No model weights are committed to the repository.

## Models Used

| Model | Task | Dimensions | Source |
|-------|------|-----------|--------|
| `google/siglip-so400m-patch14-384` | Visual embedding | 1152 | HuggingFace |
| `BAAI/bge-m3` | Semantic text embedding | 1024 | HuggingFace |
| `Qdrant/bm25` | Sparse text embedding | sparse | fastembed |
| `ultralytics/yolov8` | Object detection | — | Ultralytics |
| `microsoft/Florence-2-large` | Dense captioning | — | HuggingFace |
| TransNetV2 | Shot boundary detection | — | bundled weights |

## Pre-downloading (optional)

```bash
pip install huggingface-hub
huggingface-cli download BAAI/bge-m3
huggingface-cli download google/siglip-so400m-patch14-384
```

When using Docker, models are cached in a host-mounted volume (`~/.cache/huggingface`) and persist across container restarts.
