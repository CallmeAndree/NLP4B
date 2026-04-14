
"""
paddle_ocr.py — PaddleOCR-VL Keyframe OCR Pipeline
====================================================
Extracts text from keyframe images using PaddleOCR-VL 1.5.

Output contract (per AGENTS.md):
  <video_id>_ocr.json — array of {"image": "<filename>", "ocr_text": "<text>"}

Usage:
  python -m src.ocr.paddle_ocr -i <keyframe_dir> -o <output_dir>
  python -m src.ocr.paddle_ocr -i <keyframe_dir> -o <output_dir> --batch_size 4
  python -m src.ocr.paddle_ocr --prepare_only --hf_cache_dir /path/to/cache
"""

import argparse
import gc
import json
import logging
import os
import time
import traceback
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

try:
    import psutil
except Exception:  # pragma: no cover - optional dependency
    psutil = None

# ==========================================
# 0. LOGGING / SYSTEM HELPERS
# ==========================================
LOG_FMT = "[%(asctime)s] [ocr] %(levelname)-8s %(message)s"
DATE_FMT = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(level=logging.INFO, format=LOG_FMT, datefmt=DATE_FMT)
logger = logging.getLogger(__name__)

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

DEFAULT_MODEL_ID = "PaddlePaddle/PaddleOCR-VL-1.5"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def log_system_state(prefix: str = "") -> None:
    parts = []
    if prefix:
        parts.append(prefix)

    parts.append(f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        current = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current)
        parts.append(f"gpu={current}:{gpu_name}")
        try:
            allocated = torch.cuda.memory_allocated(current) / (1024**3)
            reserved = torch.cuda.memory_reserved(current) / (1024**3)
            parts.append(f"allocated_gb={allocated:.2f}")
            parts.append(f"reserved_gb={reserved:.2f}")
        except Exception:
            pass

    if psutil is not None:
        vm = psutil.virtual_memory()
        parts.append(f"ram_used_pct={vm.percent}")
        parts.append(f"ram_available_gb={vm.available / (1024 ** 3):.2f}")

    logger.info(" | ".join(parts))


def cleanup_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ==========================================
# 1. MODEL LOADING
# ==========================================
def prepare_hf_cache(
    model_id: str,
    cache_dir: str | Path,
    revision: str = "main",
) -> str:
    """Pre-download model to a local cache directory."""
    from huggingface_hub import snapshot_download

    cache_dir = str(cache_dir)
    logger.info(f"Preparing OCR model cache | model={model_id} | cache_dir={cache_dir}")
    local_model_dir = snapshot_download(
        repo_id=model_id,
        revision=revision,
        cache_dir=cache_dir,
    )
    logger.info(f"Model cache ready at: {local_model_dir}")
    return local_model_dir


def resolve_model_source(
    model_id: str,
    hf_cache_dir: str | Path | None = None,
    revision: str = "main",
    prepare_only: bool = False,
) -> str:
    """Return local model path when cache dir is provided, otherwise return original model id."""
    is_local_path = Path(model_id).exists()
    if is_local_path:
        logger.info(f"Using local model path: {model_id}")
        return str(Path(model_id))

    if hf_cache_dir:
        return prepare_hf_cache(model_id, cache_dir=hf_cache_dir, revision=revision)

    if prepare_only:
        raise ValueError("--prepare_only requires either a local --model path or --hf_cache_dir.")

    return model_id


def load_model(
    model_id: str = DEFAULT_MODEL_ID,
    hf_cache_dir: str | Path | None = None,
    revision: str = "main",
    device: str | None = None,
    torch_dtype: str = "auto",
) -> dict[str, Any]:
    """Load PaddleOCR-VL model and processor, returning a reusable runtime bundle."""
    runtime_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (
        torch.bfloat16
        if torch_dtype == "auto"
        else getattr(torch, torch_dtype)
    )

    log_system_state("Before model load")

    model_source = resolve_model_source(model_id, hf_cache_dir=hf_cache_dir, revision=revision)
    local_files_only = Path(model_source).exists()

    logger.info(f"Loading PaddleOCR-VL config from: {model_source}")
    config = AutoConfig.from_pretrained(
        model_source,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    # Some transformers versions require text_config to exist
    if not hasattr(config, "text_config"):
        config.text_config = config

    logger.info(f"Loading PaddleOCR-VL model | dtype={dtype} | device={runtime_device}")
    model = AutoModelForImageTextToText.from_pretrained(
        model_source,
        config=config,
        torch_dtype=dtype,
        trust_remote_code=True,
        local_files_only=local_files_only,
        dtype=dtype,
    ).to(runtime_device).eval()
    logger.info("PaddleOCR-VL model loaded")

    logger.info(f"Loading processor from: {model_source}")
    processor = AutoProcessor.from_pretrained(
        model_source,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    logger.info("Processor loaded")

    log_system_state("After model load")

    return {
        "model": model,
        "processor": processor,
        "device": runtime_device,
        "dtype": dtype,
        "model_source": model_source,
    }


# ==========================================
# 2. CORE OCR PIPELINE
# ==========================================
def ocr_batch(
    image_paths: list[Path],
    runtime: dict[str, Any],
    max_new_tokens: int = 512,
) -> list[dict]:
    """Run OCR on a batch of images, returning list of {"image": ..., "ocr_text": ...}."""
    model = runtime["model"]
    processor = runtime["processor"]

    batch_images = []
    valid_filenames = []
    for img_path in image_paths:
        try:
            with Image.open(img_path) as img:
                batch_images.append(img.convert("RGB").copy())
                valid_filenames.append(img_path.name)
        except Exception as e:
            logger.warning(f"Failed to open {img_path.name}: {e}")
            continue

    if not batch_images:
        return []

    max_pixels = 1280 * 28 * 28
    batch_messages = [
        [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": "OCR:"}]}]
        for img in batch_images
    ]

    inputs = processor.apply_chat_template(
        batch_messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        images_kwargs={"size": {"shortest_edge": 448, "longest_edge": max_pixels}},
    ).to(model.device, dtype=runtime["dtype"])

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=processor.tokenizer.pad_token_id,
    )

    prompt_len = inputs["input_ids"].shape[-1]
    results = []
    for idx, output_tensor in enumerate(outputs):
        text = processor.decode(output_tensor[prompt_len:-1], skip_special_tokens=True)
        results.append({
            "image": valid_filenames[idx],
            "ocr_text": text.strip(),
        })

    del inputs, outputs, batch_images
    return results


# ==========================================
# 3. DIRECTORY PROCESSING
# ==========================================
def discover_images(input_dir: Path) -> list[Path]:
    """Find all image files in a directory, sorted."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    images = sorted(
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )
    return images


def save_ocr_results(output_file: Path, results: list[dict]) -> None:
    """Write OCR results atomically to avoid partial files."""
    tmp_file = output_file.with_suffix(output_file.suffix + ".tmp")
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    tmp_file.replace(output_file)


def load_existing_results(output_file: Path) -> list[dict]:
    """Load previously saved OCR results for checkpoint resumption."""
    if not output_file.exists():
        return []

    try:
        with open(output_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.warning(f"Could not load existing output {output_file}: {e}")
        return []

    if not isinstance(data, list):
        logger.warning(f"Existing output {output_file} is not a JSON array. Starting fresh.")
        return []

    deduped = []
    seen = set()
    for item in data:
        if not isinstance(item, dict):
            continue
        img_name = item.get("image", "")
        if not img_name or img_name in seen:
            continue
        seen.add(img_name)
        deduped.append(item)

    if len(deduped) != len(data):
        logger.info(f"Removed {len(data) - len(deduped)} duplicate/invalid entries from existing output.")

    return deduped


def process_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    runtime: dict[str, Any],
    batch_size: int = 6,
    limit: int | None = None,
    save_every: int = 50,
    max_new_tokens: int = 512,
) -> Path:
    """Process one video directory of keyframes, producing <video_id>_ocr.json."""
    input_folder = Path(input_dir)
    output_folder = Path(output_dir)
    output_folder.mkdir(parents=True, exist_ok=True)

    image_files = discover_images(input_folder)
    if limit:
        image_files = image_files[:limit]

    video_id = input_folder.name or input_folder.resolve().name or "unknown_video"
    output_file = output_folder / f"{video_id}_ocr.json"

    # Resume from checkpoint
    all_results = load_existing_results(output_file)
    processed_images = {item["image"] for item in all_results}
    remaining_images = [img for img in image_files if img.name not in processed_images]

    logger.info(
        f"Video {video_id} | total_images={len(image_files)} "
        f"| already_processed={len(processed_images)} | remaining={len(remaining_images)}"
    )

    if not remaining_images:
        save_ocr_results(output_file, all_results)
        logger.info(f"All images already processed for {video_id}. Output at {output_file}")
        return output_file

    # Torch performance optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    processed_since_save = 0
    start_time = time.time()

    with torch.inference_mode():
        for i in tqdm(
            range(0, len(remaining_images), batch_size),
            desc=f"OCR {video_id}",
            unit="batch",
        ):
            batch_paths = remaining_images[i : i + batch_size]
            try:
                batch_results = ocr_batch(batch_paths, runtime, max_new_tokens=max_new_tokens)
                all_results.extend(batch_results)
                processed_since_save += len(batch_results)

                if save_every > 0 and processed_since_save >= save_every:
                    save_ocr_results(output_file, all_results)
                    logger.info(
                        f"Checkpoint saved for {video_id} after {processed_since_save} images: {output_file}"
                    )
                    processed_since_save = 0
            except Exception as e:
                logger.error(f"Batch failed at index {i}: {e}")
                traceback.print_exc()
            finally:
                cleanup_memory()

    # Final save
    save_ocr_results(output_file, all_results)
    elapsed = time.time() - start_time
    logger.info(
        f"Done {video_id} | {len(all_results)} results saved to {output_file} | elapsed={elapsed:.1f}s"
    )
    return output_file


# ==========================================
# 4. CLI
# ==========================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PaddleOCR-VL Keyframe OCR Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single video folder
  python -m src.ocr.paddle_ocr -i ./keyframes/video_id -o ./ocr_output

  # Pre-download model to cache
  python -m src.ocr.paddle_ocr --prepare_only --hf_cache_dir ./hf_cache

  # Use local model with custom batch size
  python -m src.ocr.paddle_ocr -i ./keyframes/video_id -o ./ocr_output --model ./local_model --batch_size 4
        """,
    )
    parser.add_argument("-i", "--input_dir", type=str, help="Path to directory containing keyframe images")
    parser.add_argument("-o", "--output_dir", type=str, default=".", help="Path to output directory for OCR JSONs")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_ID, help="HuggingFace model ID or local path")
    parser.add_argument("--revision", type=str, default="main", help="Pinned model revision / commit hash")
    parser.add_argument("--hf_cache_dir", type=str, default=None, help="HF cache dir for pre-downloading model")
    parser.add_argument("--prepare_only", action="store_true", help="Only pre-download model, then exit")
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"], help="Execution device")
    parser.add_argument(
        "--torch_dtype", type=str, default="auto",
        choices=["auto", "float16", "float32", "bfloat16"],
        help="Model dtype (default: auto → bfloat16)",
    )
    parser.add_argument("--batch_size", type=int, default=6, help="Images per batch (default: 6, adjust for VRAM)")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max tokens to generate per image")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images to process (default: all)")
    parser.add_argument(
        "--save_every", type=int, default=50,
        help="Save checkpoint JSON every N images (0 disables periodic save)",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    logger.info("═" * 60)
    logger.info("PaddleOCR-VL Keyframe OCR Pipeline")
    logger.info("═" * 60)

    # Handle prepare_only mode
    if args.prepare_only:
        model_source = resolve_model_source(
            model_id=args.model,
            hf_cache_dir=args.hf_cache_dir,
            revision=args.revision,
            prepare_only=True,
        )
        logger.info(f"Prepare-only completed. Model source ready at: {model_source}")
        return

    if not args.input_dir:
        parser.error("--input_dir / -i is required unless --prepare_only is used.")

    logger.info(f"Input directory:  {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Model:            {args.model}")
    logger.info(f"Batch size:       {args.batch_size}")
    logger.info(f"Max new tokens:   {args.max_new_tokens}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device:      {torch.cuda.get_device_name(0)}")

    # Load model
    runtime = load_model(
        model_id=args.model,
        hf_cache_dir=args.hf_cache_dir,
        revision=args.revision,
        device=args.device,
        torch_dtype=args.torch_dtype,
    )

    # Process
    process_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        runtime=runtime,
        batch_size=args.batch_size,
        limit=args.limit,
        save_every=args.save_every,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
