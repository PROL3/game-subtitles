"""Fast local English -> Hebrew translation with NLLB-200 and CTranslate2."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import ctranslate2
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer


MODEL_ID = os.getenv("NLLB_MODEL", "facebook/nllb-200-distilled-600M")
SOURCE_LANGUAGE = "eng_Latn"
TARGET_LANGUAGE = "heb_Hebr"


def _project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(__file__).parent / path


MODEL_DIR = _project_path(os.getenv("NLLB_MODEL_DIR", ".models/nllb-200-distilled-600M-int8"))
SOURCE_DIR = _project_path(os.getenv("NLLB_SOURCE_DIR", ".models/nllb-200-distilled-600M"))


def _ensure_converted_model() -> tuple[Path, Path]:
    """Download NLLB and convert it once to an INT8 CTranslate2 model."""
    tokenizer_dir = SOURCE_DIR
    if not (tokenizer_dir / "tokenizer.json").exists():
        print(f"Downloading {MODEL_ID} for NLLB tokenizer/model conversion...", flush=True)
        downloaded = Path(snapshot_download(MODEL_ID, local_dir=tokenizer_dir))
        tokenizer_dir = downloaded

    if not (MODEL_DIR / "model.bin").exists():
        MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
        converter = shutil.which("ct2-transformers-converter")
        if converter:
            command = [converter]
        else:
            command = [sys.executable, "-m", "ct2_transformers_converter"]
        print("Converting NLLB-200 to CTranslate2 INT8 (one-time setup)...", flush=True)
        subprocess.run(
            command
            + [
                "--model",
                str(tokenizer_dir),
                "--output_dir",
                str(MODEL_DIR),
                "--quantization",
                "int8",
                "--low_cpu_mem_usage",
            ],
            check=True,
        )
    return MODEL_DIR, tokenizer_dir


class NLLBTranslator:
    def __init__(self) -> None:
        model_dir, tokenizer_dir = _ensure_converted_model()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, src_lang=SOURCE_LANGUAGE)
        self.translator = ctranslate2.Translator(
            str(model_dir),
            device=os.getenv("NLLB_DEVICE", "cpu"),
            compute_type=os.getenv("NLLB_COMPUTE_TYPE", "int8"),
        )
        self.target_prefix = [TARGET_LANGUAGE]

    def translate(self, text: str) -> str:
        encoded = self.tokenizer(text, add_special_tokens=True)
        source_tokens = self.tokenizer.convert_ids_to_tokens(encoded["input_ids"])
        result = self.translator.translate_batch(
            [source_tokens],
            target_prefix=[self.target_prefix],
            beam_size=2,
            max_decoding_length=128,
        )[0]
        # The first generated token is the forced target language prefix.
        output_tokens = result.hypotheses[0][1:]
        output_ids = self.tokenizer.convert_tokens_to_ids(output_tokens)
        return self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
