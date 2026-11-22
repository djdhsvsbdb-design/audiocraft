# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Minimal public-facing, prompt-only MusicGen demo.

Run from the repository root with:
    python demos/public_musicgen_app.py --listen 0.0.0.0 --server_port 7860

The model runs locally; a CUDA-capable PyTorch installation and ffmpeg are
recommended. This app intentionally exposes no genre, style, model, or sampler
controls: the user describes the desired result in one text prompt.
"""

import argparse
import logging
import os
import subprocess
import tempfile
import threading
from pathlib import Path

import gradio as gr
import torch

from audiocraft.data.audio import audio_write
from audiocraft.models import MusicGen


MODEL_NAME = "facebook/musicgen-small"
MAX_PROMPT_LENGTH = 512
GENERATION_SECONDS = 15
MODEL = None
MODEL_LOCK = threading.Lock()


def get_model():
    """Load the model once, on the first generation request."""
    global MODEL
    if MODEL is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info("Loading %s on %s", MODEL_NAME, device)
        MODEL = MusicGen.get_pretrained(MODEL_NAME, device=device)
    return MODEL


def generate_music(prompt: str):
    """Generate one track from a free-form text prompt."""
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Describe the music you want before generating.")
    if len(prompt) > MAX_PROMPT_LENGTH:
        raise gr.Error(f"Prompt is too long. Maximum length: {MAX_PROMPT_LENGTH} characters.")

    # A single shared model is used, so inference is serialized.
    with MODEL_LOCK:
        try:
            model = get_model()
            model.set_generation_params(duration=GENERATION_SECONDS)
            wav = model.generate([prompt], progress=True)[0].detach().cpu().float()
        except gr.Error:
            raise
        except Exception as exc:
            logging.exception("Music generation failed")
            raise gr.Error("Music generation failed. Please try again later.") from exc

    with tempfile.NamedTemporaryFile(prefix="musicgen-", suffix=".wav", delete=False) as wav_file:
        wav_path = Path(wav_file.name)
    mp3_path = wav_path.with_suffix(".mp3")

    try:
        audio_write(
            str(wav_path),
            wav,
            model.sample_rate,
            strategy="loudness",
            loudness_headroom_db=16,
            loudness_compressor=True,
            add_suffix=False,
        )
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(wav_path), "-codec:a", "libmp3lame",
                 "-q:a", "2", str(mp3_path)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise gr.Error("MP3 conversion is unavailable: install ffmpeg on the server.") from exc
        except subprocess.CalledProcessError as exc:
            logging.error("ffmpeg conversion failed: %s", exc.stderr.decode(errors="replace"))
            raise gr.Error("The track was generated, but MP3 conversion failed.") from exc
        return str(wav_path), str(mp3_path)
    except Exception:
        wav_path.unlink(missing_ok=True)
        mp3_path.unlink(missing_ok=True)
        raise


def build_app():
    css = """
    .gradio-container { max-width: 900px !important; margin: auto !important; }
    #hero { text-align: center; padding: 2.5rem 1rem 1.25rem; }
    #hero h1 { font-size: clamp(2rem, 5vw, 3.5rem); letter-spacing: -0.04em; }
    #prompt textarea { font-size: 1.05rem !important; }
    """
    with gr.Blocks(title="MusicGen — AI Music Generator", css=css) as demo:
        gr.Markdown(
            """
            <div id="hero">
              <h1>Turn your words into music</h1>
              <p>Describe the sound you imagine. AI will create an original music track.</p>
            </div>
            """
        )
        prompt = gr.Textbox(
            label="Your music prompt",
            placeholder="Describe the music you want to hear...",
            lines=4,
            max_lines=8,
            max_length=MAX_PROMPT_LENGTH,
            elem_id="prompt",
        )
        generate = gr.Button("Generate music", variant="primary", size="lg")
        status = gr.Markdown("Enter a description and generate your track.")
        with gr.Group():
            wav_output = gr.Audio(label="Listen to your music", type="filepath")
            mp3_output = gr.File(label="Download MP3")
        generate.click(
            fn=generate_music,
            inputs=prompt,
            outputs=[wav_output, mp3_output],
            concurrency_limit=1,
            api_name="generate",
        )
        gr.Examples(
            examples=[
                ["A calm cinematic piano theme with warm strings, slowly building into an uplifting finale."],
                ["A futuristic electronic instrumental with deep bass, crisp percussion and glowing synth textures."],
                ["A gentle acoustic guitar melody, intimate and peaceful, like a quiet morning by the sea."],
            ],
            inputs=prompt,
            label="Prompt ideas",
        )
        gr.Markdown(
            "Generation length: 15 seconds. The first request may take longer while the AI model loads."
        )
    demo.queue(max_size=8, default_concurrency_limit=1)
    return demo


def main():
    parser = argparse.ArgumentParser(description="Public prompt-only MusicGen web app")
    parser.add_argument("--listen", default="127.0.0.1" if "SPACE_ID" not in os.environ else "0.0.0.0")
    parser.add_argument("--server_port", type=int, default=7860)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    build_app().launch(
        server_name=args.listen,
        server_port=args.server_port,
        show_error=False,
    )


if __name__ == "__main__":
    main()
