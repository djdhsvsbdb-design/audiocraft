# Public prompt-only MusicGen app

This is a minimal public-facing interface around AudioCraft's text-to-music model.
Visitors describe the music in one text field. The model chooses the musical
character from that description; the interface deliberately has no genre, style,
instrument, duration, or model selectors.

## Run

From the repository root, install AudioCraft using the instructions in the main
README, then make sure FFmpeg is installed on the server:

```bash
python demos/public_musicgen_app.py --listen 0.0.0.0 --server_port 7860
```

Open `http://localhost:7860` in a browser. The app uses
`facebook/musicgen-small`, generates a 15-second excerpt, provides in-browser
playback, and converts the result to MP3 for download.

## Deployment notes

- Run on a machine with a supported CUDA GPU for practical generation speed.
  CPU inference is possible but may be very slow.
- The first request downloads/loads model weights and will take longer.
- The Gradio queue is limited to eight waiting requests and one active generation
  per app process. This is a basic safeguard, not production-grade abuse
  prevention or a distributed queue.
- Put the app behind HTTPS and a reverse proxy when exposing it publicly.
- Add infrastructure-level rate limiting, request monitoring, and resource
  limits before opening it to unrestricted traffic.
- Install FFmpeg; the app returns a clear error if MP3 conversion is unavailable.

## Model license

AudioCraft source code is MIT-licensed, but the pretrained model weights are
licensed separately under CC BY-NC 4.0. Public availability does not make that
license commercial. Review `LICENSE_weights` and obtain appropriate rights
before any commercial deployment or monetized use.
