<h1 align="center">🎙️ Real-Time Voice Cloning</h1>
<p align="center">
  <b>Zero-shot voice cloning with SV2TTS — clone a voice from a few seconds of audio and make it say anything</b>
</p>

<p align="center">
  <a href="https://github.com/Aryannn97/Real-Time-Voice-Cloning-RTVC/actions/workflows/ci.yml"><img src="https://github.com/Aryannn97/Real-Time-Voice-Cloning-RTVC/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
  <img src="https://img.shields.io/badge/Python-3.10--3.12-3776AB?logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License" />
  <img src="https://img.shields.io/badge/Paper-IRJIET%202023-blue" alt="Paper" />
</p>

<p align="center">
  <a href="#features">Features</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="#how-it-works">How It Works</a> ·
  <a href="#xtts-v2-engine-optional">XTTS-v2</a> ·
  <a href="#project-structure">Structure</a> ·
  <a href="#publication">Publication</a>
</p>

---

## Why This Project?

Most voice-cloning systems need hours of recordings per speaker. This project follows the
[SV2TTS](https://arxiv.org/pdf/1806.04558.pdf) framework, which separates *who is speaking* from
*what is said*, so a new voice can be cloned from a short reference clip with no retraining.

On top of the base pipeline, this version adds an audio cleanup stage, a safer microphone workflow,
compatibility with current NumPy / SciPy / librosa / PyTorch releases, a command line tool and
an optional second engine (XTTS-v2).

## Features

### Cloning
| Feature | Description |
|---|---|
| 🗣️ **Zero-shot cloning** | Clone an unseen speaker from a few seconds of reference audio |
| 🧩 **Three-stage pipeline** | Speaker encoder → Tacotron 2 synthesizer → WaveRNN vocoder |
| 🎚️ **Audio cleanup** | Hiss and rumble removal, loudness leveling, click-free joins between sentences (switchable) |
| 🌍 **Optional XTTS-v2 engine** | 24 kHz multilingual synthesis from the command line |

### Tooling
| Feature | Description |
|---|---|
| 🖥️ **Desktop toolbox** | Interactive UI with embedding projections and spectrogram views |
| 🎤 **Safer microphone setup** | 10 s capture, device selector that skips virtual devices and rejects silent recordings |
| ⌨️ **CLI** | `generate.py` clones and speaks from the terminal |
| 📂 **Dataset browsing** | Load voices from files or common speech datasets (LibriSpeech and others) |
| 🔁 **Reproducible** | Fixed random seed for repeatable output |
| ✅ **Tests and CI** | pytest suite and a GitHub Actions workflow |

## Quick Start

Windows and Linux are supported.

**1. Install [ffmpeg](https://ffmpeg.org/download.html#get-packages)** (needed to read audio files):
```bash
ffmpeg -version
```

**2. Install [uv](https://docs.astral.sh/uv/):**
```bash
# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
# Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
# Any platform with pip
pip install -U uv
```

**3. Run it:**
```bash
# Desktop toolbox, NVIDIA GPU
uv run --extra cuda demo_toolbox.py
# Desktop toolbox, CPU only
uv run --extra cpu demo_toolbox.py

# Step-by-step tour of the pipeline
uv run --extra cuda demo_cli.py
```
`uv` creates a `.venv` with the right Python environment automatically. Pretrained models download on
first run into `saved_models/default`.

### Command line

```bash
uv run --extra cpu generate.py --reference voice.wav --text "Hello there." --out hello.wav
uv run --extra cpu generate.py --reference voice.wav --text-file script.txt --out out.wav
uv run --extra cpu generate.py --list_engines
```

| Flag | Description |
|---|---|
| `--reference` | Audio file of the voice to clone |
| `--text` / `--text-file` | Text to speak (a new line starts a new sentence group) |
| `--out` | Output audio file (default `output.wav`) |
| `--engine` | `classic` (default) or `xtts` |
| `--language` | Language code, e.g. `en`, `es`, `fr`, `de` (engine permitting) |
| `--speed` | Speaking speed (engine permitting) |
| `--no_enhance` | Skip the audio cleanup |
| `--cpu` | Force CPU even if a GPU exists |

### Getting a good clone

The clone is only as good as the recording it learns from.
- Choose your real microphone in the **Microphone** list. Virtual devices are skipped by default because they often deliver silence.
- Record in a quiet room, close to the microphone, and talk for the full 10 seconds.
- Play the recording back before generating. If it doesn't sound like you, record again.

## How It Works

```
 reference audio ──▶ ┌───────────────┐
 (a few seconds)     │ Speaker       │──▶ speaker embedding ─┐
                     │ Encoder       │                       │
                     └───────────────┘                       ▼
 text ───────────────────────────────────────▶ ┌────────────────────┐
                                               │ Synthesizer        │──▶ mel-spectrogram
                                               │ (Tacotron 2)       │
                                               └────────────────────┘
                                                                          │
                                                                          ▼
                                                              ┌────────────────────┐
                                                              │ Vocoder (WaveRNN)  │──▶ waveform
                                                              └────────────────────┘
                                                                          │
                                                                          ▼
                                                              audio cleanup ──▶ output .wav
```

| Stage | Role | Reference |
|---|---|---|
| **Speaker encoder** | Turns a short clip into a fixed-size speaker embedding, trained with GE2E loss | [GE2E](https://arxiv.org/pdf/1710.10467.pdf) |
| **Synthesizer** | Tacotron 2 generates a mel-spectrogram from text, conditioned on the embedding | [Tacotron](https://arxiv.org/pdf/1703.10135.pdf) |
| **Vocoder** | WaveRNN converts the mel-spectrogram to a waveform | [WaveRNN](https://arxiv.org/pdf/1802.08435.pdf) |
| **Cleanup** | Hiss and rumble removal, loudness leveling, sentence joins | `utils/postprocess.py` |

## XTTS-v2 Engine (optional)

`generate.py --engine xtts` uses XTTS-v2, a newer 24 kHz multilingual model:
```bash
uv run --extra cuda --extra xtts generate.py --engine xtts --reference voice.wav --text "Hello there." --out hello.wav
```

> **License note:** the XTTS-v2 weights use the [Coqui Public Model License](https://coqui.ai/cpml), which is
> **non-commercial**. The model downloads only after you accept it with `--accept_xtts_license`
> (or `COQUI_TOS_AGREED=1`). The classic pipeline has no such restriction.

## Datasets (optional)

To browse real speakers in the toolbox, download
[`LibriSpeech/train-clean-100`](https://www.openslr.org/resources/12/train-clean-100.tar.gz), extract it to
`<datasets_root>/LibriSpeech/train-clean-100`, and start the toolbox with `-d <datasets_root>`. Other supported
datasets are listed in `toolbox/__init__.py`. Without a dataset you can still load your own files or record with the microphone.

If the automatic model download fails, fetch `encoder.pt`, `synthesizer.pt` and `vocoder.pt` from the
[model repository](https://huggingface.co/CorentinJ/SV2TTS/tree/main) and place them in `saved_models/default`.

## Project Structure

```
Real-Time-Voice-Cloning-RTVC/
├── encoder/              # Speaker encoder: model, training, inference
├── synthesizer/          # Tacotron 2 synthesizer: model, text cleaning, inference
├── vocoder/              # WaveRNN vocoder: models, training, inference
├── engines/              # Engine layer: classic SV2TTS and optional XTTS-v2
├── toolbox/              # Desktop UI (PyQt5)
├── utils/                # Audio cleanup, model download, reference-audio handling
├── tests/                # pytest suite
├── samples/              # Example reference voices
├── demo_toolbox.py       # Launch the desktop toolbox
├── demo_cli.py           # Step-by-step pipeline tour
├── generate.py           # Command line cloning tool
└── pyproject.toml        # Dependencies (managed with uv)
```

## Running Tests

```bash
uv run --extra cpu pytest -q
```

## Responsible Use

Only clone voices you have the right to use, and never use generated audio to impersonate or deceive
people. Disclose synthetic audio when you share it.

## Publication

Dandwekar, A. M. *Real Time Voice Cloning System.* International Research Journal of Innovation in
Engineering and Technology (IRJIET), October 2023.
[Read the paper](https://irjiet.com/article/Real-Time-Voice-Cloning-System/1864).

## Credits and License

Released under the MIT License (see [LICENSE](LICENSE)). This project builds on the SV2TTS paper and on the
open-source work of Rayhane Mama (Tacotron), fatchord (WaveRNN), braindead (logmmse) and the original
[Real-Time-Voice-Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning) by Corentin Jemine. Their
copyright notices are preserved in `LICENSE`, `synthesizer/LICENSE.txt` and `vocoder/LICENSE.txt`.

| Paper | Role |
|---|---|
| [Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis](https://arxiv.org/pdf/1806.04558.pdf) | SV2TTS framework |
| [Efficient Neural Audio Synthesis](https://arxiv.org/pdf/1802.08435.pdf) | WaveRNN vocoder |
| [Tacotron: Towards End-to-End Speech Synthesis](https://arxiv.org/pdf/1703.10135.pdf) | Synthesizer |
| [Generalized End-To-End Loss for Speaker Verification](https://arxiv.org/pdf/1710.10467.pdf) | Encoder |

If this project helped you, consider giving it a star ⭐
