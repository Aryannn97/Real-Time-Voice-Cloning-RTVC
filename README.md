# Real-Time-Voice-Cloning

Clone a voice from a few seconds of audio and make it say anything you type. The project follows the
[SV2TTS](https://arxiv.org/pdf/1806.04558.pdf) framework in three stages:

1. **Encoder** – turns a short reference recording into a speaker embedding.
2. **Synthesizer** – generates a mel spectrogram from text, conditioned on that embedding.
3. **Vocoder** – converts the spectrogram into a waveform.

On top of the base pipeline this project adds an audio cleanup stage (hiss and rumble removal,
loudness leveling, click-free joins between sentences), a safer microphone setup in the toolbox and
compatibility with current NumPy, SciPy, librosa, matplotlib and PyTorch releases. An optional,
experimental XTTS-v2 engine is available from the command line (see below).

## Features

- Interactive desktop toolbox with embedding projections and spectrogram views
- 10 second microphone capture, with a microphone selector that skips virtual devices and refuses a silent recording
- Audio cleanup that can be switched on or off (Clean up audio in the toolbox, --no_enhance in the CLI)
- Loading voices from files or from common speech datasets
- Reproducible output with a fixed random seed
- Optional command line tool (generate.py) with a second, higher quality engine
## Usage

Both Windows and Linux are supported.

1. Install [ffmpeg](https://ffmpeg.org/download.html#get-packages). It is required for reading audio files. Check it with:
```
ffmpeg -version
```
2. Install [uv](https://docs.astral.sh/uv/) for Python package management:
```
# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
# Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
# Any platform with pip
pip install -U uv
```
3. Run one of the following:
```
# Toolbox with an NVIDIA GPU
uv run --extra cuda demo_toolbox.py
# Toolbox without a GPU
uv run --extra cpu demo_toolbox.py

# Step-by-step tour of the pipeline
uv run --extra cuda demo_cli.py
```
`uv` creates a `.venv` directory with the right Python environment automatically.

### Getting a good clone

The clone can only be as good as the recording it learns from.
- Choose your real microphone in the **Microphone** list. Virtual devices (phone-camera apps, voice changers) are skipped by default because they often deliver silence.
- Record in a quiet room, close to the microphone, and talk for the full 10 seconds.
- Play the recording back before generating. If it doesn't sound like your voice, record again.

### Optional: XTTS-v2 engine (command line)

`generate.py` uses the classic pipeline by default. With `--engine xtts` it uses XTTS-v2 instead, a newer 24 kHz multilingual model:
```
uv run --extra cuda --extra xtts generate.py --engine xtts --reference voice.wav --text "Hello there." --out hello.wav
uv run --extra cuda generate.py --list_engines
```

#### XTTS-v2 license

The XTTS-v2 weights are published under the [Coqui Public Model License](https://coqui.ai/cpml), which
is **non-commercial**. The model is only downloaded after you accept that license by passing
`--accept_xtts_license` (or setting the environment variable `COQUI_TOS_AGREED=1`). The classic
pipeline has no such restriction.

### Pretrained models

Pretrained models are downloaded automatically on first run into `saved_models/default`. If the
download fails, fetch `encoder.pt`, `synthesizer.pt` and `vocoder.pt` from the
[model repository](https://huggingface.co/CorentinJ/SV2TTS/tree/main) and place them there.

### Datasets (optional)

To browse real speakers in the toolbox, download
[`LibriSpeech/train-clean-100`](https://www.openslr.org/resources/12/train-clean-100.tar.gz) and extract it
as `<datasets_root>/LibriSpeech/train-clean-100`, then start the toolbox with `-d <datasets_root>`. Other
supported datasets are listed in `toolbox/__init__.py`. Without a dataset you can still load your own files or record
with the microphone.

## Tests

```
uv run --extra cpu pytest -q
```

## Responsible use

Only clone voices you have the right to use, and do not use generated audio to impersonate or deceive
people. Disclose synthetic audio when you share it.

## Publication

Dandwekar, A. M. *Real Time Voice Cloning System.* International Research Journal of Innovation in
Engineering and Technology (IRJIET), October 2023. [Read the paper](https://irjiet.com/article/Real-Time-Voice-Cloning-System/1864).

## Credits and license

This project is released under the MIT License (see [LICENSE](LICENSE)). It builds on the SV2TTS paper and on
the open-source work of Rayhane Mama (Tacotron), fatchord (WaveRNN), braindead (logmmse) and the original
Real-Time-Voice-Cloning implementation by Corentin Jemine; their copyright notices are preserved in
`LICENSE`, `synthesizer/LICENSE.txt` and `vocoder/LICENSE.txt`.

| Paper | Role |
| ----- | ---- |
| [Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis](https://arxiv.org/pdf/1806.04558.pdf) | SV2TTS framework |
| [Efficient Neural Audio Synthesis](https://arxiv.org/pdf/1802.08435.pdf) | WaveRNN vocoder |
| [Tacotron: Towards End-to-End Speech Synthesis](https://arxiv.org/pdf/1703.10135.pdf) | Synthesizer |
| [Generalized End-To-End Loss for Speaker Verification](https://arxiv.org/pdf/1710.10467.pdf) | Encoder |
