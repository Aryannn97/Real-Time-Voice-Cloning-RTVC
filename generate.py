"""
Speak text in a cloned voice from the command line.

    python generate.py --reference voice.wav --text "Hello there." --out hello.wav
    python generate.py --engine classic --reference voice.wav --text-file script.txt --out out.wav
    python generate.py --list_engines
"""
import argparse
import os
import sys
from pathlib import Path

import soundfile as sf

from engines import DEFAULT_ENGINE, EngineUnavailable, create_engine, list_engines
from engines.xtts import LICENSE_ENV_VAR
from utils import postprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engine", default=DEFAULT_ENGINE, help="Which engine to use")
    parser.add_argument("--list_engines", action="store_true", help="Show the engines and exit")
    parser.add_argument("--reference", type=Path, help="Audio file of the voice to clone")
    parser.add_argument("--text", help="The text to speak. A new line starts a new sentence group.")
    parser.add_argument("--text-file", dest="text_file", type=Path,
                        help="Read the text from this file instead of --text")
    parser.add_argument("--out", type=Path, default=Path("output.wav"), help="Output audio file")
    parser.add_argument("--language", default="en", help="Language code, e.g. en, es, fr, de")
    parser.add_argument("--speed", type=float, default=1.0, help="Speaking speed (engine permitting)")
    parser.add_argument("--no_enhance", action="store_true",
                        help="Skip the audio cleanup (hiss removal, loudness leveling)")
    parser.add_argument("--cpu", action="store_true", help="Run on the CPU even if a GPU exists")
    parser.add_argument("--accept_xtts_license", action="store_true",
                        help="Accept the non-commercial Coqui Public Model License for XTTS-v2")
    args = parser.parse_args()

    if args.list_engines:
        for name, display, ok, reason in list_engines():
            print("%-8s %-34s %s" % (name, display, "ready" if ok else "unavailable: " + reason))
        return 0

    if args.reference is None or not args.reference.exists():
        parser.error("--reference must point to an existing audio file")
    if args.text_file is not None:
        text = args.text_file.read_text(encoding="utf-8")
    elif args.text:
        text = args.text
    else:
        parser.error("provide --text or --text-file")

    if args.accept_xtts_license:
        os.environ[LICENSE_ENV_VAR] = "1"

    kwargs = {}
    if args.engine == "xtts":
        kwargs["use_gpu"] = not args.cpu
    elif args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    try:
        engine = create_engine(args.engine, **kwargs)
        print("Loading %s..." % engine.display_name)
        voice = engine.create_voice(args.reference)
        print("Generating...")
        wav = engine.synthesize(
            text, voice, language=args.language, speed=args.speed,
            progress_callback=lambda i, n: print("  part %d/%d" % (min(i + 1, n), n), end="\r"))
    except EngineUnavailable as e:
        print("\n%s" % e, file=sys.stderr)
        return 1

    if not args.no_enhance:
        wav = postprocess.enhance(wav, engine.sample_rate)
    sf.write(str(args.out), wav.astype("float32"), engine.sample_rate)
    print("\nSaved %.1f s of audio to %s" % (len(wav) / engine.sample_rate, args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
