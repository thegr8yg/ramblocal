"""Transcribe a manifest with faster-whisper and write hyps JSONL with per-item seconds."""
import argparse
import json
import pathlib
import time

from faster_whisper import WhisperModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="large-v3-turbo")
    ap.add_argument("--compute-type", default="int8")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    model = WhisperModel(args.model, device="cuda", compute_type=args.compute_type)
    base = pathlib.Path(args.manifest).parent
    items = [json.loads(l) for l in open(args.manifest, encoding="utf-8") if l.strip()]
    if args.limit:
        items = items[: args.limit]
    # Warm-up so the first item's time excludes model load and CUDA init.
    list(model.transcribe(str(base / items[0]["audio"]), language="en", beam_size=1)[0])
    with open(args.out, "w", encoding="utf-8") as f:
        for it in items:
            start = time.perf_counter()
            segments, _ = model.transcribe(str(base / it["audio"]), language="en", beam_size=1,
                                           vad_filter=False, condition_on_previous_text=False)
            text = " ".join(s.text.strip() for s in segments)
            secs = time.perf_counter() - start
            f.write(json.dumps({"audio": it["audio"], "hyp": text, "seconds": secs}) + "\n")


if __name__ == "__main__":
    main()
