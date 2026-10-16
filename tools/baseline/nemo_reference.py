"""Transcribe a manifest with NeMo (greedy, max_symbols=10) and write hyps JSONL."""
import argparse
import json
import pathlib

from omegaconf import open_dict
import nemo.collections.asr as nemo_asr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="nvidia/parakeet-tdt-0.6b-v2")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    m = nemo_asr.models.ASRModel.from_pretrained(args.model).eval()
    dc = m.cfg.decoding
    with open_dict(dc):
        dc.strategy = "greedy"
        dc.greedy.max_symbols = 10
    m.change_decoding_strategy(dc)

    base = pathlib.Path(args.manifest).parent
    items = [json.loads(l) for l in open(args.manifest, encoding="utf-8") if l.strip()]
    paths = [str(base / it["audio"]) for it in items]
    hyps = m.transcribe(paths, batch_size=16)
    if isinstance(hyps, tuple):
        hyps = hyps[0]
    with open(args.out, "w", encoding="utf-8") as f:
        for it, h in zip(items, hyps):
            text = h.text if hasattr(h, "text") else h
            f.write(json.dumps({"audio": it["audio"], "hyp": text}) + "\n")


if __name__ == "__main__":
    main()
