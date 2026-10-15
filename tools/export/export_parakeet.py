"""Export Parakeet TDT to ONNX + manifest.json + golden fixtures.

Usage:
  python export_parakeet.py --model nvidia/parakeet-tdt-0.6b-v2 \
      --out models/parakeet-tdt-0.6b-v2 --golden-out testdata/golden \
      --golden-wavs clips/*.wav
"""
import argparse
import json
import pathlib

import numpy as np
import onnx
import soundfile as sf
import torch
from omegaconf import open_dict

import nemo.collections.asr as nemo_asr

DECODER_INPUT_ORDER = ["encoder_outputs", "targets", "target_length", "input_states_1", "input_states_2"]
DECODER_OUTPUT_ORDER = ["outputs", "prednet_lengths", "output_states_1", "output_states_2"]


def graph_io(path):
    g = onnx.load(str(path), load_external_data=False).graph
    dtypes = {}
    for v in list(g.input) + list(g.output):
        dtypes[v.name] = onnx.TensorProto.DataType.Name(v.type.tensor_type.elem_type).lower()
    return [i.name for i in g.input], [o.name for o in g.output], dtypes


def ordered(names, wanted, what):
    """Return names reordered to match `wanted`; fail loudly if the export changed names."""
    missing = [w for w in wanted if w not in names]
    if missing:
        raise SystemExit(f"{what}: expected tensor names {wanted}, export has {names}")
    return list(wanted)


def resave(src, dst):
    """Re-save with all weights in one external-data file (encoder exceeds the 2 GB protobuf limit)."""
    m = onnx.load(str(src), load_external_data=True)
    onnx.save_model(m, str(dst), save_as_external_data=True,
                    all_tensors_to_one_file=True, location=dst.name + ".data")


class Pre(torch.nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, waveforms, waveforms_lens):
        return self.p(input_signal=waveforms, length=waveforms_lens)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="nvidia/parakeet-tdt-0.6b-v2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--golden-out", required=True)
    ap.add_argument("--golden-wavs", nargs="+", required=True)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "_export"
    tmp.mkdir(exist_ok=True)

    m = nemo_asr.models.ASRModel.from_pretrained(args.model, map_location="cpu").eval()
    m.preprocessor.featurizer.dither = 0.0
    m.preprocessor.featurizer.pad_to = 0

    dc = m.cfg.decoding
    with open_dict(dc):
        dc.strategy = "greedy"
        dc.greedy.max_symbols = 10
    m.change_decoding_strategy(dc)

    # Encoder + decoder_joint via NeMo's exporter.
    m.export(str(tmp / "model.onnx"))
    resave(next(tmp.glob("encoder*.onnx")), out / "encoder.onnx")
    resave(next(tmp.glob("decoder_joint*.onnx")), out / "decoder_joint.onnx")

    # Preprocessor via torch.onnx (opset 17 has STFT). If this raises, see tools/export/README.md fallback.
    torch.onnx.export(
        Pre(m.preprocessor), (torch.zeros(1, 16000), torch.tensor([16000])),
        str(out / "preprocessor.onnx"),
        input_names=["waveforms", "waveforms_lens"], output_names=["features", "features_lens"],
        dynamic_axes={"waveforms": {1: "n"}, "features": {2: "t"}}, opset_version=17, dynamo=False,
    )

    vocab_size = m.tokenizer.vocab_size
    with open(out / "vocab.txt", "w", encoding="utf-8") as f:
        for i in range(vocab_size):
            f.write(m.tokenizer.ids_to_tokens([i])[0] + "\n")

    pre_in, pre_out, pre_dt = graph_io(out / "preprocessor.onnx")
    enc_in, enc_out, enc_dt = graph_io(out / "encoder.onnx")
    dj_in, dj_out, dj_dt = graph_io(out / "decoder_joint.onnx")
    manifest = {
        "name": args.model.split("/")[-1],
        "sample_rate": int(m.cfg.preprocessor.sample_rate),
        "num_features": int(m.cfg.preprocessor.features),
        "hop_seconds": float(m.cfg.preprocessor.window_stride),
        "subsampling": int(m.cfg.encoder.subsampling_factor),
        "encoder_dim": int(m.cfg.encoder.d_model),
        "vocab_size": int(vocab_size),
        "blank_id": int(vocab_size),
        "tdt_durations": [int(d) for d in m.cfg.model_defaults.tdt_durations],
        "pred_layers": int(m.cfg.decoder.prednet.pred_rnn_layers),
        "pred_hidden": int(m.cfg.decoder.prednet.pred_hidden),
        "preprocessor": {"file": "preprocessor.onnx", "inputs": pre_in, "outputs": pre_out, "dtypes": pre_dt},
        "encoder": {"file": "encoder.onnx", "inputs": enc_in, "outputs": enc_out, "dtypes": enc_dt},
        "decoder_joint": {
            "file": "decoder_joint.onnx",
            "inputs": ordered(dj_in, DECODER_INPUT_ORDER, "decoder_joint inputs"),
            "outputs": ordered(dj_out, DECODER_OUTPUT_ORDER, "decoder_joint outputs"),
            "dtypes": dj_dt,
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    gold = pathlib.Path(args.golden_out)
    gold.mkdir(parents=True, exist_ok=True)
    for wav in args.golden_wavs:
        wav = pathlib.Path(wav)
        audio, sr = sf.read(wav, dtype="float32")
        assert sr == 16000 and audio.ndim == 1, f"{wav}: need 16 kHz mono"
        x = torch.from_numpy(audio)[None]
        xl = torch.tensor([x.shape[1]])
        with torch.no_grad():
            feats, fl = m.preprocessor(input_signal=x, length=xl)
            enc, el = m.encoder(audio_signal=feats, length=fl)
        hyp = m.transcribe([str(wav)], return_hypotheses=True)[0]
        if isinstance(hyp, list):
            hyp = hyp[0]
        mel = feats[0, :, : int(fl[0])].numpy().astype("<f4")
        e = enc[0, :, : int(el[0])].numpy().astype("<f4")
        name = wav.stem
        sf.write(gold / f"{name}.wav", audio, 16000, subtype="PCM_16")
        np.ascontiguousarray(mel).tofile(gold / f"{name}.mel.f32")
        np.ascontiguousarray(e).tofile(gold / f"{name}.enc.f32")
        (gold / f"{name}.json").write_text(json.dumps({
            "tokens": [int(t) for t in hyp.y_sequence],
            "text": hyp.text,
            "mel_shape": list(mel.shape),
            "enc_shape": list(e.shape),
        }, indent=2))
        print(f"{name}: {hyp.text}")


if __name__ == "__main__":
    main()
