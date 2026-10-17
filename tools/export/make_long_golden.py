"""NeMo reference transcription of the Go test's long clip: all golden clips (sorted by name)
repeated until >= 40 s, exactly as engine_test.go builds it. Output: <out>/long.json"""
import glob, json, os, sys, numpy as np, soundfile as sf
from omegaconf import open_dict
import nemo.collections.asr as nemo_asr
G, OUT = sys.argv[1], sys.argv[2]
names = sorted(os.path.basename(p)[:-5] for p in glob.glob(G + "/*.json"))
clips = [sf.read(f"{G}/{n}.wav", dtype="int16")[0] for n in names]
long = np.zeros(0, dtype=np.int16)
while len(long) < 40 * 16000:
    for c in clips:
        long = np.concatenate([long, c])
os.makedirs(OUT, exist_ok=True)
wav = "/tmp/long.wav"; sf.write(wav, long, 16000, subtype="PCM_16")
m = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2", map_location="cpu").eval()
dc = m.cfg.decoding
with open_dict(dc):
    dc.strategy = "greedy"; dc.greedy.max_symbols = 10
m.change_decoding_strategy(dc)
h = m.transcribe([wav], return_hypotheses=True)[0]
if isinstance(h, list): h = h[0]
json.dump({"clips": names, "seconds": len(long) / 16000, "text": h.text,
           "tokens": [int(t) for t in h.y_sequence]}, open(f"{OUT}/long.json", "w"), indent=2)
print("LONG_DONE", len(long) / 16000, "s,", len(h.y_sequence), "tokens")
print(h.text)
