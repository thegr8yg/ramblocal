"""Dump NeMo log-mel frontend constants (filterbank + window) next to an exported model.

Usage: python dump_frontend.py MODEL_DIR
The manifest.json frontend section (n_fft 512, win_length 400, hop 160, preemph 0.97,
log_guard 2**-24, norm_eps 1e-5) is added by hand to match NeMo FilterbankFeatures.
"""
import sys, numpy as np, torch, json, inspect
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.preprocessing import features as F
m = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2", map_location="cpu").eval()
fe = m.preprocessor.featurizer
attrs = {}
for k in ["sample_rate","win_length","hop_length","n_fft","preemph","nfilt","lowfreq","highfreq","log","log_zero_guard_type","log_zero_guard_value","dither","pad_to","pad_value","mag_power","normalize","use_grad_checkpointing","stft_pad_amount","exact_pad","nb_max_freq","frame_splicing","mel_norm"]:
    if hasattr(fe, k): 
        v = getattr(fe, k)
        attrs[k] = v if isinstance(v,(int,float,str,type(None),bool)) else str(v)
print("ATTRS", json.dumps(attrs))
print("fb", tuple(fe.fb.shape), fe.fb.dtype, "window", tuple(fe.window.shape), fe.window.dtype)
out = sys.argv[1].rstrip("/") + "/"  # model dir that already holds manifest.json
np.ascontiguousarray(fe.fb[0].numpy().astype("<f4")).tofile(out + "mel_filterbank.f32")
np.ascontiguousarray(fe.window.numpy().astype("<f4")).tofile(out + "mel_window.f32")
print("preproc cfg:", dict(m.cfg.preprocessor))
