# Model export

Run on Linux (WSL Ubuntu or Lambda; CPU is fine): `python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt`
then the usage line at the top of `export_parakeet.py`. Record `pip freeze > requirements.lock`.

Golden clips: 5 utterances of 3-8 s from LibriSpeech test-clean (CC BY 4.0), converted to 16 kHz mono WAV.

**Preprocessor fallback:** if `torch.onnx.export` of the preprocessor fails (complex STFT), the script exits with
a message. Download `nemo128.onnx` from the `istupakov/parakeet-tdt-0.6b-v2-onnx` Hugging Face repo and rerun with
`--preprocessor-onnx /path/to/nemo128.onnx`. The file is copied to `<out>/preprocessor.onnx` and the manifest's
`preprocessor` entry is built from it. Task 1.5's golden mel test decides whether it is acceptable.
