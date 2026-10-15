# Model export

Run on Linux (WSL Ubuntu or Lambda; CPU is fine): `python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt`
then the usage line at the top of `export_parakeet.py`. Record `pip freeze > requirements.lock`.

Golden clips: 5 utterances of 3-8 s from LibriSpeech test-clean (CC BY 4.0), converted to 16 kHz mono WAV.

**Preprocessor fallback:** if `torch.onnx.export` of the preprocessor fails (complex STFT), use the
`nemo128.onnx` preprocessor from the `istupakov/parakeet-tdt-0.6b-v2-onnx` Hugging Face repo, renamed to
`preprocessor.onnx`, and regenerate the manifest's `preprocessor` entry from it. Task 1.5's golden mel
test decides whether it is acceptable.
