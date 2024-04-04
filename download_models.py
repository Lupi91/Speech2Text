from huggingface_hub import snapshot_download

print("Downloading model 'nllb-200-distilled-600M-int8'")
snapshot_download(repo_id="next-social/nllb-200-distilled-600M-int8", local_dir="models/NLLB/nllb-200-distilled-600M-int8")

print("Downloading model 'faster-whisper-small'")
snapshot_download(repo_id="guillaumekln/faster-whisper-small", local_dir="models/whisper_small")

print("Downloading model 'faster-whisper-medium'")
snapshot_download(repo_id="guillaumekln/faster-whisper-medium", local_dir="models/whisper_medium")

print("Downloading model 'guillaumekln/faster-whisper-large-v2'")
snapshot_download(repo_id="guillaumekln/faster-whisper-large-v2", local_dir="models/whisper_large")