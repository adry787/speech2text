#!/usr/bin/env python3
import torch, torchaudio


class Transcriber:
    def __init__(self):
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.proc = torch.hub.load("snakers4/silero-models", "silero_stt", trust_repo=True)

    def transcribe(self, path):
        wav, sr = torchaudio.load(path)
        if sr != 16000:
            wav = torchaudio.transforms.Resample(sr, 16000)(wav)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        return self.model(wav)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python transcribe.py audio.mp3")
        exit()
    print(Transcriber().transcribe(sys.argv[1]))
