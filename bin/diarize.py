"""Who spoke when: pyannote segmentation-3.0 + 3D-Speaker embeddings via sherpa-onnx.

    .venv/bin/python bin/diarize.py audio.wav [--speakers N] > speakers.rttm-ish

Prints one line per turn: "start end SPEAKER_k". Pass --speakers when you know
how many people were on the call; it is much more reliable than the threshold.
"""
import sys, pathlib
import soundfile as sf
import sherpa_onnx

ROOT = pathlib.Path(__file__).resolve().parent.parent
D = ROOT / "models" / "diar"
# Same model voices.py names speakers with. The earlier 3D-Speaker ERes2Net
# (Mandarin-trained) merged Ravi into Asha on the Oct 5 call; see
# eval/speakers/.
EMBEDDING = "nemo_en_titanet_large.onnx"


# threads=8: the M4 Pro's performance cores; 58 s vs 75 s at 4, and 12 is slower
# again once work lands on efficiency cores.
def diarize(path, speakers=-1, threshold=0.6, threads=8):
    cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(D / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx")),
            num_threads=threads),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(D / EMBEDDING),
            num_threads=threads),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=speakers, threshold=threshold),
        min_duration_on=0.3, min_duration_off=0.5)
    sd = sherpa_onnx.OfflineSpeakerDiarization(cfg)
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio[:, 0]
    assert sr == sd.sample_rate, f"need {sd.sample_rate} Hz, got {sr}"
    return [(s.start, s.end, s.speaker) for s in sd.process(audio).sort_by_start_time()]


if __name__ == "__main__":
    a = sys.argv[1:]
    n = int(a[a.index("--speakers") + 1]) if "--speakers" in a else -1
    for s, e, k in diarize(a[0], n):
        print(f"{s:8.2f} {e:8.2f} SPEAKER_{k}")
