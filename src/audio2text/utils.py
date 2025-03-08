import os
import time
from typing import cast

import librosa  # pylint: disable=import-error
import numpy as np
import soundfile as sf
import torch
import tqdm
from datasets import load_dataset
from jiwer import cer, wer
from scipy.io.wavfile import write
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, WhisperForConditionalGeneration, WhisperProcessor


def load_data_golos(path_to_audio: str) -> list[str]:
    """Loads test dataset from GOLOS and saves audio samples to files."""
    ds = load_dataset("bond005/sberdevices_golos_10h_crowd")
    data_ = ds["test"]
    real_transcriptions = []
    for i, smpl in tqdm.tqdm(enumerate(data_)):
        filename = path_to_audio + f"/output_{i}.wav"
        write(filename, smpl["audio"]["sampling_rate"], smpl["audio"]["array"])
        real_transcriptions.append(smpl["transcription"])
        # Audio(smpl["audio"]["array"], rate=smpl["audio"]["sampling_rate"])
    return real_transcriptions


def gen_transcriptions_golos(
    model: Wav2Vec2ForCTC, processor: Wav2Vec2Processor, num_samples: int = 10, whisper_flag: bool = False
) -> list[str]:
    """Generates transcriptions for stored audio files using a speech model."""
    transcriptions = []
    audio_paths = [f"./audio_files/output_{i}.wav" for i in range(num_samples)]
    for audio_path in audio_paths:
        if whisper_flag:
            transcription = gen_transcription_whisper(model, processor, audio_path)
        else:
            transcription = gen_transcription(model, processor, audio_path)
        transcriptions.append(transcription)
    return transcriptions


def calc_test_metrics(real_transcriptions: list[str], generated_transcriptions: list[str]) -> None:
    """Calculates Word Error Rate (WER) and Character Error Rate (CER)."""
    nan_indices = [
        i for i, val in enumerate(real_transcriptions) if val is None or (isinstance(val, float) and np.isnan(val))
    ]
    generated_transcriptions = [val for i, val in enumerate(generated_transcriptions) if i not in nan_indices]
    real_transcriptions = [val for i, val in enumerate(real_transcriptions) if i not in nan_indices]
    len_ = len(generated_transcriptions)  # len(generated_transcriptions)
    print("WER:", wer(real_transcriptions[:len_], generated_transcriptions[:len_]))
    print("CER:", cer(real_transcriptions[:len_], generated_transcriptions[:len_]))


def gen_transcription(model: Wav2Vec2ForCTC, processor: Wav2Vec2Processor, path_to_file: str) -> str:
    """Generates a transcription for a single audio file."""
    if os.path.exists(path_to_file):
        speech_array, _ = librosa.load(path_to_file, sr=16_000)
        inputs = processor(speech_array, sampling_rate=16_000, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        decoded = processor.batch_decode(predicted_ids)
        transcription = cast(str, decoded[0]) if decoded else ""
        return transcription
    raise FileNotFoundError(f"No file {path_to_file}")


def gen_transcription_whisper(
    model: WhisperForConditionalGeneration, processor: WhisperProcessor, path_to_file: str
) -> str:
    """Generates a transcription for a single audio file."""
    if os.path.exists(path_to_file):
        audio_input, _ = librosa.load(path_to_file, sr=16000)
        input_features = processor(audio_input, return_tensors="pt").input_features
        with torch.no_grad():
            generated_ids = model.generate(input_features)
        transcription = processor.decode(generated_ids[0], skip_special_tokens=True)
        transcription = str(transcription) if transcription else ""
        return transcription
    raise FileNotFoundError(f"No file {path_to_file}")


def ogg_to_wav(path_to_file: str, path_to_new_file: str) -> None:
    """Converts an OGG audio file to WAV format."""
    timestr = time.strftime("%Y%m%d-%H%M%S")
    data, samplerate = sf.read(path_to_file)
    sf.write(path_to_new_file + f"/{timestr}_wav_transformed.wav", data, samplerate)
