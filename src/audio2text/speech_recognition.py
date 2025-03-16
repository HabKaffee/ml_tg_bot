import os
import time
from typing import List, cast

import librosa
import numpy as np
import soundfile as sf
import torch
import tqdm
from datasets import load_dataset
from jiwer import cer, wer
from scipy.io.wavfile import write
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, WhisperForConditionalGeneration, WhisperProcessor


class SpeechRecognition:
    def __init__(self, model_path: str, processor_path: str, device: str = "cpu", whisper: bool = False):
        self.device = device
        self.whisper = whisper

        if whisper:
            self.model = WhisperForConditionalGeneration.from_pretrained(model_path).to(self.device)
            self.processor = WhisperProcessor.from_pretrained(processor_path)
        else:
            self.model = Wav2Vec2ForCTC.from_pretrained(model_path).to(self.device)
            self.processor = Wav2Vec2Processor.from_pretrained(processor_path)

    def load_data_golos(self, path_to_audio: str) -> List[str]:
        """Loads test dataset from GOLOS and saves audio samples to files."""
        ds = load_dataset("bond005/sberdevices_golos_10h_crowd")
        data_ = ds["test"]
        real_transcriptions = []
        for i, smpl in tqdm.tqdm(enumerate(data_)):
            filename = os.path.join(path_to_audio, f"output_{i}.wav")
            write(filename, smpl["audio"]["sampling_rate"], smpl["audio"]["array"])
            real_transcriptions.append(smpl["transcription"])
        return real_transcriptions

    def gen_transcriptions_golos(self, num_samples: int = 10, folder_to_audio: str = "audio_files") -> List[str]:
        """Generates transcriptions for stored audio files using the speech model."""
        transcriptions = []
        audio_paths = [os.path.join(folder_to_audio, f"output_{i}.wav") for i in range(num_samples)]
        for audio_path in audio_paths:
            transcription = self._gen_transcription(audio_path)
            transcriptions.append(transcription)
        return transcriptions

    def _gen_transcription(self, path_to_file: str) -> str:
        """Generates a transcription for a single audio file."""
        if not os.path.exists(path_to_file):
            raise FileNotFoundError(f"No file {path_to_file}")

        if self.whisper:
            return self._gen_transcription_whisper(path_to_file)
        return self._gen_transcription_wav2vec(path_to_file)

    def _gen_transcription_wav2vec(self, path_to_file: str) -> str:
        speech_array, _ = librosa.load(path_to_file, sr=16_000)
        inputs = self.processor(speech_array, sampling_rate=16_000, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        logits = self.model(inputs.input_values, attention_mask=inputs.attention_mask).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        decoded = self.processor.batch_decode(predicted_ids)
        transcription = cast(str, decoded[0]) if decoded else ""
        return transcription

    def _gen_transcription_whisper(self, path_to_file: str) -> str:
        allowed_formats = {".wav", ".ogg"}
        _, ext = os.path.splitext(path_to_file)
        ext = ext.lower()
        if ext not in allowed_formats:
            raise ValueError(f"Unsupported file format '{ext}'. Allowed formats: {', '.join(allowed_formats)}")

        if ext == ".ogg":
            path_to_file = self._ogg_to_wav(path_to_file)

        audio_input, _ = librosa.load(path_to_file, sr=16000)
        input_features = self.processor(audio_input, sample_rate=16000, return_tensors="pt").input_features
        input_features = input_features.to(self.device)
        generated_ids = self.model.generate(input_features)
        decoded = self.processor.decode(generated_ids[0], skip_special_tokens=True)
        transcription = cast(str, decoded) if decoded else ""
        return transcription

    def _ogg_to_wav(self, path_to_file: str) -> str:
        timestr = time.strftime("%Y%m%d-%H%M%S")
        data, samplerate = sf.read(path_to_file)
        if not os.path.exists("./temp/"):
            os.makedirs("./temp/")
        file_name = f"./temp/{timestr}_wav_transformed.wav"
        sf.write(file_name, data, samplerate)
        return file_name

    def calc_test_metrics(self, real_transcriptions: List[str], generated_transcriptions: List[str]) -> None:
        """Calculates Word Error Rate (WER) and Character Error Rate (CER)."""
        nan_indices = [
            i for i, val in enumerate(real_transcriptions) if val is None or (isinstance(val, float) and np.isnan(val))
        ]
        generated_transcriptions = [val for i, val in enumerate(generated_transcriptions) if i not in nan_indices]
        real_transcriptions = [val for i, val in enumerate(real_transcriptions) if i not in nan_indices]
        len_ = len(generated_transcriptions)
        print("WER:", wer(real_transcriptions[:len_], generated_transcriptions[:len_]))
        print("CER:", cer(real_transcriptions[:len_], generated_transcriptions[:len_]))
