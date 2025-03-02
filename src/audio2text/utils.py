from datasets import load_dataset
from scipy.io.wavfile import write
import tqdm
import numpy as np
from jiwer import wer, cer
import soundfile as sf
import time

def load_data_golos(path_to_audio):
    ds = load_dataset("bond005/sberdevices_golos_10h_crowd")
    data_ = ds['test']
    real_transcriptions = []
    for i, smpl in tqdm.tqdm(enumerate(data_)):
        filename = path_to_audio + f"/output_{i}.wav"
        write(filename, smpl["audio"]["sampling_rate"], smpl["audio"]["array"])
        real_transcriptions.append(smpl['transcription'])
        # Audio(smpl["audio"]["array"], rate=smpl["audio"]["sampling_rate"])
    return real_transcriptions

def generate_transcriptions_golos(model, num_samples=100):
    audio_paths = [f"./audio_files/output_{i}.wav" for i in range(num_samples)]
    transcriptions = model.transcribe(audio_paths)
    generated_transcriptions = [tr['transcription'] for tr in transcriptions]
    return generated_transcriptions

def calc_test_metrics(real_transcriptions, generated_transcriptions):
    nan_indices = [i for i, val in enumerate(real_transcriptions) if val is None or (isinstance(val, float) and np.isnan(val))]
    generated_transcriptions = [val for i, val in enumerate(generated_transcriptions) if i not in nan_indices]
    real_transcriptions = [val for i, val in enumerate(real_transcriptions) if i not in nan_indices]
    len_ = len(generated_transcriptions)# len(generated_transcriptions)
    print("WER:", wer(real_transcriptions[:len_], generated_transcriptions[:len_]))
    print("CER:", cer(real_transcriptions[:len_], generated_transcriptions[:len_]))

def generate_transcription(model, path_to_file):
    transcriptions = model.transcribe([path_to_file])
    generated_transcriptions = [tr['transcription'] for tr in transcriptions]
    return generated_transcriptions

def ogg_to_wav(path_to_file, path_to_new_file):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    data, samplerate = sf.read(path_to_file)
    sf.write(path_to_new_file + f'/{timestr}wav_transformed.wav', data, samplerate)