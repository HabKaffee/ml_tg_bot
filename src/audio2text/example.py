from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, WhisperForConditionalGeneration, WhisperProcessor
from src.audio2text.utils import (calc_test_metrics, gen_transcription, gen_transcription_whisper, gen_transcriptions_golos,
                   load_data_golos)

# load whisper model and processor
processor_wh = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
model_wh = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2")
model_wh.config.forced_decoder_ids = None

# load model and processor
processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-russian")
model = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-russian")
real_transcriptions = load_data_golos("./audio_files")

# Generate transcriptions for golos data
generated_transcriptions = gen_transcriptions_golos(model, processor, num_samples=10, whisper_flag=False)

# Calculate test metrics
calc_test_metrics(real_transcriptions, generated_transcriptions)

generated_transcription_wh = gen_transcription_whisper(model_wh, processor_wh, "./audio_files/output_1.wav")
generated_transcription = gen_transcription(model, processor, "./audio_files/output_321.wav")
print(generated_transcription)
print(generated_transcription_wh)
