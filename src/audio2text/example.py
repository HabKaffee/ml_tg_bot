from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from src.audio2text.utils import load_data_golos, gen_transcriptions_golos, calc_test_metrics, gen_transcription

processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-russian")
model = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-russian")
real_transcriptions = load_data_golos("./audio_files")

# Generate transcriptions for golos data
generated_transcriptions = gen_transcriptions_golos(model, processor)

# Calculate test metrics
calc_test_metrics(real_transcriptions, generated_transcriptions)

generated_transcription = gen_transcription(model, processor, "./audio_files/output_321.wav")
print(generated_transcription)
