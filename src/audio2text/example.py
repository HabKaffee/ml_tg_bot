from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from utils import load_data_golos, generate_transcriptions_golos, calc_test_metrics, generate_transcription

processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-russian")
model = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-russian")
real_transcriptions = load_data_golos("./audio_files")

# Generate transcriptions for golos data
generated_transcriptions = generate_transcriptions_golos(model, processor)

# Calculate test metrics
calc_test_metrics(real_transcriptions, generated_transcriptions)

generated_transcription = generate_transcription(model, processor, "./audio_files/output_321.wav")
print(generated_transcription)
