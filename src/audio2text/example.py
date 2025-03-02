from huggingsound import SpeechRecognitionModel
from utils import *

model = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-russian")
real_transcriptions = load_data_golos("./audio_files")

# Generate transcriptions for golos data
generated_transcriptions = generate_transcriptions_golos(model)

# Calculate test metrics
calc_test_metrics(real_transcriptions, generated_transcriptions)

generated_transcription = generate_transcription(model, "./audio_files/output_321.wav")
print(generated_transcription)
