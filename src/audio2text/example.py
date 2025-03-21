from pathlib import Path

from src.audio2text.speech_recognition import SpeechRecognition


def main() -> None:

    MODEL_PATH = "openai/whisper-large-v2"
    PROCESSOR_PATH = "openai/whisper-large-v2"

    speech_rec = SpeechRecognition(model_path=MODEL_PATH, processor_path=PROCESSOR_PATH, whisper=True)

    generated_transcription_wh = speech_rec.gen_transcription(Path("./audio_files/Example.ogg"))
    print(generated_transcription_wh)


if __name__ == "__main__":
    main()
