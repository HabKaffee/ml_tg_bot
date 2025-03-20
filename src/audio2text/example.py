import torch
from speech_recognition import SpeechRecognition


def main() -> None:

    MODEL_PATH = "openai/whisper-large-v2"
    PROCESSOR_PATH = "openai/whisper-large-v2"

    # device = torch.distributed.get_backend()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    speech_rec = SpeechRecognition(model_path=MODEL_PATH, processor_path=PROCESSOR_PATH, device=device, whisper=True)

    generated_transcription_wh = speech_rec.gen_transcription_whisper("./audio_files/Example.ogg")
    print(generated_transcription_wh)


if __name__ == "__main__":
    main()
