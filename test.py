from transformers import TFTacotron2Model, Wav2Vec2ForCTC
import torch
from TTS.utils.synthesizer import Synthesizer

# Downloading Tacotron 2 model
tacotron2_model = TFTacotron2Model.from_pretrained("tacotron2-pretrained-model-name")

# Downloading WaveGlow model
waveglow_model = Wav2Vec2ForCTC.from_pretrained("waveglow-pretrained-model-name")


# Load pre-trained models
synthesizer = Synthesizer(
    tts_checkpoint=tacotron2_model,
    vocoder_checkpoint=waveglow_model,
    use_cuda=False
)

# Text to clone
text = "नमस्ते, यह एक उदाहरण है।"

# Synthesize speech
wav = synthesizer.tts(text, speaker_wav=r"C:\Users\harsh\Desktop\3rd year\Sem 5\SGP-2\WavFiles\HarshHindiRecording10sec.wav", language="hi")

# Save the output
torch.save(wav, "output.wav")

