"""
Handles STT
"""

from src.Base_AI_Model import BaseModel
from dotenv import dotenv_values
import os
from pydub import AudioSegment
import numpy as np
import librosa
import io
from torch import ones_like as torch_ones_like

from transformers import WhisperProcessor, WhisperForConditionalGeneration

from torch.cuda import is_available as is_cuda_available


config = dotenv_values(".env")


class STT_Model(BaseModel):
    def __init__(self, debug=False, model_sample_rate=16000, device=None):
        super().__init__(debug)

        self.stt_pipeline = None
        self.stt_processor = None
        self.model_sampling_rate = model_sample_rate
        self.device = device or 'cuda' if is_cuda_available() else 'cpu'

    def initialise_stt(self, model_name: str, model_path: str = None):
        """
        Initialise the STT pipeline
        Available models:
        openai/whisper-base.en
        """

        if not model_name.startswith('openai/whisper'):
            self._debug_print(f"Invalid model name: {model_name}")

            raise ValueError(f"Oh noes! Invalid model name: {model_name}")

        self._debug_print("🔄 Loading STT model...")

        model_save_dir = model_path

        self.stt_processor = WhisperProcessor.from_pretrained(
            model_name, cache_dir=model_save_dir)

        self.stt_pipeline = WhisperForConditionalGeneration.from_pretrained(
            model_name, cache_dir=model_save_dir).to(self.device)

    def stt(self, audio):
        """
        Perform speech-to-text on the given audio
        """

        assert self.stt_pipeline is not None, "STT model not initialised"

        self._debug_print("🔄 Performing STT...")

        audio_array, _ = self._convert_bytes_to_array(audio)

        input_features = self.stt_processor(audio_array, sampling_rate=self.model_sampling_rate,
                                            return_tensors="pt").input_features.to(self.device)

        predicted_ids = self.stt_pipeline.generate(
            input_features, attention_mask=torch_ones_like(input_features).to(self.device), pad_token_id=self.stt_processor.tokenizer.eos_token_id)

        transcription = self.stt_processor.batch_decode(
            predicted_ids, skip_special_tokens=True)[0]

        self._debug_print(f"🔄 Transcription: {transcription}")

        # Return the first (and only) transcription
        return transcription

    def _convert_bytes_to_array(self, audio_bytes):
        """
        Convert audio bytes to a numpy array of float32 samples and ensure the correct sampling rate.
        """
        # Load audio using pydub.AudioSegment
        audio_segment = AudioSegment.from_file(
            io.BytesIO(audio_bytes), format="webm")
        audio_segment = audio_segment.set_channels(1)

        sampling_rate = audio_segment.frame_rate
        samples = audio_segment.get_array_of_samples()

        samples = np.array(samples).T.astype(np.float32)
        samples /= np.iinfo(np.int32).max  # Bruh is int32 not int16

        # Resample if necessary
        if sampling_rate != self.model_sampling_rate:
            self._debug_print(
                f"🔄 Resampling from {sampling_rate} Hz to {self.model_sampling_rate} Hz ..."
            )
            samples = librosa.resample(
                samples, orig_sr=sampling_rate, target_sr=self.model_sampling_rate)
            sampling_rate = self.model_sampling_rate

        return samples, sampling_rate


if __name__ == '__main__':

    # test tts
    STT_MODEL_PATH = os.path.abspath(
        config.get('STT_MODEL_PATH', './models/stt_model'))
    os.makedirs(STT_MODEL_PATH, exist_ok=True)

    STT_MODEL_NAME = 'openai/whisper-base.en'

    stt = STT_Model(debug=True)
    stt.initialise_stt(STT_MODEL_NAME, STT_MODEL_PATH)

    audio_file_path = r'C:\Users\bryan\Documents\GitHub\NTU-FYP-Chatbot-AI\temp.webm'

    # Convert audio to bytes
    audio = open(audio_file_path, 'rb').read()

    transcription = stt.stt(audio)
