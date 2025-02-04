"""
Handles STT
"""

from Base_AI_Model import BaseModel
from dotenv import dotenv_values
import os
from pydub import AudioSegment
import numpy as np
import librosa
import io

from transformers import WhisperProcessor, WhisperForConditionalGeneration


config = dotenv_values(".env")


class STT_Model(BaseModel):
    def __init__(self, debug=False, model_sample_rate=16000):
        super().__init__(debug)

        self.stt_pipeline = None
        self.stt_processor = None
        self.model_sample_rate = model_sample_rate

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
            model_name, cache_dir=model_save_dir)

    def stt(self, audio):
        """
        Perform speech-to-text on the given audio
        """

        assert self.stt_pipeline is not None, "STT model not initialised"

        self._debug_print("🔄 Performing STT...")

        audio_array, _ = self._convert_bytes_to_array(audio)

        input_features = self.stt_processor(audio_array, sampling_rate=self.model_sample_rate,
                                            return_tensors="pt").input_features

        predicted_ids = self.stt_pipeline.generate(input_features)

        transcription = self.stt_processor.batch_decode(
            predicted_ids, skip_special_tokens=True)[0]

        self._debug_print(f"🔄 Transcription: {transcription}")
        # Return the first (and only) transcription
        return transcription

    def _convert_bytes_to_array(self, bytes):
        """
        Convert audio blob to numpy array
        """
        audio_file = io.BytesIO(bytes)
        audio = AudioSegment.from_file(
            audio_file, format='webm')  # Format is webm

        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        sampling_rate = audio.frame_rate  # Get sample rate

        if sampling_rate != self.model_sample_rate:
            samples = self._convert_audio_sampling_rate(samples, sampling_rate)

        return samples, sampling_rate

    def _convert_audio_sampling_rate(self, audio_array, sampling_rate):
        """
        Convert the audio sampling rate to the model's sampling rate
        """
        if sampling_rate != self.model_sample_rate:
            self._debug_print(
                f"🔄 Converting audio sampling rate from {sampling_rate} to {self.model_sample_rate}..."
            )

            audio_array = librosa.core.resample(
                y=audio_array, orig_sr=sampling_rate, target_sr=self.model_sample_rate)

        return audio_array


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

    # print("audio", audio)

    transcription = stt.stt(audio)
