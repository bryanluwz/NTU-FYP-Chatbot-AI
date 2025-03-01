"""
Handles STT
"""

from src.Base_AI_Model import BaseModel
from dotenv import dotenv_values
import os

from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types


config = dotenv_values(".env")


class STT_Model_API(BaseModel):
    def __init__(self, debug=False, model_sample_rate=16000, stt_api_key=None):
        super().__init__(debug)

        self.stt_pipeline = None
        self.model_sampling_rate = model_sample_rate
        self.stt_api_key = stt_api_key

    def initialise_stt(self):
        """
        Initialise the STT pipeline
        Available models:
        """
        assert self.stt_api_key, "STT API key not provided."

        # God knows why google did this
        os.environ["GOOGLE_API_KEY"] = self.stt_api_key

        self.stt_pipeline = speech.SpeechClient()

    def stt(self, audio):
        """
        Perform speech-to-text on the given audio
        """

        assert self.stt_pipeline is not None, "STT model not initialised"

        self._debug_print("🔄 Performing STT...")

        audio = speech.RecognitionAudio(content=audio)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=self.model_sampling_rate,
            language_code='en-US',
        )

        response = self.stt_pipeline.recognize(config=config, audio=audio)

        for result in response.results:
            return result.alternatives[0].transcript
