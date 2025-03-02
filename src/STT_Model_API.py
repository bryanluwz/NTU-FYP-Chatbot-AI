"""
Handles STT
"""

from src.Base_AI_Model import BaseModel
from dotenv import dotenv_values

from google.cloud import speech
from google.oauth2 import service_account
from pydub import AudioSegment
import io


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

        self.stt_pipeline = speech.SpeechClient(
            credentials=service_account.Credentials.from_service_account_file(self.stt_api_key))

    def stt(self, audio, transcription_model="latest_short"):
        """
        Perform speech-to-text on the given audio
        """

        assert self.stt_pipeline is not None, "STT model not initialised"

        self._debug_print("🔄 Performing STT...")

        audio = AudioSegment.from_file(io.BytesIO(audio), format="webm")

        # Convert to WAV (16-bit PCM)
        audio = audio.set_frame_rate(self.model_sampling_rate).set_channels(
            1).set_sample_width(2)  # 16-bit PCM

        pcm_data = io.BytesIO()

        audio.export(pcm_data, format="raw")

        audio = speech.RecognitionAudio(content=pcm_data.getvalue())
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=self.model_sampling_rate,
            language_code='en-US',
            model=transcription_model
        )

        response = self.stt_pipeline.recognize(config=config, audio=audio)

        for result in response.results:
            return result.alternatives[0].transcript
