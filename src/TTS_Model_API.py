"""
Handles TTS 
"""

import uuid
from dotenv import dotenv_values
import os
import soundfile as sf
import numpy as np
from google.cloud import texttospeech
from google.oauth2 import service_account

from src.Base_AI_Model import BaseModel

config = dotenv_values(".env")


class TTS_Model_Map:
    # All American English
    model_map = {
        "default": "en-US-Chirp3-HD-Aoede",  # Example default voice
        "Apple": "en-US-Chirp-HD-D",
        "Project Manager": "en-US-Chirp-HD-F",
        "Human Resource": "en-US-Chirp-HD-O",
        "Aoede": "en-US-Chirp3-HD-Aoede",
        "Charon": "en-US-Chirp3-HD-Charon",
        "Fenrir": "en-US-Chirp3-HD-Fenrir",
        "Kore": "en-US-Chirp3-HD-Kore",
        "Leda": "en-US-Chirp3-HD-Leda",
        "Orus": "en-US-Chirp3-HD-Orus",
        "Puck": "en-US-Chirp3-HD-Puck",
        "Zephyr": "en-US-Chirp3-HD-Zephyr",
    }

    @staticmethod
    def get_model(name):
        if name in TTS_Model_Map.model_map.values():
            return name

        return TTS_Model_Map.model_map.get(name, TTS_Model_Map.model_map["default"])


class TTS_Model_API(BaseModel):
    def __init__(self, debug=False, tts_api_key=None):
        super().__init__(debug)
        self.tts_pipeline = None
        self.tts_api_key = tts_api_key

    def initialise_tts(self):
        """
        Initialise the TTS pipeline
        """
        assert self.tts_api_key, "TTS API key not provided."
        self.tts_pipeline = texttospeech.TextToSpeechClient(
            credentials=service_account.Credentials.from_service_account_file(self.tts_api_key))

    def tts(self, tts_name, text):
        """
        Convert text to speech using Google TTS
        """
        if not self.tts_pipeline:
            raise ValueError(
                "TTS pipeline not initialized. Call initialise_tts() first.")

        voice_name = TTS_Model_Map.get_model(tts_name)
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",  # Adjust language as needed
            ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL,
            # Default voice
            name=voice_name if voice_name else TTS_Model_Map.model_map["default"]
        )

        audio_config = texttospeech.AudioConfig(
            # Outputting as raw PCM (16-bit signed linear)
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000  # Ensure the sample rate is 16000 Hz
        )

        synthesis_input = texttospeech.SynthesisInput(text=text)
        response = self.tts_pipeline.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )

        # Convert the byte audio content to numpy array to match the _tts_output_to_file method
        audio_array = np.frombuffer(response.audio_content, dtype=np.int16)

        # Use the _tts_output_to_file method to save the audio file
        output_file = self._tts_output_to_file(
            audio_array, sampling_rate=16000, file_ext="wav")

        return output_file  # Return the filename for further processing

    def _tts_output_to_file(self, audio_array, sampling_rate, file_ext="wav", file_save_dir=None, file_name="tts_output"):
        """
        Save the TTS output to a file
        """
        if file_save_dir is None:
            TEMP_STORAGE_PATH = os.path.abspath(
                config.get('TEMP_STORAGE_PATH', './temp_storage'))
            file_save_dir = TEMP_STORAGE_PATH

        file_name = f"{file_name}_{uuid.uuid4()}"

        final_file_path = os.path.join(
            file_save_dir, f"{file_name}.{file_ext}")

        self._debug_print(f"📁 Saving TTS output to {final_file_path}...")

        with open(f"{final_file_path}", "wb") as _:
            sf.write(final_file_path, audio_array, sampling_rate)

        return final_file_path
