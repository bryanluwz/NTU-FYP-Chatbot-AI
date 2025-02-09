"""
Handles TTS 
"""

import uuid
from dotenv import dotenv_values
import os
from torch.cuda import is_available as is_cuda_available
from transformers import pipeline
from kokoro import KPipeline
import soundfile as sf
import numpy as np

from src.Base_AI_Model import BaseModel

config = dotenv_values(".env")


class TTS_Model_Map:
    # All American English
    model_map = {
        "default": "hexgrad/Kokoro-82M?af_heart",
        "Heart": "hexgrad/Kokoro-82M?af_heart",  # hey alexa
        "Bella": "hexgrad/Kokoro-82M?af_bella",  # alexa but throat is dry
        "Nicole": "hexgrad/Kokoro-82M?af_nicole",  # that annoying tiktok asmr voiceover
        "Santa": "hexgrad/Kokoro-82M?am_santa",  # santa claus but he's unemployed
        "Michael": "hexgrad/Kokoro-82M?am_michael",  # generic tts voice no 69420
        "Facebook": "facebook/mms-tts-eng",  # david attenborough but he had a stroke
    }

    @staticmethod
    def get_model(name):
        if name in TTS_Model_Map.model_map.values():
            return name

        return TTS_Model_Map.model_map.get(name, TTS_Model_Map.model_map["default"])


class TTS_Model(BaseModel):
    def __init__(self, debug=False, device=None):
        super().__init__(debug)
        self.tts_pipelines = {}
        self.active_tts_pipeline = None

        self.device = device or 'cuda' if is_cuda_available() else 'cpu'

    def initialise_tts(self, model_name: str, model_path: str = None, task: str = 'text-to-speech'):
        """
        Initialise the TTS pipeline
        Available models:
        hexgrad/Kokoro-82M
        facebook/mms-tts-eng
        """

        actual_model_name = TTS_Model_Map.get_model(model_name)

        if "hexgrad/Kokoro-82M" in actual_model_name:
            self.initialise_kokoro_tts(model_name)
            return

        match actual_model_name:
            case "facebook/mms-tts-eng":
                self.initialise_normal_tts(model_name, model_path)
            case _:
                self._debug_print(
                    f"🚫 Model '{model_name}' not found. Using default model...")

                self.initialise_normal_tts('default', model_path, task)

    def initialise_kokoro_tts(self, model_name: str):
        """
        Initialise the TTS pipeline using the Kokoro pipeline
        """

        self._debug_print(f"🔄 Loading Kokoro model")
        voice = TTS_Model_Map.get_model(
            model_name).split('?')[-1] or 'af_heart'
        lang_code = voice[0]
        self.tts_pipelines[model_name] = KPipeline(
            lang_code=lang_code)

    def initialise_normal_tts(self, model_name: str, model_path: str = None, task: str = 'text-to-speech'):
        """
        Initialise the TTS pipeline using the normal pipeline
        """
        model_save_path = os.path.join(model_path, model_name)

        if os.path.exists(model_save_path):
            self._debug_print(f"🔄 Loading model from {model_save_path}...")

            try:
                self.tts_pipelines[model_name] = pipeline(
                    task=task, model=model_save_path,
                    device_map=self.device)
            except:
                self.tts_pipelines[model_name] = pipeline(
                    task=task, model=TTS_Model_Map.get_model(model_name),
                    device_map='cpu')

        else:
            self._debug_print(
                f"⬇️ Downloading and saving model '{model_name}' to {model_save_path}...")

            try:
                tts_pipeline = pipeline(
                    task=task, model=TTS_Model_Map.get_model(model_name),
                    device_map=self.device)
            except:
                tts_pipeline = pipeline(
                    task=task, model=TTS_Model_Map.get_model(model_name),
                    device_map='cpu')

            tts_pipeline.model.save_pretrained(
                model_save_path)

            self.tts_pipelines[model_name] = tts_pipeline

            self._debug_print(
                f"✅ Model '{model_name}' saved to {model_save_path}.")

    def tts(self, tts_name, tts_model_path, text):
        """
        Text-to-Speech
        """
        if tts_name not in self.tts_pipelines.keys():
            self.initialise_tts(tts_name, tts_model_path)

        self._debug_print(f"🔊 Text-to-Speech using '{tts_name}'...")

        self.active_tts_pipeline = self.tts_pipelines[tts_name]

        if "hexgrad/Kokoro-82M" in TTS_Model_Map.get_model(tts_name):
            voice = TTS_Model_Map.get_model(
                tts_name).split('?')[-1]

            final_audio = []
            for _, _, audio in self.active_tts_pipeline(text, voice=voice):
                # Assume that there is only one audio output in this weired for loop
                final_audio += audio

            self._debug_print(f"✅ Text-to-Speech completed.")
            file_path = self._tts_output_to_file(
                final_audio, 24000, file_name=f"{tts_name}_tts_output")

        else:
            audio = self.active_tts_pipeline(text)

            self._debug_print(f"✅ Text-to-Speech completed.")

            audio_array = np.array(audio["audio"]).squeeze()
            sampling_rate = audio['sampling_rate']
            file_path = self._tts_output_to_file(
                audio_array, sampling_rate, file_name=f"{tts_name}_tts_output")

        return file_path

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

    def stt(self):
        pass


if __name__ == '__main__':

    # test tts
    TTS_MODEL_PATH = os.path.abspath(
        config.get('TTS_MODEL_PATH', './models/tts_model'))
    os.makedirs(TTS_MODEL_PATH, exist_ok=True)

    speech_model = TTS_Model(debug=True)

    for voice in TTS_Model_Map.model_map.keys():
        MODEL_NAME = voice

        # speech_model.initialise_tts(MODEL_NAME, )
        audio_file_path = speech_model.tts(MODEL_NAME, TTS_MODEL_PATH,
                                           "Testing the text-to-speech functionality of the Kokoro model."
                                           )
        print(audio_file_path)
