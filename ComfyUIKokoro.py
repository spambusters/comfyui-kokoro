import onnxruntime as ort

# --- GPU PATCH START ---
_original_inference_session = ort.InferenceSession
def patched_inference_session(*args, **kwargs):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    kwargs['sess_options'] = so
    kwargs['providers'] = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    return _original_inference_session(*args, **kwargs)
ort.InferenceSession = patched_inference_session
# --- GPU PATCH END ---

import numpy as np
import torch
from kokoro_onnx import Kokoro
import logging
import os
import requests
from tqdm import tqdm
import io
import re

logger = logging.getLogger(__name__)

MODEL_URL = "https://github.com/taylorchu/kokoro-onnx/releases/download/v0.2.0/kokoro.onnx"
MODEL_FILENAME = "kokoro_v1.onnx"
VOICES_FILENAME = "voices_v1.bin"

supported_languages_display = ["English", "English (British)","French", "Japanese", "Hindi", "Mandarin Chinese", "Spanish", "Brazilian Portuguese", "Italian"]

supported_languages = {
    supported_languages_display[0]: "en-us",
    supported_languages_display[1]: "en-gb",
    supported_languages_display[2]: "fr-fr",
    supported_languages_display[3]: "ja",
    supported_languages_display[4]: "hi",
    supported_languages_display[5]: "cmn",
    supported_languages_display[6]: "es",
    supported_languages_display[7]: "pt-br",
    supported_languages_display[8]: "it",
}

supported_voices =[
    "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica", "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
    "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael", "am_onyx", "am_puck", "am_santa",
    "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
    "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
    "jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo",
    "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi", "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang",
    "ef_dora", "em_alex", "em_santa", "ff_siwis", "hf_alpha", "hf_beta", "hm_omega", "hm_psi", "if_sara", "im_nicola", "pf_dora", "pm_alex", "pm_santa",
]

def download_file(url, file_name, path):
    if not os.path.exists(path):
        os.makedirs(path)
    with requests.get(url, stream=True, allow_redirects=True) as response:
        total_size = int(response.headers.get('content-length', 0))
        block_size = 4096 
        progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, desc=file_name)
        with open(os.path.join(path, file_name), 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()

def download_voices(path):
    file_path = os.path.join(path, VOICES_FILENAME)
    if os.path.exists(file_path):
        return
    names = supported_voices
    pattern = "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/voices/{name}.pt"
    voices = {}
    for name in names:
        url = pattern.format(name=name)
        r = requests.get(url)
        r.raise_for_status()
        content = io.BytesIO(r.content)
        data: np.ndarray = torch.load(content, weights_only=True).numpy()
        voices[name] = data
    with open(file_path, "wb") as f:
        np.savez(f, **voices)

def download_model(path):
    if os.path.exists(os.path.join(path, MODEL_FILENAME)):
        return
    download_file(MODEL_URL, MODEL_FILENAME, path)

class KokoroSpeaker:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"speaker_name": (supported_voices, {"default": "af_sarah"})}}
    RETURN_TYPES = ("KOKORO_SPEAKER",)
    RETURN_NAMES = ("speaker",)
    FUNCTION = "select"
    CATEGORY = "kokoro"
    def __init__(self):
        self.node_dir = os.path.dirname(os.path.abspath(__file__))
    def select(self, speaker_name):
        download_model(self.node_dir)
        download_voices(self.node_dir)
        kokoro = Kokoro(os.path.join(self.node_dir, MODEL_FILENAME), os.path.join(self.node_dir, VOICES_FILENAME))
        speaker: np.ndarray = kokoro.get_voice_style(speaker_name)
        return ({"speaker": speaker},)

class KokoroSpeakerCombiner:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"speaker_a": ("KOKORO_SPEAKER", ), "speaker_b": ("KOKORO_SPEAKER", ), "weight": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.05})}}
    RETURN_TYPES = ("KOKORO_SPEAKER",)
    RETURN_NAMES = ("speaker",)
    FUNCTION = "combine"
    CATEGORY = "kokoro"
    def combine(self, speaker_a, speaker_b, weight):
        speaker = np.add(speaker_a["speaker"] * weight, speaker_b["speaker"] * (1.0 - weight))
        return ({"speaker": speaker},)

class KokoroGenerator:
    @classmethod
    def INPUT_TYPES(s):
        # UI Default text updated with a Cheat Sheet
        default_text = (
            "Hello! [pause] This is normal. [speed:0.7] [vol:0.6] I am now slow and quiet. "
            "[reset] [pause:1.2s] And I am back to normal."
        )
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": default_text}),
                "speaker": ("KOKORO_SPEAKER", ),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 4.0, "step": 0.05}),
                "lang": (supported_languages_display, {"default": "English"}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "kokoro"

    def __init__(self):
        self.node_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(self.node_dir, MODEL_FILENAME)
        self.voices_path = os.path.join(self.node_dir, VOICES_FILENAME)

    def generate(self, text, speaker, speed, lang):
        download_model(self.node_dir)
        download_voices(self.node_dir)

        lang_code = supported_languages.get(lang, "en-us")
        
        try:
            kokoro = Kokoro(model_path=self.model_path, voices_path=self.voices_path)
        except Exception as e:
             logger.error(f"Error loading Kokoro: {e}")
             return (None,)

        # Unified tag pattern
        tag_pattern = r"(\[pause(?::[\d.]+s)?\]|\[speed:[\d.]+\]|\[vol:[\d.]+\]|\[reset\])"
        parts = re.split(tag_pattern, text)
        
        combined_audio = []
        current_speed = speed 
        current_vol = 1.0
        sample_rate = 24000 

        print(f"--- Kokoro Processing Start ---")

        for part in parts:
            if not part:
                continue
            
            # 1. Handle Tags
            if part.startswith("["):
                if "[pause" in part:
                    pause_match = re.search(r"[\d.]+", part)
                    duration = float(pause_match.group()) if pause_match else 0.5
                    silence_len = int(duration * sample_rate)
                    
                    if silence_len > 0:
                        # 1. Generate the Room Tone
                        # If you find the noise is still too "hissy," you can lower room_tone_vol to 0.0001.
                        # If you want it to sound like a more "lo-fi" recording, you can raise it to 0.001.
                        room_tone_vol = 0.0005 
                        noise = np.random.uniform(-1, 1, silence_len).astype(np.float32) * room_tone_vol
                        
                        # 2. Smooth the edges (Fade in/out) to prevent clicks
                        fade_len = min(int(sample_rate * 0.005), silence_len // 2) # 5ms fade
                        fade_in = np.linspace(0, 1, fade_len)
                        fade_out = np.linspace(1, 0, fade_len)
                        noise[:fade_len] *= fade_in
                        noise[-fade_len:] *= fade_out
                        
                        combined_audio.append(noise)
                
                elif "[speed" in part:
                    speed_match = re.search(r"[\d.]+", part)
                    if speed_match:
                        current_speed = float(speed_match.group())
                
                elif "[vol" in part:
                    vol_match = re.search(r"[\d.]+", part)
                    if vol_match:
                        current_vol = float(vol_match.group())
                
                elif "[reset]" in part:
                    current_speed = speed
                    current_vol = 1.0
                
                continue 

            # 2. Handle Text
            clean_segment = part.strip()
            if clean_segment:
                try:
                    audio, sr = kokoro.create(clean_segment, voice=speaker["speaker"], speed=current_speed, lang=lang_code)
                    if audio is not None:
                        # Scale volume
                        if current_vol != 1.0:
                            audio = audio * current_vol
                        combined_audio.append(audio)
                        sample_rate = sr
                except Exception as e:
                    logger.error(f"Generation error on '{clean_segment}': {e}")

        if not combined_audio:
             return (None,)

        final_audio = np.concatenate(combined_audio)
        audio_tensor = torch.from_numpy(final_audio).unsqueeze(0).unsqueeze(0).float()

        return ({"waveform": audio_tensor, "sample_rate": sample_rate},)

NODE_CLASS_MAPPINGS = {
    "KokoroGenerator": KokoroGenerator,
    "KokoroSpeaker": KokoroSpeaker,
    "KokoroSpeakerCombiner": KokoroSpeakerCombiner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KokoroGenerator": "Kokoro Generator",
    "KokoroSpeaker": "Kokoro Speaker",
    "KokoroSpeakerCombiner": "Kokoro Speaker Combiner",
}
