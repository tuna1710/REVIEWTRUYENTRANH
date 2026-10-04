"""
Module 4A: TTS Engine (VieNeu-TTS-v3-Turbo & Edge-TTS Fallback)
High-fidelity 48kHz Vietnamese speech synthesis with emotion tags and instant voice cloning.
"""

import os
import re
import subprocess
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import soundfile as sf

try:
    from vieneu import Vieneu
    HAS_VIENEU = True
except ImportError:
    HAS_VIENEU = False


class TTSEngine:
    def __init__(
        self,
        voice_preset: str = "NamMinh",
        reference_audio: Optional[str] = None,
        output_dir: str = "./workspace/audio",
        sample_rate: int = 48000
    ):
        self.voice_preset = voice_preset
        self.reference_audio = reference_audio
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sample_rate = sample_rate
        self.engine = None

        if HAS_VIENEU:
            print("[TTS Engine] Initializing VieNeu-TTS-v3-Turbo...")
            try:
                self.engine = Vieneu()
            except Exception as e:
                print(f"[TTS Engine] Notice initializing VieNeu: {e}")

    def preview_voice(self, sample_text: str = "Chào mừng bạn đến với Quán Khuya, nơi những câu chuyện rùng rợn bắt đầu...") -> str:
        """
        Quick voice audition without rendering full chapter scenes.
        """
        preview_path = self.output_dir / "voice_sample_preview.wav"
        print(f"[TTS Engine] Synthesizing preview audition: \"{sample_text[:40]}...\"")
        if self.engine is not None:
            try:
                clean_voice = self.voice_preset.split(" ")[0].strip()
                if self.reference_audio and os.path.exists(self.reference_audio):
                    result = self.engine.infer(text=sample_text, ref_audio=self.reference_audio, denoise=True)
                else:
                    result = self.engine.infer(text=sample_text, voice=clean_voice)

                if isinstance(result, tuple) and len(result) == 2:
                    audio_array, sr = result
                else:
                    audio_array = result
                    sr = getattr(self.engine, "sample_rate", self.sample_rate)

                if hasattr(self.engine, "save"):
                    self.engine.save(audio_array, str(preview_path))
                else:
                    sf.write(str(preview_path), audio_array, sr)
                return str(preview_path)
            except Exception as e:
                print(f"[TTS Engine] VieNeu preview notice: {e}, falling back to edge-tts...")

        self._fallback_edge_tts(sample_text, str(preview_path))
        return str(preview_path)

    def synthesize_scene(self, scene_id: int, text: str) -> Dict:
        """
        Synthesizes speech for a single scene and returns the audio path and duration.
        """
        out_filename = f"scene_{scene_id:04d}.wav"
        out_path = self.output_dir / out_filename

        print(f"[TTS Engine] Synthesizing Scene {scene_id}: \"{text[:40]}...\"")

        # 1. Try VieNeu-TTS-v3-Turbo
        if self.engine is not None:
            try:
                clean_voice = self.voice_preset.split(" ")[0].strip()
                if self.reference_audio and os.path.exists(self.reference_audio):
                    result = self.engine.infer(
                        text=text,
                        ref_audio=self.reference_audio,
                        denoise=True
                    )
                else:
                    result = self.engine.infer(
                        text=text,
                        voice=clean_voice
                    )

                if isinstance(result, tuple) and len(result) == 2:
                    audio_array, sr = result
                else:
                    audio_array = result
                    sr = getattr(self.engine, "sample_rate", self.sample_rate)

                if hasattr(self.engine, "save"):
                    self.engine.save(audio_array, str(out_path))
                else:
                    sf.write(str(out_path), audio_array, sr)

                duration = len(audio_array) / float(sr)
                return {"audio_path": str(out_path), "duration": duration}
            except Exception as e:
                print(f"[TTS Engine] VieNeu synthesis error: {e}. Attempting edge-tts fallback...")

        # 2. Fallback using edge-tts (Vietnamese neural voice)
        duration = self._fallback_edge_tts(text, str(out_path))
        return {"audio_path": str(out_path), "duration": duration}

    def _fallback_edge_tts(self, text: str, output_path: str) -> float:
        """
        Clean fallback using edge-tts CLI if VieNeu has an environment issue.
        """
        clean_text = re.sub(r'\[.*?\]', '', text).strip()
        if not clean_text:
            clean_text = "..."

        voice_name = "vi-VN-NamMinhNeural"
        if "bac" in self.voice_preset.lower() or "hoaimy" in self.voice_preset.lower():
            voice_name = "vi-VN-HoaiMyNeural"

        cmd = [
            "edge-tts",
            "--voice", voice_name,
            "--text", clean_text,
            "--write-media", output_path
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            # Generate placeholder tone proportional to text length (~0.35s per word)
            word_count = max(1, len(clean_text.split()))
            est_duration = max(3.0, word_count * 0.35)
            num_samples = int(self.sample_rate * est_duration)
            t = np.linspace(0, est_duration, num_samples, endpoint=False)
            dummy = (0.05 * np.sin(2 * np.pi * 120 * t)).astype(np.float32)
            sf.write(output_path, dummy, self.sample_rate)

        # Get duration using soundfile
        info = sf.info(output_path)
        return info.duration
