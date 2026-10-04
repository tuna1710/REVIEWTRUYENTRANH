"""
Module 4A: TTS Engine (VieNeu-TTS-v3-Turbo & Edge-TTS Fallback)
High-fidelity 48kHz Vietnamese speech synthesis with emotion tags and instant voice cloning.
Supports full official VieNeu voice catalog (Thiện Minh, Minh Đức, Hải Đăng, Quang Sơn, Thùy Dung, etc.).
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

# Official VieNeu Voice Catalog
VIENEU_VOICES = [
    'Thiện Minh', 'Minh Đức', 'Hải Đăng', 'Quang Sơn', 'Thùy Dung',
    'Mai Anh', 'Phạm Tuyên', 'Thái Sơn', 'Xuân Vĩnh', 'Thanh Bình',
    'Ngọc Linh', 'Đoan Trang', 'Thục Đoan', 'Minh Triết', 'Mỹ Duyên',
    'Quỳnh Anh', 'Đức Trí', 'Kim Thanh', 'Adam', 'Quốc Tuấn', 'Trúc Ly',
    'Thiền Tâm Đức', 'Ngọc Huyền', 'Ngọc Trân', 'Adam bựa'
]


class TTSEngine:
    def __init__(
        self,
        voice_preset: str = "Thiện Minh",
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

    def _resolve_voice(self, name: str) -> str:
        """
        Resolves voice name from UI string or legacy aliases to official VieNeu voice name.
        """
        if not name:
            return "Thiện Minh"

        # Check exact matches against VieNeu catalog
        for v in VIENEU_VOICES:
            if v.lower() in name.lower():
                return v

        # Legacy alias mappings
        lower = name.lower()
        if "bacminh" in lower or "bac" in lower:
            return "Minh Đức"
        if "namminh" in lower or "nam" in lower:
            return "Thiện Minh"
        if "trung" in lower:
            return "Hải Đăng"
        if "nu" in lower or "female" in lower:
            return "Thùy Dung"

        return "Thiện Minh"

    def preview_voice(self, sample_text: str = "Chào mừng bạn đến với Quán Khuya, nơi những câu chuyện rùng rợn bắt đầu...") -> str:
        """
        Quick voice audition without rendering full chapter scenes.
        """
        preview_path = self.output_dir / "voice_sample_preview.wav"
        clean_voice = self._resolve_voice(self.voice_preset)
        print(f"[TTS Engine] Synthesizing preview audition (Voice: '{clean_voice}'): \"{sample_text[:40]}...\"")

        if self.engine is not None:
            try:
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
        clean_voice = self._resolve_voice(self.voice_preset)

        print(f"[TTS Engine] Synthesizing Scene {scene_id} (Voice: '{clean_voice}'): \"{text[:40]}...\"")

        # 1. Try VieNeu-TTS-v3-Turbo
        if self.engine is not None:
            try:
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
                print(f"[TTS Engine] VieNeu synthesis notice: {e}. Attempting edge-tts fallback...")

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

        # Select female or male edge-tts neural voice based on preset
        female_keywords = ["dung", "anh", "huyen", "tran", "ly", "linh", "trang", "doan", "duyen", "thanh"]
        clean_voice = self._resolve_voice(self.voice_preset).lower()

        if any(k in clean_voice for k in female_keywords):
            voice_name = "vi-VN-HoaiMyNeural"
        else:
            voice_name = "vi-VN-NamMinhNeural"

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
