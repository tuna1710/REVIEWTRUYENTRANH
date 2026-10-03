"""
Module 4A: TTS Engine (VieNeu-TTS-v3-Turbo)
High-fidelity 48kHz Vietnamese speech synthesis with emotion tags and instant voice cloning.
"""

import os
import subprocess
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
            print(f"[TTS Engine] Initializing VieNeu-TTS-v3-Turbo...")
            try:
                self.engine = Vieneu()
            except Exception as e:
                print(f"[TTS Engine] Notice initializing VieNeu: {e}")

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
                # Handle reference audio cloning if available
                if self.reference_audio and os.path.exists(self.reference_audio):
                    audio_array, sr = self.engine.infer(
                        text=text,
                        reference_audio=self.reference_audio
                    )
                else:
                    audio_array, sr = self.engine.infer(
                        text=text,
                        voice=self.voice_preset
                    )
                sf.write(str(out_path), audio_array, sr)
                duration = len(audio_array) / float(sr)
                return {"audio_path": str(out_path), "duration": duration}
            except Exception as e:
                print(f"[TTS Engine] VieNeu synthesis error: {e}. Attempting edge-tts fallback...")

        # 2. Fallback using edge-tts (Vietnamese Nam voice: vi-VN-NamMinhNeural)
        duration = self._fallback_edge_tts(text, str(out_path))
        return {"audio_path": str(out_path), "duration": duration}

    def _fallback_edge_tts(self, text: str, output_path: str) -> float:
        """
        Clean fallback using edge-tts CLI if VieNeu has an environment issue.
        """
        # Clean custom emotion tags like [thở dài] for edge-tts
        import re
        clean_text = re.sub(r'\[.*?\]', '', text).strip()
        if not clean_text:
            clean_text = "..."

        cmd = [
            "edge-tts",
            "--voice", "vi-VN-NamMinhNeural",
            "--text", clean_text,
            "--write-media", output_path
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            # Create a silent placeholder wav if CLI edge-tts is missing
            dummy = [0.0] * (self.sample_rate * 3) # 3 seconds silence
            sf.write(output_path, dummy, self.sample_rate)

        # Get duration using soundfile
        info = sf.info(output_path)
        return info.duration
