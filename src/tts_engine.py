"""
Module 4A: TTS Engine (VieNeu-TTS-v3-Turbo & Edge-TTS Fallback)
High-fidelity 48kHz Vietnamese speech synthesis with emotion tags and instant voice cloning.
Supports full official VieNeu voice catalog (Thiện Minh, Minh Đức, Hải Đăng, Quang Sơn, Thùy Dung, etc.).
"""

import os
import sys

# Comprehensive sanitization of NO_PROXY to prevent httpx/huggingface IPv6 port parsing crash
for _k in ["NO_PROXY", "no_proxy", "GLOBAL_AGENT_NO_PROXY"]:
    if _k in os.environ:
        parts = [p.strip() for p in os.environ[_k].split(",") if p.strip() and "::" not in p and "[" not in p]
        os.environ[_k] = ",".join(parts)

import re
import shutil
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
        self.clean_output_dir()

    def _fix_hf_cache_symlinks(self):
        """
        Fixes HuggingFace Hub symlink issue on Linux/Colab where ONNX Runtime rejects
        external data file paths ('External data path escapes model directory').
        Replaces cross-directory symlinks in the snapshot directory with hardlinks or copies.
        """
        try:
            hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
            if not hf_cache.exists():
                return
            for snap_dir in hf_cache.glob("**/snapshots/*"):
                for p in snap_dir.rglob("*"):
                    if p.is_symlink():
                        target = p.resolve()
                        if target.exists():
                            p.unlink()
                            try:
                                os.link(target, p)
                            except Exception:
                                shutil.copyfile(target, p)
        except Exception:
            pass

    def clean_output_dir(self):
        """Cleans previous audio files to avoid mixing audio between chapters."""
        if self.output_dir.exists():
            for item in self.output_dir.iterdir():
                try:
                    if item.is_file() or item.is_symlink():
                        if item.name.startswith("scene_"):
                            item.unlink()
                except Exception:
                    pass
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if HAS_VIENEU:
            print("[TTS Engine] Initializing VieNeu-TTS-v3-Turbo...")
            try:
                self.engine = Vieneu()
            except Exception as e:
                err_msg = str(e).lower()
                if "escapes model directory" in err_msg or "external data path" in err_msg or "symlink" in err_msg:
                    print("[TTS Engine] Resolving ONNX model cache layout for VieNeu...")
                    self._fix_hf_cache_symlinks()
                    try:
                        self.engine = Vieneu()
                        print("[TTS Engine] VieNeu-TTS successfully initialized after cache resolution.")
                    except Exception as e2:
                        print(f"[TTS Engine] Notice initializing VieNeu after relink: {e2}")
                else:
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
        Reliable fallback using edge_tts Python API with proxy support and 48kHz stereo WAV transcoding.
        """
        clean_text = re.sub(r'\[.*?\]', '', text).strip()
        if not clean_text:
            clean_text = "..."

        female_keywords = ["dung", "anh", "huyen", "tran", "ly", "linh", "trang", "doan", "duyen", "thanh"]
        clean_voice = self._resolve_voice(self.voice_preset).lower()

        if any(k in clean_voice for k in female_keywords):
            voice_name = "vi-VN-HoaiMyNeural"
        else:
            voice_name = "vi-VN-NamMinhNeural"

        temp_mp3 = Path(output_path).with_suffix(".temp.mp3")
        proxy = (
            os.environ.get("https_proxy") or os.environ.get("http_proxy") or
            os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY") or None
        )

        success = False

        # Attempt A: Direct edge_tts Python library
        try:
            import edge_tts
            import asyncio

            async def _synthesize():
                comm = edge_tts.Communicate(clean_text, voice_name, proxy=proxy)
                await comm.save(str(temp_mp3))

            try:
                asyncio.run(_synthesize())
            except RuntimeError:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    try:
                        import nest_asyncio
                        nest_asyncio.apply()
                    except ImportError:
                        pass
                loop.run_until_complete(_synthesize())

            if temp_mp3.exists() and temp_mp3.stat().st_size > 100:
                trans_cmd = [
                    "ffmpeg", "-y",
                    "-i", str(temp_mp3),
                    "-ar", "48000", "-ac", "2",
                    "-c:a", "pcm_s16le",
                    str(output_path)
                ]
                subprocess.run(trans_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                success = True
        except Exception as py_err:
            # Attempt B: CLI invocation fallback
            try:
                cmd = [
                    sys.executable, "-m", "edge_tts",
                    "--voice", voice_name,
                    "--text", clean_text,
                    "--write-media", str(temp_mp3)
                ]
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                trans_cmd = [
                    "ffmpeg", "-y",
                    "-i", str(temp_mp3),
                    "-ar", "48000", "-ac", "2",
                    "-c:a", "pcm_s16le",
                    str(output_path)
                ]
                subprocess.run(trans_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                success = True
            except Exception as cli_err:
                print(f"[TTS Engine Notice] edge-tts Python ({py_err}) and CLI ({cli_err}) unavailable.")

        temp_mp3.unlink(missing_ok=True)

        if not success or not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            print("[TTS Engine Notice] Both VieNeu and Edge-TTS unavailable (offline/blocked). Generating audible preview tone.")
            word_count = max(1, len(clean_text.split()))
            est_duration = max(3.0, word_count * 0.35)
            num_samples = int(self.sample_rate * est_duration)
            t = np.linspace(0, est_duration, num_samples, endpoint=False)
            dummy = (0.25 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
            sf.write(output_path, dummy, self.sample_rate)

        info = sf.info(output_path)
        return info.duration
