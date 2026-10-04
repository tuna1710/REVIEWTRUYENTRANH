"""
Module 4B: Subtitle Engine (Faster-Whisper)
Extracts word-level timestamps from synthesized audio and generates dynamic .srt / .ass subtitles.
Supports both 16:9 Landscape (YouTube) and 9:16 Portrait (TikTok/Shorts/Reels).
"""

import os
from pathlib import Path
from typing import List, Dict, Optional

try:
    from faster_whisper import WhisperModel
    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False


class SubtitleGenerator:
    def __init__(
        self,
        model_size: str = "small",
        device: str = "cuda",
        compute_type: str = "float16",
        output_dir: str = "./workspace/subtitles",
        aspect_ratio: str = "16:9"
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.aspect_ratio = aspect_ratio
        self.model = None

        if HAS_WHISPER:
            try:
                print(f"[Subtitle Engine] Loading Faster-Whisper ({model_size}) on {device}...")
                self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
            except Exception as e:
                print(f"[Subtitle Engine] CUDA load failed ({e}), loading on CPU int8...")
                self.model = WhisperModel(model_size, device="cpu", compute_type="int8")

    def transcribe_and_generate_sub(self, scene_id: int, audio_path: str, reference_text: str = "") -> Dict:
        """
        Transcribes audio with word-level timestamps and exports .srt and .ass files.
        """
        out_srt = self.output_dir / f"scene_{scene_id:04d}.srt"
        out_ass = self.output_dir / f"scene_{scene_id:04d}.ass"

        words_data = []

        if self.model is not None and os.path.exists(audio_path):
            try:
                segments, info = self.model.transcribe(
                    audio_path,
                    language="vi",
                    word_timestamps=True,
                    beam_size=5
                )

                srt_entries = []
                counter = 1

                for segment in segments:
                    for word in segment.words:
                        words_data.append({
                            "word": word.word.strip(),
                            "start": word.start,
                            "end": word.end
                        })

                    start_str = self._format_timestamp_srt(segment.start)
                    end_str = self._format_timestamp_srt(segment.end)
                    srt_entries.append(f"{counter}\n{start_str} --> {end_str}\n{segment.text.strip()}\n")
                    counter += 1

                with open(out_srt, "w", encoding="utf-8") as f:
                    f.write("\n".join(srt_entries))

            except Exception as e:
                print(f"[Subtitle Engine] Faster-Whisper warning: {e}. Writing fallback subtitle.")
                self._write_simple_srt(out_srt, reference_text, duration=5.0)
        else:
            self._write_simple_srt(out_srt, reference_text, duration=5.0)

        # Generate stylized ASS subtitle tailored to aspect ratio
        self._generate_stylized_ass(out_ass, words_data, fallback_text=reference_text)

        return {
            "srt_path": str(out_srt),
            "ass_path": str(out_ass),
            "words": words_data
        }

    def _format_timestamp_srt(self, seconds: float) -> str:
        millis = int((seconds - int(seconds)) * 1000)
        s = int(seconds)
        m, s = divmod(s, 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d},{millis:03d}"

    def _format_timestamp_ass(self, seconds: float) -> str:
        cs = int((seconds - int(seconds)) * 100) # centiseconds
        s = int(seconds)
        m, s = divmod(s, 60)
        h, m = divmod(m, 60)
        return f"{h:01d}:{m:02d}:{s:02d}.{cs:02d}"

    def _write_simple_srt(self, path: Path, text: str, duration: float):
        end_str = self._format_timestamp_srt(duration)
        content = f"1\n00:00:00,000 --> {end_str}\n{text}\n"
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    def _generate_stylized_ass(self, path: Path, words: List[Dict], fallback_text: str = ""):
        """
        Generates modern YouTube/TikTok-style subtitles:
        - Bold font, Yellow highlight / White text, Deep black shadow/outline.
        - Responsive resolution & margins based on 16:9 vs 9:16.
        """
        if self.aspect_ratio == "9:16":
            res_x, res_y = 1080, 1920
            font_size = 64
            margin_v = 360  # Safe zone for mobile UI buttons
            chunk_size = 4
        else:
            res_x, res_y = 1920, 1080
            font_size = 56
            margin_v = 90
            chunk_size = 5

        header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {res_x}
PlayResY: {res_y}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: RecapDefault,Arial,{font_size},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3.5,2.0,2,40,40,{margin_v},1
Style: HighlightWord,Arial,{font_size + 4},&H0000FFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,4.0,2.5,2,40,40,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        events = []
        if words:
            for i in range(0, len(words), chunk_size):
                chunk = words[i:i + chunk_size]
                start_t = self._format_timestamp_ass(chunk[0]["start"])
                end_t = self._format_timestamp_ass(chunk[-1]["end"])
                sentence = " ".join([w["word"] for w in chunk])
                events.append(f"Dialogue: 0,{start_t},{end_t},RecapDefault,,0,0,0,,{sentence}")
        else:
            events.append(f"Dialogue: 0,0:00:00.00,0:00:05.00,RecapDefault,,0,0,0,,{fallback_text}")

        with open(path, "w", encoding="utf-8") as f:
            f.write(header + "\n".join(events))
