"""
Module 5A: Video Assembly Engine (FFmpeg with NVENC Hardware Acceleration)
Creates cinematic 1080p video clips with Ken Burns camera motion, blurred background filler,
subtitles, sound effects, and mixes with ducked ambient horror BGM.
Supports both 16:9 Landscape (YouTube) and 9:16 Portrait (TikTok/Shorts/Reels).
"""

import os
import json
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict, Optional


class VideoEngine:
    def __init__(
        self,
        output_dir: str = "./workspace/rendered_scenes",
        final_output_path: str = "./workspace/final_recap_video.mp4",
        aspect_ratio: str = "16:9",
        use_nvenc: bool = True,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: int = 30
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.final_output_path = Path(final_output_path)
        self.aspect_ratio = aspect_ratio
        
        # Determine resolution based on aspect ratio
        if width is not None and height is not None:
            self.width = width
            self.height = height
        elif aspect_ratio == "9:16":
            self.width = 1080
            self.height = 1920
        else:
            self.width = 1920
            self.height = 1080

        self.use_nvenc = use_nvenc
        self.fps = fps
        self.encoder = self._detect_encoder()

    def _detect_encoder(self) -> str:
        """
        Detects if NVIDIA NVENC hardware encoder is actively functional on Colab T4.
        """
        if not self.use_nvenc:
            return "libx264"
        try:
            # Probe with a fast 0.05s null test to confirm active GPU hardware encoding
            test_cmd = [
                "ffmpeg", "-y", "-f", "lavfi", "-i", "nullsrc=s=64x64:d=0.05",
                "-c:v", "h264_nvenc", "-f", "null", "-"
            ]
            res = subprocess.run(test_cmd, capture_output=True)
            if res.returncode == 0:
                print("[Video Engine] NVIDIA NVENC hardware acceleration confirmed & ready!")
                return "h264_nvenc"
        except Exception:
            pass
        print("[Video Engine] NVENC unavailable or no active GPU, using CPU libx264 encoder.")
        return "libx264"

    def render_scene(
        self,
        scene_id: int,
        panel_path: str,
        audio_path: str,
        sub_ass_path: Optional[str] = None,
        camera_motion: str = "zoom_in",
        duration: Optional[float] = None
    ) -> str:
        """
        Renders a single scene with Ken Burns pan/zoom, blurred background, and subtitles.
        """
        out_mp4 = self.output_dir / f"scene_{scene_id:04d}.mp4"

        # Determine audio duration if not provided
        if duration is None or duration <= 0:
            probe_cmd = f"ffprobe -v error -show_entries format=duration -of json \"{audio_path}\""
            try:
                res = subprocess.check_output(probe_cmd, shell=True)
                duration = float(json.loads(res)['format']['duration'])
            except Exception:
                duration = 5.0 # fallback

        total_frames = max(1, int(duration * self.fps))

        # Configure Ken Burns zoom formula
        if camera_motion == "zoom_in":
            zoom_expr = "min(zoom+0.0015,1.25)"
            x_expr = "iw/2-(iw/zoom/2)"
            y_expr = "ih/2-(ih/zoom/2)"
        elif camera_motion == "zoom_out":
            zoom_expr = "max(1.25-0.0015*on,1.0)"
            x_expr = "iw/2-(iw/zoom/2)"
            y_expr = "ih/2-(ih/zoom/2)"
        elif camera_motion == "pan_left":
            zoom_expr = "1.2"
            x_expr = f"max(0, (iw-iw/zoom)*(1-on/{total_frames}))"
            y_expr = "ih/2-(ih/zoom/2)"
        else: # static / default
            zoom_expr = "1.05"
            x_expr = "iw/2-(iw/zoom/2)"
            y_expr = "ih/2-(ih/zoom/2)"

        # Escape path for FFmpeg subtitles filter
        filter_complex = (
            f"[0:v]scale={self.width}:{self.height}:force_original_aspect_ratio=increase,"
            f"crop={self.width}:{self.height},boxblur=20:5[bg];"
            f"[0:v]scale={self.width}:{self.height}:force_original_aspect_ratio=decrease[scaled_fg];"
            f"[scaled_fg]zoompan=z='{zoom_expr}':d={total_frames}:x='{x_expr}':y='{y_expr}':s={self.width}x{self.height}:fps={self.fps}[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2[base]"
        )

        if sub_ass_path and os.path.exists(sub_ass_path):
            abs_sub = Path(sub_ass_path).resolve().as_posix()
            escaped_sub = abs_sub.replace(":", r"\:")
            filter_complex += f";[base]ass='{escaped_sub}'[v]"
            video_map = "[v]"
        else:
            video_map = "[base]"

        def _build_cmd(encoder: str):
            return [
                "ffmpeg", "-y",
                "-loop", "1", "-i", str(Path(panel_path).resolve()),
                "-i", str(Path(audio_path).resolve()),
                "-filter_complex", filter_complex,
                "-map", video_map,
                "-map", "1:a",
                "-c:v", encoder,
                "-preset", "p4" if encoder == "h264_nvenc" else "veryfast",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k",
                "-t", str(duration),
                str(out_mp4)
            ]

        try:
            cmd = _build_cmd(self.encoder)
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            if self.encoder == "h264_nvenc":
                print("[Video Engine] NVENC encoding failed, retrying scene with CPU libx264...")
                self.encoder = "libx264"
                cmd = _build_cmd("libx264")
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            else:
                # If ASS subtitle filter failed, retry without ASS filter
                print(f"[Video Engine] Render failed ({e}), retrying without ASS overlay...")
                clean_filter = filter_complex.split(";[base]ass")[0]
                fallback_cmd = [
                    "ffmpeg", "-y",
                    "-loop", "1", "-i", str(Path(panel_path).resolve()),
                    "-i", str(Path(audio_path).resolve()),
                    "-filter_complex", clean_filter,
                    "-map", "[base]",
                    "-map", "1:a",
                    "-c:v", "libx264",
                    "-preset", "veryfast",
                    "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "192k",
                    "-t", str(duration),
                    str(out_mp4)
                ]
                subprocess.run(fallback_cmd, check=True)

        return str(out_mp4)

    def concatenate_scenes(
        self,
        scene_video_paths: List[str],
        bgm_path: Optional[str] = None,
        bgm_volume_db: int = -18
    ) -> str:
        """
        Concatenates all scene MP4 files and blends background music (BGM).
        """
        print(f"[Video Engine] Stitching {len(scene_video_paths)} scene clips ({self.aspect_ratio}) into final video...")
        concat_txt = self.output_dir / "concat_list.txt"
        with open(concat_txt, "w", encoding="utf-8") as f:
            for p in scene_video_paths:
                f.write(f"file '{Path(p).resolve().as_posix()}'\n")

        temp_combined = self.output_dir / "temp_combined.mp4"

        # Concat demuxer (lossless & instant)
        cmd_concat = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_txt),
            "-c", "copy",
            str(temp_combined)
        ]
        subprocess.run(cmd_concat, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Blend BGM if present
        if bgm_path and os.path.exists(bgm_path):
            print(f"[Video Engine] Mixing ambient horror BGM: {bgm_path} (volume: {bgm_volume_db}dB)...")
            cmd_mix = [
                "ffmpeg", "-y",
                "-i", str(temp_combined),
                "-stream_loop", "-1", "-i", str(Path(bgm_path).resolve()),
                "-filter_complex",
                f"[1:a]volume={bgm_volume_db}dB[bgm];[0:a][bgm]amix=inputs=2:duration=first[a]",
                "-map", "0:v",
                "-map", "[a]",
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k",
                str(self.final_output_path)
            ]
            subprocess.run(cmd_mix, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            temp_combined.unlink(missing_ok=True)
        else:
            shutil.move(str(temp_combined), str(self.final_output_path))

        print(f"[Video Engine] Master video rendered successfully: {self.final_output_path}")
        return str(self.final_output_path)
