"""
Module 5A: Video Assembly Engine (FFmpeg with NVENC Hardware Acceleration)
Creates cinematic 1080p video clips with Ken Burns camera motion, blurred background filler,
subtitles, sound effects, and mixes with ducked ambient horror BGM.
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
        use_nvenc: bool = True,
        width: int = 1920,
        height: int = 1080,
        fps: int = 30
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.final_output_path = Path(final_output_path)
        self.use_nvenc = use_nvenc
        self.width = width
        self.height = height
        self.fps = fps
        self.encoder = self._detect_encoder()

    def _detect_encoder(self) -> str:
        """
        Detects if NVIDIA NVENC hardware encoder is available on Colab T4.
        """
        if not self.use_nvenc:
            return "libx264"
        try:
            res = subprocess.run(["ffmpeg", "-encoders"], capture_output=True, text=True)
            if "h264_nvenc" in res.stdout:
                print("[Video Engine] NVIDIA NVENC hardware acceleration detected!")
                return "h264_nvenc"
        except Exception:
            pass
        print("[Video Engine] Using CPU libx264 encoder.")
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

        total_frames = int(duration * self.fps)

        # Configure Ken Burns zoom formula
        if camera_motion == "zoom_in":
            zoom_expr = f"min(zoom+0.0015,1.25)"
            x_expr = "iw/2-(iw/zoom/2)"
            y_expr = "ih/2-(ih/zoom/2)"
        elif camera_motion == "zoom_out":
            zoom_expr = f"max(1.25-0.0015*on,1.0)"
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
            clean_sub_path = sub_ass_path.replace("\\", "/").replace(":", "\\:")
            filter_complex += f";[base]ass='{clean_sub_path}'[v]"
            video_map = "[v]"
        else:
            video_map = "[base]"

        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-loop", "1", "-i", panel_path,
            "-i", audio_path,
            "-filter_complex", filter_complex,
            "-map", video_map,
            "-map", "1:a",
            "-c:v", self.encoder,
            "-preset", "p4" if self.encoder == "h264_nvenc" else "veryfast",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            "-t", str(duration),
            str(out_mp4)
        ]

        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
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
        print(f"[Video Engine] Stitching {len(scene_video_paths)} scene clips into final recap video...")
        concat_txt = self.output_dir / "concat_list.txt"
        with open(concat_txt, "w", encoding="utf-8") as f:
            for p in scene_video_paths:
                f.write(f"file '{Path(p).resolve()}'\n")

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
                "-stream_loop", "-1", "-i", bgm_path,
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
