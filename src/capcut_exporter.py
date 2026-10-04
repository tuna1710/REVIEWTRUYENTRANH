"""
Module 5B: CapCut / JianYing Draft Exporter
Exports timeline JSON, images, audio, and subtitles directly into a CapCut PC Project folder.
Users can unzip directly into their CapCut Projects folder to edit transitions, text, and effects.
"""

import os
import json
import shutil
import uuid
import time
from pathlib import Path
from typing import Dict, List, Optional
import soundfile as sf


class CapCutExporter:
    def __init__(self, output_dir: str = "./workspace/capcut_draft"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_draft(
        self,
        timeline_data: Dict,
        panels_dir: str,
        audio_dir: str,
        subtitles_data: Optional[List[Dict]] = None
    ) -> str:
        """
        Generates standard CapCut/JianYing draft folder containing draft_content.json and draft_meta_info.json.
        """
        timestamp_sec = int(time.time())
        draft_id = str(uuid.uuid4()).upper()
        draft_name = f"MangaRecap_{timestamp_sec}"
        draft_root = self.output_dir / draft_name
        draft_root.mkdir(parents=True, exist_ok=True)
        materials_dir = draft_root / "materials"
        materials_dir.mkdir(exist_ok=True)

        print(f"[CapCut Exporter] Building CapCut project draft '{draft_name}' in: {draft_root}...")

        materials_videos = []
        materials_audios = []
        materials_texts = []
        materials_speeds = []

        tracks = [
            {"id": str(uuid.uuid4()).upper(), "type": "video", "segments": []},
            {"id": str(uuid.uuid4()).upper(), "type": "audio", "segments": []},
            {"id": str(uuid.uuid4()).upper(), "type": "text", "segments": []}
        ]

        current_time_us = 0  # microseconds

        for idx, scene in enumerate(timeline_data.get("scenes", []), start=1):
            src_panel_file = Path(panels_dir) / scene["panel_file"]
            src_audio_file = Path(audio_dir) / f"scene_{idx:04d}.wav"

            # Determine duration from audio
            duration_us = 4 * 1000000
            if src_audio_file.exists():
                try:
                    info = sf.info(str(src_audio_file))
                    duration_us = int(info.duration * 1000000)
                except Exception:
                    pass

            # Copy assets locally into draft/materials
            dst_panel = materials_dir / f"panel_{idx:04d}.png"
            if src_panel_file.exists():
                shutil.copy2(src_panel_file, dst_panel)
            else:
                dst_panel = src_panel_file

            dst_audio = materials_dir / f"audio_{idx:04d}.wav"
            if src_audio_file.exists():
                shutil.copy2(src_audio_file, dst_audio)
            else:
                dst_audio = src_audio_file

            # 1. Video / Image Material
            img_mat_id = str(uuid.uuid4()).upper()
            speed_mat_id = str(uuid.uuid4()).upper()
            materials_speeds.append({"id": speed_mat_id, "speed": 1.0, "type": "speed"})
            materials_videos.append({
                "id": img_mat_id,
                "path": str(dst_panel.resolve()),
                "type": "photo",
                "width": 1920,
                "height": 1080
            })
            tracks[0]["segments"].append({
                "id": str(uuid.uuid4()).upper(),
                "material_id": img_mat_id,
                "speed_id": speed_mat_id,
                "target_timerange": {"start": current_time_us, "duration": duration_us}
            })

            # 2. Audio Material
            if dst_audio.exists():
                audio_mat_id = str(uuid.uuid4()).upper()
                materials_audios.append({
                    "id": audio_mat_id,
                    "path": str(dst_audio.resolve()),
                    "type": "extract_music",
                    "duration": duration_us
                })
                tracks[1]["segments"].append({
                    "id": str(uuid.uuid4()).upper(),
                    "material_id": audio_mat_id,
                    "target_timerange": {"start": current_time_us, "duration": duration_us}
                })

            # 3. Subtitle / Text Material
            narration_text = scene.get("narration", "")
            if narration_text:
                text_mat_id = str(uuid.uuid4()).upper()
                materials_texts.append({
                    "id": text_mat_id,
                    "content": narration_text,
                    "type": "text"
                })
                tracks[2]["segments"].append({
                    "id": str(uuid.uuid4()).upper(),
                    "material_id": text_mat_id,
                    "target_timerange": {"start": current_time_us, "duration": duration_us}
                })

            current_time_us += duration_us

        # Build draft_content.json
        draft_content = {
            "id": draft_id,
            "canvas_config": {"width": 1920, "height": 1080, "ratio": "16:9"},
            "duration": current_time_us,
            "fps": 30.0,
            "materials": {
                "videos": materials_videos,
                "audios": materials_audios,
                "texts": materials_texts,
                "speeds": materials_speeds
            },
            "tracks": tracks
        }
        content_path = draft_root / "draft_content.json"
        with open(content_path, "w", encoding="utf-8") as f:
            json.dump(draft_content, f, ensure_ascii=False, indent=2)

        # Build draft_meta_info.json required by CapCut PC
        now_us = int(time.time() * 1000000)
        draft_meta = {
            "draft_id": draft_id,
            "draft_name": draft_name,
            "draft_fold_path": str(draft_root.resolve()),
            "draft_timeline_materials_size": 0,
            "tm_draft_create": now_us,
            "tm_draft_modified": now_us,
            "draft_root_path": str(self.output_dir.resolve()),
            "draft_removable_storage_device": ""
        }
        meta_path = draft_root / "draft_meta_info.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(draft_meta, f, ensure_ascii=False, indent=2)

        print(f"[CapCut Exporter] Draft ready at: {draft_root}")
        return str(draft_root)
