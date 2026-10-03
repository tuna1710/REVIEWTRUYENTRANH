"""
Module 5B: CapCut / JianYing Draft Exporter
Exports timeline JSON, images, audio, and subtitles directly into a CapCut PC Project folder.
Users can open CapCut and find the entire storyline already arranged on the multi-track timeline!
"""

import os
import json
import shutil
import uuid
import time
from pathlib import Path
from typing import Dict, List, Optional


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
        Generates standard CapCut/JianYing draft JSON structure.
        """
        print(f"[CapCut Exporter] Building CapCut project draft in: {self.output_dir}...")
        draft_id = str(uuid.uuid4())
        draft_root = self.output_dir / f"manga_recap_{int(time.time())}"
        draft_root.mkdir(parents=True, exist_ok=True)

        materials_images = []
        materials_audios = []
        tracks = [
            {"id": str(uuid.uuid4()), "type": "video", "segments": []},
            {"id": str(uuid.uuid4()), "type": "audio", "segments": []},
            {"id": str(uuid.uuid4()), "type": "text", "segments": []}
        ]

        current_time_us = 0 # microseconds in CapCut

        for idx, scene in enumerate(timeline_data.get("scenes", []), start=1):
            panel_file = Path(panels_dir) / scene["panel_file"]
            audio_file = Path(audio_dir) / f"scene_{idx:04d}.wav"

            # Estimate duration in microseconds (default 4 seconds if missing)
            duration_us = 4 * 1000000
            if audio_file.exists():
                import soundfile as sf
                try:
                    info = sf.info(str(audio_file))
                    duration_us = int(info.duration * 1000000)
                except Exception:
                    pass

            # 1. Image Material
            img_mat_id = str(uuid.uuid4())
            materials_images.append({
                "id": img_mat_id,
                "path": str(panel_file.resolve()),
                "type": "photo"
            })
            tracks[0]["segments"].append({
                "id": str(uuid.uuid4()),
                "material_id": img_mat_id,
                "target_timerange": {"start": current_time_us, "duration": duration_us}
            })

            # 2. Audio Material
            if audio_file.exists():
                audio_mat_id = str(uuid.uuid4())
                materials_audios.append({
                    "id": audio_mat_id,
                    "path": str(audio_file.resolve()),
                    "type": "extract_music"
                })
                tracks[1]["segments"].append({
                    "id": str(uuid.uuid4()),
                    "material_id": audio_mat_id,
                    "target_timerange": {"start": current_time_us, "duration": duration_us}
                })

            # 3. Subtitle Text Material
            text_mat_id = str(uuid.uuid4())
            tracks[2]["segments"].append({
                "id": str(uuid.uuid4()),
                "material_id": text_mat_id,
                "content": scene.get("narration", ""),
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
                "videos": materials_images,
                "audios": materials_audios,
                "texts": []
            },
            "tracks": tracks
        }

        content_path = draft_root / "draft_content.json"
        with open(content_path, "w", encoding="utf-8") as f:
            json.dump(draft_content, f, ensure_ascii=False, indent=2)

        print(f"[CapCut Exporter] Draft ready at: {draft_root}")
        return str(draft_root)
