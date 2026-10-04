"""
Main Orchestration Pipeline for Manga Recap AI Studio
Coordinates all steps: Download -> Panel Extract -> AI Script -> TTS + Subtitle -> Video Render / CapCut Export.
"""

import os
import sys
import yaml
import argparse
from pathlib import Path

# Fix httpx NO_PROXY bug with IPv6 brackets
for _k in ["NO_PROXY", "no_proxy"]:
    if _k in os.environ and "::1" in os.environ[_k]:
        os.environ[_k] = os.environ[_k].replace(",[::1]", "").replace(",::1", "").replace("[::1]", "").replace("::1", "")

# Add src to python path
sys.path.append(str(Path(__file__).parent))

from src.downloader import MangaDownloader
from src.panel_extractor import MangaPanelExtractor
from src.script_generator import ScriptGenerator
from src.tts_engine import TTSEngine
from src.subtitle_generator import SubtitleGenerator
from src.video_engine import VideoEngine
from src.capcut_exporter import CapCutExporter


def run_pipeline(
    source_input: str,
    gemini_api_key: str = None,
    story_synopsis: str = "",
    voice_preset: str = "NamMinh",
    reference_audio: str = None,
    export_capcut: bool = False,
    config_path: str = "config.yaml"
):
    print("=" * 60)
    print("🔥 MANGA RECAP AI STUDIO - AUTOMATED RECAP PIPELINE 🔥")
    print("=" * 60)

    # 1. Load Configurations
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}

    work_dir = Path("./workspace")
    work_dir.mkdir(parents=True, exist_ok=True)

    # 2. Step 1: Download or Ingest Manga Pages
    print("\n--- [STEP 1/5] Ingesting Manga Chapter ---")
    downloader = MangaDownloader(output_dir=str(work_dir / "raw_pages"))
    if source_input.startswith("http://") or source_input.startswith("https://"):
        page_paths = downloader.download_from_url(source_input)
    else:
        page_paths = downloader.load_from_local_archive(source_input)

    if not page_paths:
        print("❌ Error: No pages found. Aborting.")
        return

    # 3. Step 2: Extract & Sort Manga Panels
    print("\n--- [STEP 2/5] Extracting & Sorting Manga Panels (RTL) ---")
    extractor = MangaPanelExtractor(
        output_dir=str(work_dir / "panels"),
        reading_order=cfg.get("panel_extractor", {}).get("reading_order", "RTL")
    )
    panels_meta = extractor.process_all_pages(page_paths)

    # 4. Step 3: AI Climax Filtering & Scriptwriting
    print("\n--- [STEP 3/5] AI Climax Filtering & Script Generation ---")
    active_key = gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
    if active_key in ["PLACEHOLDER", "your_gemini_api_key_here"]:
        active_key = ""

    script_gen = ScriptGenerator(
        provider=cfg.get("script_generator", {}).get("provider", "gemini"),
        api_key=active_key,
        model_name=cfg.get("script_generator", {}).get("gemini_model", "gemini-3.8-flash"),
        output_file=str(work_dir / "timeline.json")
    )
    timeline = script_gen.generate_timeline(
        panels_metadata=panels_meta,
        story_synopsis=story_synopsis,
        min_score=cfg.get("script_generator", {}).get("min_drama_score", 7),
        max_scenes=cfg.get("script_generator", {}).get("max_selected_panels", 35)
    )

    # 5. Step 4: TTS & Subtitle Generation
    print("\n--- [STEP 4/5] VieNeu-TTS Speech & Faster-Whisper Subtitles ---")
    tts = TTSEngine(
        voice_preset=voice_preset,
        reference_audio=reference_audio,
        output_dir=str(work_dir / "audio")
    )
    sub_gen = SubtitleGenerator(output_dir=str(work_dir / "subtitles"))

    rendered_scene_videos = []
    video_engine = VideoEngine(
        output_dir=str(work_dir / "rendered_scenes"),
        final_output_path=str(work_dir / "final_recap_video.mp4")
    )

    scenes = timeline.get("scenes", [])
    for idx, sc in enumerate(scenes, start=1):
        panel_file_path = str(work_dir / "panels" / sc["panel_file"])
        if not os.path.exists(panel_file_path):
            continue

        # TTS audio
        audio_info = tts.synthesize_scene(scene_id=idx, text=sc["narration"])

        # Subtitle
        sub_info = sub_gen.transcribe_and_generate_sub(
            scene_id=idx,
            audio_path=audio_info["audio_path"],
            reference_text=sc["narration"]
        )

        # Render individual scene MP4
        scene_mp4 = video_engine.render_scene(
            scene_id=idx,
            panel_path=panel_file_path,
            audio_path=audio_info["audio_path"],
            sub_ass_path=sub_info["ass_path"],
            camera_motion=sc.get("camera_motion", "zoom_in"),
            duration=audio_info["duration"]
        )
        rendered_scene_videos.append(scene_mp4)

    # 6. Step 5: Final Video Assembly / Optional CapCut Draft
    print("\n--- [STEP 5/5] Final Video Assembly ---")
    bgm_file = "./assets/bgm/eerie_ambient.mp3"
    final_video = video_engine.concatenate_scenes(
        scene_video_paths=rendered_scene_videos,
        bgm_path=bgm_file if os.path.exists(bgm_file) else None
    )

    if export_capcut:
        print("\n--- Exporting CapCut Project Draft ---")
        capcut_exp = CapCutExporter(output_dir=str(work_dir / "capcut_draft"))
        draft_path = capcut_exp.export_draft(
            timeline_data=timeline,
            panels_dir=str(work_dir / "panels"),
            audio_dir=str(work_dir / "audio")
        )
        print(f"✅ CapCut draft saved at: {draft_path}")

    print("\n" + "=" * 60)
    print(f"🎉 SUCCESS! Final recap video generated at: {final_video}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manga Recap AI Studio")
    parser.add_argument("--input", required=True, help="URL of chapter or path to .zip/.cbz/.pdf/folder")
    parser.add_argument("--api-key", default=None, help="Google Gemini API key")
    parser.add_argument("--synopsis", default="", help="Brief story context")
    parser.add_argument("--voice", default="NamMinh", help="TTS voice preset")
    parser.add_argument("--ref-audio", default=None, help="Audio file for voice cloning")
    parser.add_argument("--capcut", action="store_true", help="Also export CapCut draft")
    args = parser.parse_args()

    run_pipeline(
        source_input=args.input,
        gemini_api_key=args.api_key,
        story_synopsis=args.synopsis,
        voice_preset=args.voice,
        reference_audio=args.ref_audio,
        export_capcut=args.capcut
    )
