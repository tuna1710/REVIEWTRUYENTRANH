"""
Main Orchestration Pipeline for Manga Recap AI Studio
Coordinates all steps: Download -> Panel Extract -> AI Script (Qwen2.5-VL / Gemini) -> TTS + Subtitle -> Video Render / CapCut Export.
Supports both 16:9 Landscape (YouTube) and 9:16 Portrait (TikTok/Shorts/Reels).
"""

import os
import sys
import yaml
import argparse
import shutil
from datetime import datetime
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
    provider: str = None,
    model_name: str = None,
    gemini_api_key: str = None,
    story_synopsis: str = "",
    voice_preset: str = "Thiện Minh",
    reference_audio: str = None,
    export_capcut: bool = False,
    aspect_ratio: str = "16:9",
    config_path: str = "config.yaml"
):
    print("=" * 60)
    print("🔥 MANGA RECAP AI STUDIO - AUTOMATED RECAP PIPELINE 🔥")
    print(f"🎬 Target Aspect Ratio: {aspect_ratio}")
    print("=" * 60)

    # 1. Load Configurations
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}

    work_dir = Path("./workspace")
    work_dir.mkdir(parents=True, exist_ok=True)

    # Clean workspace directories before starting new run to avoid mixing previous chapters
    for sub in ["raw_pages", "panels", "audio", "subtitles", "rendered_scenes"]:
        d = work_dir / sub
        if d.exists():
            for item in d.iterdir():
                try:
                    if item.is_file() or item.is_symlink():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item, ignore_errors=True)
                except Exception:
                    pass
        d.mkdir(parents=True, exist_ok=True)

    # 2. Step 1: Download or Ingest Manga Pages
    print("\n--- [STEP 1/5] Ingesting Manga Chapter ---")
    downloader = MangaDownloader(output_dir=str(work_dir / "raw_pages"))
    if source_input.startswith("http://") or source_input.startswith("https://"):
        page_paths = downloader.download_from_url(source_input)
    else:
        page_paths = downloader.load_from_local_archive(source_input)

    if not page_paths:
        raise RuntimeError("No valid manga pages were loaded. Pipeline stopped.")
    print(f"✅ Loaded {len(page_paths)} manga pages.")

    # 3. Step 2: Panel Extraction & Manga RTL Ordering
    print("\n--- [STEP 2/5] Detecting & Cropping Panels (RTL Reading Order) ---")
    extractor = MangaPanelExtractor(
        output_dir=str(work_dir / "panels"),
        confidence_threshold=cfg.get("panel_extractor", {}).get("confidence_threshold", 0.45),
        reading_order=cfg.get("panel_extractor", {}).get("reading_order", "RTL")
    )
    panels_meta = extractor.process_all_pages(page_paths)
    if not panels_meta:
        raise RuntimeError("No panels were extracted from the manga pages.")
    print(f"✅ Extracted {len(panels_meta)} manga panels.")

    # 4. Step 3: AI Climax Filtering & Scriptwriting
    print("\n--- [STEP 3/5] AI Climax Filtering & Script Generation ---")
    active_key = gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
    if active_key in ["PLACEHOLDER", "your_gemini_api_key_here"]:
        active_key = ""

    active_provider = provider or cfg.get("script_generator", {}).get("provider", "qwen_vl")
    if model_name:
        active_model = model_name
    elif active_provider == "qwen_vl":
        active_model = cfg.get("script_generator", {}).get("qwen_model", "Qwen/Qwen2.5-VL-7B-Instruct")
    else:
        active_model = cfg.get("script_generator", {}).get("gemini_model", "gemini-3.8-flash")

    print(f"🤖 Active Script Generator: Provider={active_provider} | Model={active_model}")

    script_gen = ScriptGenerator(
        provider=active_provider,
        api_key=active_key,
        model_name=active_model,
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
    sub_gen = SubtitleGenerator(
        output_dir=str(work_dir / "subtitles"),
        aspect_ratio=aspect_ratio
    )

    rendered_scene_videos = []
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    ratio_str = aspect_ratio.replace(":", "_")
    final_output_file = work_dir / f"final_recap_{timestamp_str}_{ratio_str}.mp4"

    video_engine = VideoEngine(
        output_dir=str(work_dir / "rendered_scenes"),
        final_output_path=str(final_output_file),
        aspect_ratio=aspect_ratio
    )

    scenes = timeline.get("scenes", [])
    for idx, sc in enumerate(scenes, start=1):
        panel_file_path = str(work_dir / "panels" / sc["panel_file"])
        if not os.path.exists(panel_file_path):
            continue

        print(f"\n🎬 Processing Scene {idx}/{len(scenes)}: {sc['panel_file']}")
        print(f"   💬 Narration: {sc['narration']}")

        # 4a. TTS Audio
        audio_info = tts.synthesize_scene(scene_id=idx, text=sc["narration"])

        # 4b. Word-level Subtitle (.ass)
        sub_info = sub_gen.transcribe_and_generate_sub(
            scene_id=idx,
            audio_path=audio_info["audio_path"],
            reference_text=sc["narration"]
        )

        # 4c. Render Individual Scene MP4
        scene_mp4 = video_engine.render_scene(
            scene_id=idx,
            panel_path=panel_file_path,
            audio_path=audio_info["audio_path"],
            sub_ass_path=sub_info["ass_path"],
            camera_motion=sc.get("camera_motion", "zoom_in"),
            duration=audio_info["duration"]
        )
        rendered_scene_videos.append(scene_mp4)

    # 6. Step 5: Final Video Assembly & BGM Ducking
    print("\n--- [STEP 5/5] Final Video Assembly with Background Music & Ducking ---")
    bgm_path = Path("assets/bgm/eerie_ambient.mp3")
    final_video = video_engine.concatenate_scenes(
        scene_video_paths=rendered_scene_videos,
        bgm_path=str(bgm_path) if bgm_path.exists() else None
    )

    # Optional: CapCut Project Draft Export
    if export_capcut:
        print("\n--- [BONUS] Exporting CapCut / JianYing Project Draft ---")
        capcut_exp = CapCutExporter(output_dir=str(work_dir / "capcut_draft"))
        draft_path = capcut_exp.export_draft(
            timeline_data=timeline,
            panels_dir=str(work_dir / "panels"),
            audio_dir=str(work_dir / "audio"),
            aspect_ratio=aspect_ratio
        )
        print(f"✅ CapCut draft saved at: {draft_path}")

    print("\n" + "=" * 60)
    print(f"🎉 SUCCESS! Final recap video generated at: {final_video}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manga Recap AI Studio")
    parser.add_argument("--input", required=True, help="URL of chapter or path to .zip/.cbz/.pdf/folder")
    parser.add_argument("--provider", default=None, choices=["qwen_vl", "gemini"], help="AI Vision director provider (qwen_vl or gemini)")
    parser.add_argument("--model", default=None, help="Model name (e.g. Qwen/Qwen2.5-VL-7B-Instruct, Qwen/Qwen2.5-VL-3B-Instruct, or gemini-3.8-flash)")
    parser.add_argument("--api-key", default=None, help="Google Gemini API key")
    parser.add_argument("--synopsis", default="", help="Brief story context")
    parser.add_argument("--voice", default="Thiện Minh", help="TTS voice preset")
    parser.add_argument("--ref-audio", default=None, help="Audio file for voice cloning")
    parser.add_argument("--capcut", action="store_true", help="Also export CapCut draft")
    parser.add_argument("--aspect-ratio", default="16:9", choices=["16:9", "9:16"], help="Video aspect ratio: 16:9 or 9:16")
    args = parser.parse_args()

    run_pipeline(
        source_input=args.input,
        provider=args.provider,
        model_name=args.model,
        gemini_api_key=args.api_key,
        story_synopsis=args.synopsis,
        voice_preset=args.voice,
        reference_audio=args.ref_audio,
        export_capcut=args.capcut,
        aspect_ratio=args.aspect_ratio
    )
