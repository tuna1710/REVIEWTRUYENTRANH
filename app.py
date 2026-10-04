"""
Manga Recap AI Studio - Gradio Web UI (Phase 2 Enhanced)
Giao diện trực quan chạy trực tiếp trên Google Colab GPU T4.
Bao gồm 2 chế độ:
- Chế độ 1: ⚡ Tạo Nhanh Tự Động (1-Click Auto Recap)
- Chế độ 2: 🎬 Đạo Diễn Tương Tác (Studio Director Mode): Xem Gallery panels, chỉnh sửa kịch bản trực tiếp trên bảng, tùy chọn 16:9 / 9:16 Shorts.
"""

import os
import sys
import json
import os
for _k in ["NO_PROXY", "no_proxy"]:
    if _k in os.environ and "::1" in os.environ[_k]:
        os.environ[_k] = os.environ[_k].replace(",[::1]", "").replace(",::1", "").replace("[::1]", "").replace("::1", "")

import yaml
import shutil
import zipfile
from pathlib import Path
import pandas as pd
import gradio as gr
from datetime import datetime

# Fix httpx NO_PROXY bug with IPv6 brackets
for _k in ["NO_PROXY", "no_proxy"]:
    if _k in os.environ and "::1" in os.environ[_k]:
        os.environ[_k] = os.environ[_k].replace(",[::1]", "").replace(",::1", "").replace("[::1]", "").replace("::1", "")

BASE_DIR = Path(__file__).parent.resolve()
sys.path.append(str(BASE_DIR))

from src.downloader import MangaDownloader
from src.panel_extractor import MangaPanelExtractor
from src.script_generator import ScriptGenerator
from src.tts_engine import TTSEngine
from src.subtitle_generator import SubtitleGenerator
from src.video_engine import VideoEngine
from src.capcut_exporter import CapCutExporter


def detect_storage_dir(prefer_drive: bool = True) -> Path:
    drive_path = Path("/content/drive/MyDrive/MangaRecap")
    if prefer_drive and Path("/content/drive/MyDrive").exists():
        drive_path.mkdir(parents=True, exist_ok=True)
        return drive_path
    local_path = Path("/content/workspace") if Path("/content").exists() else BASE_DIR / "workspace"
    local_path.mkdir(parents=True, exist_ok=True)
    return local_path


# --- VOICE PREVIEW HELPER ---
def preview_voice_sample(voice_preset: str, ref_audio):
    ref_audio_path = ref_audio.name if ref_audio is not None else None
    storage_dir = detect_storage_dir(prefer_drive=False)
    audio_dir = storage_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    tts = TTSEngine(
        voice_preset=voice_preset,
        reference_audio=ref_audio_path,
        output_dir=str(audio_dir)
    )
    sample_text = "Chào mừng bạn đến với Quán Khuya. Đừng bao giờ quay đầu lại nếu bạn nghe thấy tiếng bước chân phía sau..."
    audio_file = tts.preview_voice(sample_text)
    return audio_file


# --- MODE 1: 1-CLICK AUTO RECAP ---
def process_auto_pipeline(
    manga_url: str,
    local_file,
    gemini_key: str,
    model_choice: str,
    aspect_ratio: str,
    synopsis: str,
    voice_preset: str,
    ref_audio,
    dramatic_threshold: int,
    max_panels: int,
    camera_motion_style: str,
    export_capcut_check: bool,
    save_to_drive_check: bool,
    progress=gr.Progress(track_tqdm=True)
):
    storage_dir = detect_storage_dir(prefer_drive=save_to_drive_check)
    raw_dir = storage_dir / "raw_pages"
    panels_dir = storage_dir / "panels"
    audio_dir = storage_dir / "audio"
    sub_dir = storage_dir / "subtitles"
    scenes_dir = storage_dir / "rendered_scenes"

    # Clean intermediate directories to prevent mixing with previous chapter
    for d in [raw_dir, panels_dir, audio_dir, sub_dir, scenes_dir]:
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

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    ratio_str = "9_16" if "9:16" in aspect_choice else "16_9"
    final_video_path = storage_dir / f"final_recap_{timestamp_str}_{ratio_str}.mp4"
    draft_dir = storage_dir / "capcut_draft"

    status_log = []
    def log(msg):
        print(msg)
        status_log.append(msg)
        return "\n".join(status_log)

    # 1. Download
    progress(0.1, desc="Đang nạp truyện tranh...")
    log(f"📁 Thư mục lưu trữ: {storage_dir}")
    downloader = MangaDownloader(output_dir=str(raw_dir))

    if manga_url and manga_url.strip().startswith("http"):
        log(f"📥 Đang cào ảnh từ URL: {manga_url}")
        pages = downloader.download_from_url(manga_url.strip())
    elif local_file is not None:
        log(f"📦 Đang xử lý file tải lên: {local_file.name}")
        pages = downloader.load_from_local_archive(local_file.name)
    else:
        return None, None, None, log("❌ Lỗi: Vui lòng nhập link truyện tranh hoặc tải lên file .zip/.cbz/.pdf!")

    if not pages:
        return None, None, None, log("❌ Không tìm thấy trang truyện nào!")
    log(f"✅ Đã chuẩn bị {len(pages)} trang truyện gốc.")

    # 2. Extract panels
    progress(0.25, desc="Đang nhận diện và cắt ô tranh (RTL)...")
    extractor = MangaPanelExtractor(output_dir=str(panels_dir), reading_order="RTL")
    panels_meta = extractor.process_all_pages(pages)
    log(f"✅ Đã cắt thành công {len(panels_meta)} panels ô tranh.")

    # 3. AI Scriptwriting
    clean_model_name = model_choice.split(" ")[0].strip() if model_choice else "Qwen/Qwen2.5-VL-7B-Instruct"
    provider_to_use = "qwen_vl" if "qwen" in clean_model_name.lower() else "gemini"
    progress(0.45, desc=f"Đạo diễn AI ({clean_model_name}) đang viết kịch bản...")
    api_key_to_use = gemini_key.strip() if gemini_key else os.environ.get("GEMINI_API_KEY", "")
    timeline_file = storage_dir / "timeline.json"

    script_gen = ScriptGenerator(
        provider=provider_to_use,
        api_key=api_key_to_use,
        model_name=clean_model_name,
        output_file=str(timeline_file)
    )

    timeline = script_gen.generate_timeline(
        panels_metadata=panels_meta,
        story_synopsis=synopsis,
        min_score=dramatic_threshold,
        max_scenes=max_panels
    )
    selected_scenes = timeline.get("scenes", [])
    log(f"🎬 Kịch bản: '{timeline.get('title', '')}' ({len(selected_scenes)} cảnh)")

    # 4. TTS & Video Rendering
    progress(0.65, desc="VieNeu-TTS đang đọc & dựng từng cảnh...")
    ref_audio_path = ref_audio.name if ref_audio is not None else None
    tts = TTSEngine(
        voice_preset=voice_preset,
        reference_audio=ref_audio_path,
        output_dir=str(audio_dir)
    )
    sub_gen = SubtitleGenerator(
        output_dir=str(sub_dir),
        aspect_ratio=aspect_ratio
    )
    video_engine = VideoEngine(
        output_dir=str(scenes_dir),
        final_output_path=str(final_video_path),
        aspect_ratio=aspect_ratio
    )

    rendered_scenes = []
    for idx, sc in enumerate(selected_scenes, start=1):
        panel_file_path = str(panels_dir / sc["panel_file"])
        if not os.path.exists(panel_file_path):
            continue

        progress(0.65 + 0.25 * (idx / max(1, len(selected_scenes))), desc=f"Dựng Scene {idx}/{len(selected_scenes)}...")
        audio_info = tts.synthesize_scene(scene_id=idx, text=sc["narration"])
        sub_info = sub_gen.transcribe_and_generate_sub(
            scene_id=idx,
            audio_path=audio_info["audio_path"],
            reference_text=sc["narration"]
        )
        motion = camera_motion_style if camera_motion_style != "auto" else sc.get("camera_motion", "zoom_in")
        scene_mp4 = video_engine.render_scene(
            scene_id=idx,
            panel_path=panel_file_path,
            audio_path=audio_info["audio_path"],
            sub_ass_path=sub_info["ass_path"],
            camera_motion=motion,
            duration=audio_info["duration"]
        )
        rendered_scenes.append(scene_mp4)

    # 5. Concatenate & CapCut
    progress(0.92, desc="Ghép video & hòa âm BGM...")
    bgm_path = BASE_DIR / "assets" / "bgm" / "eerie_ambient.mp3"
    final_output = video_engine.concatenate_scenes(
        scene_video_paths=rendered_scenes,
        bgm_path=str(bgm_path) if bgm_path.exists() else None
    )
    log(f"🎉 Xuất video hoàn tất ({aspect_ratio}): {final_output}")

    capcut_zip_path = None
    if export_capcut_check:
        log("✂️ Đang đóng gói dự án CapCut PC...")
        capcut_exp = CapCutExporter(output_dir=str(draft_dir))
        draft_res = capcut_exp.export_draft(
            timeline_data=timeline,
            panels_dir=str(panels_dir),
            audio_dir=str(audio_dir),
            aspect_ratio=aspect_ratio
        )
        capcut_zip_path = str(storage_dir / f"CapCut_Draft_{aspect_ratio.replace(':', '_')}.zip")
        with zipfile.ZipFile(capcut_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(draft_res):
                for file in files:
                    file_p = Path(root) / file
                    zipf.write(file_p, arcname=file_p.relative_to(Path(draft_res).parent))
        log(f"📦 Đã xuất file CapCut zip: {capcut_zip_path}")

    first_audio = str(audio_dir / "scene_0001.wav") if (audio_dir / "scene_0001.wav").exists() else None
    return str(final_output), first_audio, capcut_zip_path, "\n".join(status_log)


# --- MODE 2: INTERACTIVE STUDIO DIRECTOR ---
def step1_interactive_extract(manga_url: str, local_file, progress=gr.Progress(track_tqdm=True)):
    storage_dir = detect_storage_dir(prefer_drive=False)
    raw_dir = storage_dir / "raw_pages"
    panels_dir = storage_dir / "panels"
    for d in [raw_dir, panels_dir]:
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

    downloader = MangaDownloader(output_dir=str(raw_dir))
    if manga_url and manga_url.strip().startswith("http"):
        pages = downloader.download_from_url(manga_url.strip())
    elif local_file is not None:
        pages = downloader.load_from_local_archive(local_file.name)
    else:
        return [], "❌ Vui lòng nhập link hoặc upload file .zip/.cbz/.pdf!", ""

    extractor = MangaPanelExtractor(output_dir=str(panels_dir), reading_order="RTL")
    panels_meta = extractor.process_all_pages(pages)

    gallery_items = [p["file_path"] for p in panels_meta]
    panels_json = json.dumps(panels_meta, ensure_ascii=False)
    status = f"✅ Đã trích xuất thành công {len(panels_meta)} ô tranh theo thứ tự đọc Manga (RTL)!"
    return gallery_items, status, panels_json


def step2_interactive_generate_script(
    panels_json_str: str,
    gemini_key: str,
    model_choice: str,
    synopsis: str,
    dramatic_threshold: int,
    max_panels: int,
    progress=gr.Progress(track_tqdm=True)
):
    if not panels_json_str:
        return "", "", pd.DataFrame(), "❌ Hãy hoàn thành Bước 1 (Trích xuất ô tranh) trước!"

    panels_meta = json.loads(panels_json_str)
    storage_dir = detect_storage_dir(prefer_drive=False)
    timeline_file = storage_dir / "timeline.json"

    clean_model_name = model_choice.split(" ")[0].strip() if model_choice else "Qwen/Qwen2.5-VL-7B-Instruct"
    provider_to_use = "qwen_vl" if "qwen" in clean_model_name.lower() else "gemini"
    api_key_to_use = gemini_key.strip() if gemini_key else os.environ.get("GEMINI_API_KEY", "")

    script_gen = ScriptGenerator(
        provider=provider_to_use,
        api_key=api_key_to_use,
        model_name=clean_model_name,
        output_file=str(timeline_file)
    )

    timeline = script_gen.generate_timeline(
        panels_metadata=panels_meta,
        story_synopsis=synopsis,
        min_score=dramatic_threshold,
        max_scenes=max_panels
    )

    rows = []
    for sc in timeline.get("scenes", []):
        rows.append({
            "Khung Tranh": sc.get("panel_file", ""),
            "Điểm Kịch Tính": sc.get("dramatic_score", 8),
            "Lời Bình Dẫn Chuyện (Có thể sửa)": sc.get("narration", ""),
            "Camera Motion": sc.get("camera_motion", "zoom_in"),
            "Âm Thanh SFX": sc.get("sfx_cue", "heart_beat")
        })

    df = pd.DataFrame(rows)
    return (
        timeline.get("title", "Kịch Bản Review Truyện"),
        timeline.get("intro_hook", "Mở đầu kịch tính..."),
        df,
        f"✅ AI đã hoàn tất phân tích và chọn {len(rows)} cảnh đắt giá. Bạn có thể sửa trực tiếp bảng dưới đây!"
    )


def step3_interactive_render(
    edited_df: pd.DataFrame,
    video_title: str,
    intro_hook: str,
    aspect_ratio: str,
    voice_preset: str,
    ref_audio,
    export_capcut: bool,
    progress=gr.Progress(track_tqdm=True)
):
    if edited_df is None or edited_df.empty:
        return None, None, "❌ Chưa có nội dung kịch bản để dựng video!"

    storage_dir = detect_storage_dir(prefer_drive=False)
    panels_dir = storage_dir / "panels"
    audio_dir = storage_dir / "audio"
    sub_dir = storage_dir / "subtitles"
    scenes_dir = storage_dir / "rendered_scenes"

    for d in [audio_dir, sub_dir, scenes_dir]:
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

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    ratio_str = "9_16" if "9:16" in aspect_ratio else "16_9"
    final_video_path = storage_dir / f"final_recap_{timestamp_str}_{ratio_str}.mp4"
    draft_dir = storage_dir / "capcut_draft"

    # Reconstruct timeline from edited dataframe
    scenes = []
    for _, row in edited_df.iterrows():
        scenes.append({
            "panel_file": str(row.get("Khung Tranh", "")),
            "dramatic_score": int(row.get("Điểm Kịch Tính", 8)),
            "narration": str(row.get("Lời Bình Dẫn Chuyện (Có thể sửa)", "")),
            "camera_motion": str(row.get("Camera Motion", "zoom_in")),
            "sfx_cue": str(row.get("Âm Thanh SFX", "none"))
        })

    timeline_data = {
        "title": video_title,
        "intro_hook": intro_hook,
        "total_scenes": len(scenes),
        "scenes": scenes
    }

    ref_audio_path = ref_audio.name if ref_audio is not None else None
    tts = TTSEngine(
        voice_preset=voice_preset,
        reference_audio=ref_audio_path,
        output_dir=str(audio_dir)
    )
    sub_gen = SubtitleGenerator(output_dir=str(sub_dir), aspect_ratio=aspect_ratio)
    video_engine = VideoEngine(
        output_dir=str(scenes_dir),
        final_output_path=str(final_video_path),
        aspect_ratio=aspect_ratio
    )

    rendered_scenes = []
    for idx, sc in enumerate(scenes, start=1):
        panel_file_path = str(panels_dir / sc["panel_file"])
        if not os.path.exists(panel_file_path):
            continue

        progress(0.2 + 0.7 * (idx / max(1, len(scenes))), desc=f"Đang dựng Scene {idx}/{len(scenes)}...")
        audio_info = tts.synthesize_scene(scene_id=idx, text=sc["narration"])
        sub_info = sub_gen.transcribe_and_generate_sub(
            scene_id=idx,
            audio_path=audio_info["audio_path"],
            reference_text=sc["narration"]
        )
        scene_mp4 = video_engine.render_scene(
            scene_id=idx,
            panel_path=panel_file_path,
            audio_path=audio_info["audio_path"],
            sub_ass_path=sub_info["ass_path"],
            camera_motion=sc.get("camera_motion", "zoom_in"),
            duration=audio_info["duration"]
        )
        rendered_scenes.append(scene_mp4)

    bgm_path = BASE_DIR / "assets" / "bgm" / "eerie_ambient.mp3"
    final_output = video_engine.concatenate_scenes(
        scene_video_paths=rendered_scenes,
        bgm_path=str(bgm_path) if bgm_path.exists() else None
    )

    capcut_zip_path = None
    if export_capcut:
        capcut_exp = CapCutExporter(output_dir=str(draft_dir))
        draft_res = capcut_exp.export_draft(
            timeline_data=timeline_data,
            panels_dir=str(panels_dir),
            audio_dir=str(audio_dir),
            aspect_ratio=aspect_ratio
        )
        capcut_zip_path = str(storage_dir / f"CapCut_Custom_{aspect_ratio.replace(':', '_')}.zip")
        with zipfile.ZipFile(capcut_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(draft_res):
                for file in files:
                    file_p = Path(root) / file
                    zipf.write(file_p, arcname=file_p.relative_to(Path(draft_res).parent))

    return str(final_output), capcut_zip_path, f"🎉 Đã dựng xong video hoàn chỉnh từ kịch bản bạn chỉnh sửa ({aspect_ratio})!"


# --- GRADIO INTERFACE LAYOUT ---
def create_ui():
    custom_theme = gr.themes.Soft(
        primary_hue="red",
        secondary_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "sans-serif"]
    )

    with gr.Blocks(title="Manga Recap AI Studio - Phase 2") as demo:
        gr.Markdown(
            """
            # 🎬 Manga Recap AI Studio
            ### Xưởng Dựng Video Tóm Tắt Truyện Tranh Bằng AI (Phong Cách *Quán Khuya*)
            *Tối ưu cho Google Colab GPU T4 | Hỗ trợ Video Ngang YouTube 16:9 & Dọc TikTok/Shorts 9:16*
            """
        )

        with gr.Tabs():
            # --- TAB 1: TẠO NHANH ---
            with gr.TabItem("⚡ 1-Click Auto Recap (Tự Động Trọn Gói)"):
                with gr.Row():
                    with gr.Column(scale=5):
                        with gr.Group():
                            gr.Markdown("### 📥 1. Nguồn Truyện Tranh")
                            auto_url = gr.Textbox(
                                label="Link Chương Truyện Online",
                                placeholder="https://mangadex.org/chapter/... hoặc trang truyện bất kỳ",
                                lines=1
                            )
                            auto_file = gr.File(
                                label="HOẶC Tải lên file .zip / .cbz / .pdf từ máy",
                                file_types=[".zip", ".cbz", ".pdf"]
                            )

                        with gr.Group():
                            gr.Markdown("### 🧠 2. Cấu Hình Đạo Diễn AI")
                            with gr.Row():
                                auto_model = gr.Dropdown(
                                    label="Mô hình AI Đạo Diễn (Qwen2.5-VL / Gemini)",
                                    choices=[
                                        "Qwen/Qwen2.5-VL-7B-Instruct (Chạy Offline 4-bit GPU T4 - Khuyên Dùng)",
                                        "Qwen/Qwen2.5-VL-3B-Instruct (Chạy Offline Siêu Tốc & Nhẹ Máy)",
                                        "gemini-3.8-flash (Google AI Studio Free API)",
                                        "gemini-3.5-flash-lite (Google AI Studio API)",
                                        "gemini-2.5-flash (Google AI Studio API)"
                                    ],
                                    value="Qwen/Qwen2.5-VL-7B-Instruct (Chạy Offline 4-bit GPU T4 - Khuyên Dùng)"
                                )
                                auto_aspect = gr.Dropdown(
                                    label="Định Dạng Video",
                                    choices=["16:9 (YouTube Ngang - 1920x1080)", "9:16 (Shorts/TikTok Dọc - 1080x1920)"],
                                    value="16:9 (YouTube Ngang - 1920x1080)"
                                )

                            auto_gemini_key = gr.Textbox(
                                label="Gemini API Key (Chỉ cần khi chọn Gemini, bỏ trống nếu dùng Qwen2.5-VL)",
                                placeholder="Để trống nếu đã cài trong môi trường",
                                type="password"
                            )
                            auto_synopsis = gr.Textbox(
                                label="Gợi ý cốt truyện / tông giọng",
                                value="Một vụ án kinh hoàng lúc nửa đêm, không khí căng thẳng, u ám, giọng kể rùng rợn và kích thích tò mò.",
                                lines=2
                            )
                            with gr.Row():
                                auto_drama = gr.Slider(minimum=5, maximum=10, value=7, step=1, label="Điểm kịch tính tối thiểu")
                                auto_max_p = gr.Slider(minimum=10, maximum=60, value=35, step=5, label="Số khung tranh tối đa")

                        with gr.Group():
                            gr.Markdown("### 🎙️ 3. Giọng Kể Chuyện & Hiệu Ứng")
                            with gr.Row():
                                auto_voice = gr.Dropdown(
                                    label="Giọng Đọc VieNeu-TTS (48kHz)",
                                    choices=["Thiện Minh (Trầm ấm, rùng rợn - Chuẩn Quán Khuya)", "Minh Đức (Miền Bắc, rõ ràng, kịch tính)", "Hải Đăng (Nam tính, bí ẩn)", "Quang Sơn (Mạnh mẽ, dồn dập)", "Thùy Dung (Giọng Nữ u uất, truyền cảm)", "Mai Anh (Giọng Nữ kịch tính)", "Quốc Tuấn (Chững chạc)", "Minh Triết (Trầm tĩnh)"],
                                    value="Thiện Minh (Trầm ấm, rùng rợn - Chuẩn Quán Khuya)"
                                )
                                auto_camera = gr.Dropdown(
                                    label="Hiệu Ứng Chuyển Động Camera",
                                    choices=["auto", "zoom_in", "zoom_out", "pan_left"],
                                    value="auto"
                                )
                            with gr.Row():
                                btn_preview_voice = gr.Button("🔊 Nghe Thử Voice Mẫu", size="sm")
                                voice_preview_player = gr.Audio(label="Nghe thử giọng", interactive=False)

                            auto_ref_audio = gr.Audio(label="Clone giọng (mẫu audio 3-5s - Tùy chọn)", type="filepath")

                        with gr.Group():
                            gr.Markdown("### ⚙️ 4. Xuất Bản & Lưu Trữ")
                            with gr.Row():
                                auto_drive = gr.Checkbox(label="Lưu vào Google Drive", value=True)
                                auto_capcut = gr.Checkbox(label="Xuất file CapCut PC (.zip)", value=True)

                        btn_run_auto = gr.Button("🚀 BẮT ĐẦU TẠO VIDEO (1-CLICK)", variant="primary", size="lg")

                    with gr.Column(scale=5):
                        gr.Markdown("### 📺 Video Thành Phẩm")
                        auto_video_out = gr.Video(label="Video Recap Hoàn Chỉnh")
                        with gr.Row():
                            auto_audio_out = gr.Audio(label="Voice Scene 1", type="filepath")
                            auto_capcut_out = gr.File(label="Tải Gói Dự Án CapCut (.zip)")
                        auto_logs = gr.Textbox(label="Nhật Ký Tiến Trình", lines=12, interactive=False)

                # Event bindings for Tab 1
                def _wrap_aspect(val):
                    return "9:16" if "9:16" in val else "16:9"

                btn_preview_voice.click(
                    fn=preview_voice_sample,
                    inputs=[auto_voice, auto_ref_audio],
                    outputs=[voice_preview_player]
                )
                btn_run_auto.click(
                    fn=lambda u, f, k, m, asp, s, v, r, d, mp, cam, cap, drv: process_auto_pipeline(
                        u, f, k, m, _wrap_aspect(asp), s, v, r, d, mp, cam, cap, drv
                    ),
                    inputs=[
                        auto_url, auto_file, auto_gemini_key, auto_model, auto_aspect,
                        auto_synopsis, auto_voice, auto_ref_audio, auto_drama, auto_max_p,
                        auto_camera, auto_capcut, auto_drive
                    ],
                    outputs=[auto_video_out, auto_audio_out, auto_capcut_out, auto_logs]
                )

            # --- TAB 2: ĐẠO DIỄN TƯƠNG TÁC (STUDIO DIRECTOR) ---
            with gr.TabItem("🎬 Studio Director Mode (Xem Panels & Sửa Kịch Bản)"):
                gr.Markdown("#### Quy trình 3 bước chuyên nghiệp: Trích xuất tranh ➔ AI đề xuất kịch bản & Bạn chỉnh sửa ➔ Dựng video")
                
                # Hidden state storing panels JSON
                stored_panels_state = gr.State("")

                with gr.Accordion("📌 Bước 1: Nạp Truyện Tranh & Xem Thư Viện Panels (RTL)", open=True):
                    with gr.Row():
                        dir_url = gr.Textbox(label="Link Truyện Online", placeholder="URL chương truyện...", scale=3)
                        dir_file = gr.File(label="Tải file .zip / .cbz / .pdf", file_types=[".zip", ".cbz", ".pdf"], scale=2)
                    btn_step1 = gr.Button("✂️ BÓC TÁCH KHUNG TRANH (YOLO Manga109)", variant="secondary")
                    step1_status = gr.Markdown("")
                    panels_gallery = gr.Gallery(label="Bộ sưu tập khung tranh đã bóc tách (Thứ tự đọc Manga RTL)", columns=6, height="auto")

                with gr.Accordion("📝 Bước 2: AI Đạo Diễn & Chỉnh Sửa Kịch Bản Trực Tiếp", open=True):
                    with gr.Row():
                        dir_model = gr.Dropdown(
                            label="Mô hình AI Đạo Diễn (Qwen2.5-VL / Gemini)",
                            choices=[
                                "Qwen/Qwen2.5-VL-7B-Instruct (Chạy Offline 4-bit GPU T4 - Khuyên Dùng)",
                                "Qwen/Qwen2.5-VL-3B-Instruct (Chạy Offline Siêu Tốc & Nhẹ Máy)",
                                "gemini-3.8-flash (Google AI Studio Free API)",
                                "gemini-3.5-flash-lite (Google AI Studio API)",
                                "gemini-2.5-flash (Google AI Studio API)"
                            ],
                            value="Qwen/Qwen2.5-VL-7B-Instruct (Chạy Offline 4-bit GPU T4 - Khuyên Dùng)"
                        )
                        dir_key = gr.Textbox(label="Gemini API Key (Chỉ cần khi chọn Gemini)", placeholder="Để trống nếu có env key", type="password")
                        dir_drama = gr.Slider(minimum=5, maximum=10, value=7, step=1, label="Điểm kịch tính tối thiểu")
                        dir_max_p = gr.Slider(minimum=10, maximum=50, value=25, step=5, label="Số cảnh tối đa")

                    dir_synopsis = gr.Textbox(
                        label="Chỉ dẫn cốt truyện & phong cách kịch bản",
                        value="Giọng kể Quán Khuya rùng rợn, nhấn mạnh sự bí ẩn, từng khung tranh như một câu đố chết người.",
                        lines=2
                    )
                    btn_step2 = gr.Button("🧠 TẠO KỊCH BẢN ĐỀ XUẤT", variant="secondary")
                    step2_status = gr.Markdown("")

                    with gr.Row():
                        script_title = gr.Textbox(label="Tiêu Đề Video (YouTube Title)", scale=3)
                        script_intro = gr.Textbox(label="Intro Hook Mở Đầu", scale=3)

                    gr.Markdown("##### ✏️ Bảng Kịch Bản (Bạn có thể nhấn đúp vào ô để sửa lời bình, đổi camera motion hoặc SFX):")
                    script_table = gr.Dataframe(
                        headers=["Khung Tranh", "Điểm Kịch Tính", "Lời Bình Dẫn Chuyện (Có thể sửa)", "Camera Motion", "Âm Thanh SFX"],
                        datatype=["str", "number", "str", "str", "str"],
                        interactive=True,
                        wrap=True
                    )

                with gr.Accordion("🎥 Bước 3: Dựng Video & Xuất CapCut Từ Kịch Bản Đã Sửa", open=True):
                    with gr.Row():
                        dir_aspect = gr.Dropdown(
                            label="Định Dạng Video",
                            choices=["16:9 (YouTube Ngang - 1920x1080)", "9:16 (Shorts/TikTok Dọc - 1080x1920)"],
                            value="16:9 (YouTube Ngang - 1920x1080)"
                        )
                        dir_voice = gr.Dropdown(
                            label="Giọng Đọc Kể Chuyện",
                            choices=["Thiện Minh (Trầm ấm, rùng rợn - Chuẩn Quán Khuya)", "Minh Đức (Miền Bắc, rõ ràng, kịch tính)", "Hải Đăng (Nam tính, bí ẩn)", "Quang Sơn (Mạnh mẽ, dồn dập)", "Thùy Dung (Giọng Nữ u uất, truyền cảm)", "Mai Anh (Giọng Nữ kịch tính)", "Quốc Tuấn (Chững chạc)", "Minh Triết (Trầm tĩnh)"],
                            value="Thiện Minh (Trầm ấm, rùng rợn - Chuẩn Quán Khuya)"
                        )
                        dir_capcut = gr.Checkbox(label="Xuất CapCut PC Project (.zip)", value=True)

                    btn_step3 = gr.Button("🎬 BẮT ĐẦU DỰNG VIDEO THÀNH PHẨM", variant="primary", size="lg")
                    step3_status = gr.Markdown("")

                    with gr.Row():
                        dir_video_out = gr.Video(label="Video Thành Phẩm Hoàn Chỉnh")
                        dir_capcut_out = gr.File(label="Tải Dự Án CapCut PC (.zip)")

                # Event bindings for Tab 2
                btn_step1.click(
                    fn=step1_interactive_extract,
                    inputs=[dir_url, dir_file],
                    outputs=[panels_gallery, step1_status, stored_panels_state]
                )

                btn_step2.click(
                    fn=step2_interactive_generate_script,
                    inputs=[stored_panels_state, dir_key, dir_model, dir_synopsis, dir_drama, dir_max_p],
                    outputs=[script_title, script_intro, script_table, step2_status]
                )

                btn_step3.click(
                    fn=lambda df, t, i, asp, v, cap: step3_interactive_render(
                        df, t, i, _wrap_aspect(asp), v, None, cap
                    ),
                    inputs=[script_table, script_title, script_intro, dir_aspect, dir_voice, dir_capcut],
                    outputs=[dir_video_out, dir_capcut_out, step3_status]
                )

        gr.Markdown(
            """
            ---
            *Manga Recap AI Studio - Phiên bản 2.0 (Hỗ trợ Đạo Diễn Tương Tác & Đa Tỉ Lệ 16:9 / 9:16)*
            """
        )

    return demo


if __name__ == "__main__":
    demo = create_ui()
    allowed_dirs = [
        str(BASE_DIR),
        str(BASE_DIR / "workspace"),
        "/content/workspace",
        "/content/drive/MyDrive/MangaRecap",
        "/tmp"
    ]
    demo.launch(
        allowed_paths=allowed_dirs,
        share=True,
        debug=True
    )
