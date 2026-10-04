"""
Manga Recap AI Studio - Gradio Web UI
Giao diện trực quan chạy trực tiếp trên Google Colab GPU T4.
Hỗ trợ xem trước Panels, nghe thử giọng đọc VieNeu-TTS, chỉnh sửa kịch bản và xem video thành phẩm.
"""

import os
import sys
import yaml
import shutil
import zipfile
from pathlib import Path
import gradio as gr

# Add project root to sys.path
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
    """
    Xác định thư mục lưu trữ:
    - Nếu chọn Google Drive và Drive đã được mount: lưu vào /content/drive/MyDrive/MangaRecap
    - Nếu không mount Drive hoặc tắt tùy chọn: lưu cục bộ vào /content/workspace (hoặc ./workspace)
    """
    drive_path = Path("/content/drive/MyDrive/MangaRecap")
    if prefer_drive and Path("/content/drive/MyDrive").exists():
        drive_path.mkdir(parents=True, exist_ok=True)
        return drive_path
    
    # Fallback to local /content/workspace
    local_path = Path("/content/workspace") if Path("/content").exists() else BASE_DIR / "workspace"
    local_path.mkdir(parents=True, exist_ok=True)
    return local_path


def process_pipeline(
    manga_url: str,
    local_file,
    gemini_key: str,
    model_choice: str,
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
    final_video_path = storage_dir / "final_recap_video.mp4"
    draft_dir = storage_dir / "capcut_draft"

    for d in [raw_dir, panels_dir, audio_dir, sub_dir, scenes_dir]:
        d.mkdir(parents=True, exist_ok=True)

    status_log = []
    def log(msg):
        print(msg)
        status_log.append(msg)
        return "\n".join(status_log)

    # --- BƯỚC 1: TẢI TRUYỆN ---
    progress(0.1, desc="Đang nạp truyện tranh...")
    log(f"📁 Thư mục lưu trữ: {storage_dir}")
    downloader = MangaDownloader(output_dir=str(raw_dir))

    if manga_url and manga_url.strip().startswith("http"):
        log(f"📥 Đang cào ảnh từ URL: {manga_url}")
        pages = downloader.download_from_url(manga_url.strip())
    elif local_file is not None:
        log(f"📦 Đang giải nén/trích xuất file tải lên: {local_file.name}")
        pages = downloader.load_from_local_archive(local_file.name)
    else:
        return None, None, None, log("❌ Lỗi: Vui lòng nhập link truyện tranh hoặc tải lên file .zip/.cbz/.pdf!")

    if not pages:
        return None, None, None, log("❌ Không tìm thấy trang truyện nào!")

    log(f"✅ Đã chuẩn bị xong {len(pages)} trang truyện gốc.")

    # --- BƯỚC 2: CẮT PANEL & SẮP XẾP RTL ---
    progress(0.25, desc="Đang cắt ô tranh bằng AI...")
    log("✂️ Đang nhận diện ô tranh (YOLO Manga109) & sắp xếp thứ tự đọc Manga (RTL)...")
    extractor = MangaPanelExtractor(output_dir=str(panels_dir), reading_order="RTL")
    panels_meta = extractor.process_all_pages(pages)
    log(f"✅ Đã cắt thành công {len(panels_meta)} panels ô tranh.")

    # --- BƯỚC 3: GEMINI CHỌN CẢNH & VIẾT KỊCH BẢN ---
    clean_model_name = model_choice.split(" ")[0].strip() if model_choice else "gemini-3.8-flash"
    progress(0.45, desc=f"Đạo diễn AI ({clean_model_name}) đang viết kịch bản...")
    log(f"🧠 Đang gửi hình ảnh qua AI ({clean_model_name}) để chấm điểm kịch tính và chọn cảnh đắt giá...")
    
    api_key_to_use = gemini_key.strip() if gemini_key else os.environ.get("GEMINI_API_KEY", "")
    timeline_file = storage_dir / "timeline.json"

    script_gen = ScriptGenerator(
        provider="gemini" if api_key_to_use else "qwen_vl",
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
    log(f"🎬 Đạo diễn AI đã chọn ra {len(selected_scenes)} cảnh cao trào cho video: '{timeline.get('title', '')}'")

    # --- BƯỚC 4: GIỌNG ĐỌC VIENEU-TTS & PHỤ ĐỀ WHISPER ---
    progress(0.65, desc="VieNeu-TTS đang đọc & Faster-Whisper tạo sub...")
    ref_audio_path = ref_audio.name if ref_audio is not None else None
    tts = TTSEngine(
        voice_preset=voice_preset,
        reference_audio=ref_audio_path,
        output_dir=str(audio_dir)
    )
    sub_gen = SubtitleGenerator(output_dir=str(sub_dir))
    video_engine = VideoEngine(
        output_dir=str(scenes_dir),
        final_output_path=str(final_video_path)
    )

    rendered_scenes = []
    for idx, sc in enumerate(selected_scenes, start=1):
        panel_file_path = str(panels_dir / sc["panel_file"])
        if not os.path.exists(panel_file_path):
            continue

        progress(0.65 + 0.25 * (idx / max(1, len(selected_scenes))), desc=f"Xử lý Scene {idx}/{len(selected_scenes)}...")
        
        # TTS Audio
        audio_info = tts.synthesize_scene(scene_id=idx, text=sc["narration"])

        # Word-level Subtitle
        sub_info = sub_gen.transcribe_and_generate_sub(
            scene_id=idx,
            audio_path=audio_info["audio_path"],
            reference_text=sc["narration"]
        )

        # Render Scene Clip with Ken Burns & NVENC
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

    # --- BƯỚC 5: GHÉP VIDEO HOÀN CHỈNH ---
    progress(0.92, desc="Đang hòa âm & xuất video hoàn chỉnh...")
    bgm_path = BASE_DIR / "assets" / "bgm" / "eerie_ambient.mp3"
    final_output = video_engine.concatenate_scenes(
        scene_video_paths=rendered_scenes,
        bgm_path=str(bgm_path) if bgm_path.exists() else None
    )
    log(f"🎉 Video hoàn tất: {final_output}")

    # CapCut Export (nếu chọn)
    capcut_zip_path = None
    if export_capcut_check:
        log("✂️ Đang đóng gói dự án CapCut Draft...")
        capcut_exp = CapCutExporter(output_dir=str(draft_dir))
        draft_res = capcut_exp.export_draft(
            timeline_data=timeline,
            panels_dir=str(panels_dir),
            audio_dir=str(audio_dir)
        )
        # Nén thư mục draft thành file zip để người dùng tải về dễ dàng
        capcut_zip_path = str(storage_dir / "CapCut_Draft_Project.zip")
        with zipfile.ZipFile(capcut_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(draft_res):
                for file in files:
                    file_p = Path(root) / file
                    zipf.write(file_p, arcname=file_p.relative_to(Path(draft_res).parent))
        log(f"📦 Đã xuất file CapCut zip: {capcut_zip_path}")

    first_audio = str(audio_dir / "scene_0001.wav") if (audio_dir / "scene_0001.wav").exists() else None
    return str(final_output), first_audio, capcut_zip_path, "\n".join(status_log)


def create_ui():
    custom_theme = gr.themes.Soft(
        primary_hue="red",
        secondary_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "sans-serif"]
    )

    with gr.Blocks(theme=custom_theme, title="Manga Recap AI Studio") as demo:
        gr.Markdown(
            """
            # 🎬 Manga Recap AI Studio
            ### Hệ thống dựng Video Tóm tắt / Review Truyện tranh Kịch tính (Phong cách *Quán Khuya*)
            *Tối ưu hóa chạy trơn tru trên **Google Colab GPU T4 (16GB VRAM)** | Chi phí **0 ĐỒNG***
            """
        )

        with gr.Row():
            with gr.Column(scale=5):
                with gr.Group():
                    gr.Markdown("### 📥 1. Đầu Vào Truyện Tranh")
                    input_url = gr.Textbox(
                        label="Link Web Truyện Online",
                        placeholder="https://mangadex.org/chapter/... hoặc link web truyện bất kỳ",
                        lines=1
                    )
                    local_archive = gr.File(
                        label="HOẶC Tải lên file .zip / .cbz / .pdf từ máy tính",
                        file_types=[".zip", ".cbz", ".pdf"]
                    )

                with gr.Group():
                    gr.Markdown("### 🧠 2. Đạo Diễn AI (Google Gemini)")
                    with gr.Row():
                        model_choice = gr.Dropdown(
                            label="Phiên bản Gemini Model",
                            choices=[
                                "gemini-3.8-flash (Tối ưu nhất - Gemini 3)",
                                "gemini-3.5-flash-lite (Siêu tốc & Tiết kiệm)",
                                "gemini-2.5-flash (Thế hệ trước)"
                            ],
                            value="gemini-3.8-flash (Tối ưu nhất - Gemini 3)"
                        )
                    gemini_api_key = gr.Textbox(
                        label="Google Gemini API Key (Miễn phí 100% tại aistudio.google.com)",
                        placeholder="Để trống nếu đã cài trong Colab Secrets hoặc dùng Offline Qwen-VL",
                        type="password"
                    )
                    synopsis_input = gr.Textbox(
                        label="Gợi ý Cốt truyện / Phong cách dẫn dắt",
                        value="Một vụ án kinh hoàng lúc nửa đêm, không khí căng thẳng, u ám, giọng kể rùng rợn và kích thích tò mò.",
                        lines=2
                    )
                    with gr.Row():
                        drama_slider = gr.Slider(
                            minimum=5, maximum=10, value=7, step=1,
                            label="Độ kịch tính tối thiểu (Chỉ chọn cảnh điểm >= giá trị này)"
                        )
                        max_panels_slider = gr.Slider(
                            minimum=10, maximum=60, value=35, step=5,
                            label="Số khung tranh tối đa trong 1 video"
                        )

                with gr.Group():
                    gr.Markdown("### 🎙️ 3. Giọng Đọc VieNeu-TTS-v3-Turbo (48kHz)")
                    with gr.Row():
                        voice_choice = gr.Dropdown(
                            label="Giọng đọc kể chuyện",
                            choices=["NamMinh (Trầm ấm - Kịch tính)", "BacMinh (Rõ ràng - Truyền cảm)", "TrungNam (Bí ẩn)"],
                            value="NamMinh (Trầm ấm - Kịch tính)"
                        )
                        camera_motion = gr.Dropdown(
                            label="Hiệu ứng Camera (Ken Burns)",
                            choices=["auto", "zoom_in", "zoom_out", "pan_left"],
                            value="auto"
                        )
                    ref_audio_input = gr.Audio(
                        label="Nhái giọng (Instant Voice Cloning) - Tải lên mẫu audio 3-5 giây (Tùy chọn)",
                        type="filepath"
                    )

                with gr.Group():
                    gr.Markdown("### ⚙️ 4. Tùy Chọn Xuất Video & Lưu Trữ")
                    with gr.Row():
                        save_drive = gr.Checkbox(
                            label="Lưu vào Google Drive (nếu đã mount)",
                            value=True,
                            info="Nếu tắt hoặc chưa mount Drive, video sẽ lưu an toàn trong thư mục /content cục bộ"
                        )
                        export_capcut = gr.Checkbox(
                            label="Xuất gói dự án CapCut PC (.zip)",
                            value=True,
                            info="Mở file trên CapCut PC để chỉnh sửa chuyển cảnh hoặc thêm sticker"
                        )

                btn_generate = gr.Button("🚀 BẮT ĐẦU TẠO VIDEO RECAP", variant="primary", size="lg")

            with gr.Column(scale=5):
                gr.Markdown("### 📺 Kết Quả Thành Phẩm")
                video_output = gr.Video(label="Video Recap Hoàn Chỉnh (1080p)")
                
                with gr.Row():
                    audio_preview = gr.Audio(label="Nghe thử Voice AI mẫu (Scene 1)", type="filepath")
                    capcut_download = gr.File(label="Tải về File Dự Án CapCut (.zip)")

                gr.Markdown("### 📜 Tiến Trình Thực Thi")
                log_box = gr.Textbox(
                    label="Nhật ký hệ thống (Console Logs)",
                    lines=14,
                    interactive=False
                )

        btn_generate.click(
            fn=process_pipeline,
            inputs=[
                input_url,
                local_archive,
                gemini_api_key,
                model_choice,
                synopsis_input,
                voice_choice,
                ref_audio_input,
                drama_slider,
                max_panels_slider,
                camera_motion,
                export_capcut,
                save_drive
            ],
            outputs=[video_output, audio_preview, capcut_download, log_box]
        )

        gr.Markdown(
            """
            ---
            *Manga Recap AI Studio - Thiết kế cho Google Colab GPU T4.*
            """
        )

    return demo


if __name__ == "__main__":
    demo = create_ui()
    # launch with share=True on Colab to provide a public URL
    demo.launch(share=True, debug=True)
