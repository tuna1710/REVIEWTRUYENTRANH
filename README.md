# 🎬 Manga Recap AI Studio
> **Tự động hóa sản xuất video tóm tắt / review truyện tranh kịch tính, phong cách kinh dị (Quán Khuya Style)**  
> Tối ưu hóa vận hành trên **Google Colab (GPU NVIDIA Tesla T4 - 16GB VRAM)**, chi phí **0 ĐỒNG** với Google AI Studio Free Tier & Open-Source AI.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-username/manga-recap-ai/blob/main/notebooks/Manga_Recap_Colab.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🌟 Tính Năng Nổi Bật

- 📥 **Tự động tải truyện:** Hỗ trợ link đọc truyện online từ 1000+ website qua `gallery-dl`, hoặc nạp trực tiếp file `.zip`, `.cbz`, `.pdf`.
- ✂️ **Cắt ô tranh thông minh (Panel Extraction):** Mô hình YOLO chuyên biệt nhận diện chính xác từng khung tranh, tự động sắp xếp theo **thứ tự đọc Manga Nhật Bản (Right-to-Left, Top-to-Bottom)**.
- 🧠 **Đạo diễn AI chọn cảnh đắt giá (AI Climax Filter):** Sử dụng **Gemini 3.8 Flash (Free Tier)** để phân tích thị giác, loại bỏ 70% khung hình tĩnh thừa thãi, chỉ giữ lại 30–40 khung tranh kịch tính nhất và viết lời bình tiếng Việt rùng rợn, lôi cuốn.
- 🎙️ **Giọng đọc kể chuyện 48kHz chất lượng cao:** Tích hợp mô hình TTS tiếng Việt tiên tiến nhất **`VieNeu-TTS-v3-Turbo`**, hỗ trợ tag cảm xúc `[thở dài]`, `[hắng giọng]` và **Instant Voice Cloning (nhái giọng)** chỉ với 3–5 giây audio mẫu.
- 💬 **Phụ đề nhảy chữ Karaoke:** Bóc tách timestamp từng từ siêu chuẩn xác với **`Faster-Whisper`** trên CUDA float16, xuất file phụ đề động `.ass` chuẩn phong cách YouTube Shorts / Recap hiện đại.
- 🎞️ **Engine dựng Video điện ảnh:** Hiệu ứng **Ken Burns Zoom/Pan chậm**, nền mờ xóa bỏ viền đen 16:9, lồng nhạc nền rùng rợn (BGM) với Audio Ducking, tăng tốc render bằng **NVIDIA NVENC** trên GPU T4.
- ✂️ **Xuất Project CapCut / JianYing (`pyJianYingDraft`):** Tự động xuất toàn bộ kịch bản, panel, audio và sub thành một file dự án CapCut PC để bạn có thể mở lên chỉnh sửa thủ công nếu muốn.

---

## 🏗️ Kiến Trúc Hệ Thống (Pipeline Architecture)

```
[Link truyện online / Local Zip]
              │
              ▼ (gallery-dl / custom downloader)
[Ảnh gốc các trang Manga]
              │
              ▼ (YOLO26n / Manga Panel Detector trên GPU T4)
[Tách Panels & Sắp xếp thứ tự đọc Manga RTL]
              │
              ▼ (Gemini 3.8 Flash Vision Free API / Qwen2.5-VL 4bit)
[Lọc 30-40 cảnh cao trào + Xuất kịch bản timeline.json]
              │
      ┌───────┴───────────────────────┐
      ▼                               ▼
[VieNeu-TTS-v3-Turbo]        [Faster-Whisper CUDA]
(Giọng kể chuyện 48kHz)     (Bóc tách Timestamp từng chữ)
      └───────┬───────────────────────┘
              │
              ▼ (FFmpeg NVENC / pyJianYingDraft)
[Xuất Video MP4 1080p + Subtitle Nhảy Chữ HOẶC Dự Án CapCut]
```

---

## 🚀 Hướng Dẫn Sử Dụng Trên Google Colab

1. Nhấp vào nút **Open In Colab** ở đầu trang hoặc mở file `notebooks/Manga_Recap_Colab.ipynb`.
2. Chọn Runtime: **Runtime > Change runtime type > T4 GPU**.
3. Điền các thông số trong Form:
   - `MANGA_SOURCE`: Link chapter truyện cần tóm tắt.
   - `GEMINI_API_KEY`: API Key lấy miễn phí 100% tại [Google AI Studio](https://aistudio.google.com).
   - `VOICE_PRESET`: Chọn giọng đọc (`NamMinh`, `BacMinh`, v.v.).
4. Nhấn **Run All** (Chạy tất cả) và thư giãn. Video thành phẩm sẽ được lưu tự động vào Google Drive của bạn!

---

## 💻 Cài Đặt & Chạy Cục Bộ (Local Machine)

### 1. Yêu cầu hệ thống
- Python 3.10+
- FFmpeg đã được cài đặt vào PATH
- Card đồ họa NVIDIA (khuyên dùng từ 6GB VRAM trở lên)

### 2. Cài đặt thư viện
```bash
git clone https://github.com/your-username/manga-recap-ai.git
cd manga-recap-ai
pip install -r requirements.txt
```

### 3. Thực thi dòng lệnh (CLI)
```bash
python main.py \
  --input "https://link-truyen-chapter-1.html" \
  --api-key "AIzaSy..." \
  --synopsis "Một ngôi làng bí ẩn với lời nguyền kinh dị..." \
  --voice "NamMinh" \
  --capcut
```

---

## 📁 Cấu Trúc Thư Mục Dự Án

```text
manga-recap-ai/
├── README.md                      # Tài liệu dự án
├── requirements.txt               # Danh sách thư viện phụ thuộc
├── config.yaml                    # File cấu hình trung tâm
├── main.py                        # Script điều phối toàn bộ pipeline
├── notebooks/
│   └── Manga_Recap_Colab.ipynb    # Jupyter Notebook chạy 1-click trên Google Colab
├── src/
│   ├── __init__.py
│   ├── downloader.py              # Module tải truyện từ web hoặc giải nén
│   ├── panel_extractor.py         # Module tách panel tranh & sắp xếp RTL
│   ├── script_generator.py        # Module AI chọn cảnh & viết kịch bản JSON
│   ├── tts_engine.py              # Module giọng đọc VieNeu-TTS-v3-Turbo
│   ├── subtitle_generator.py      # Module tạo sub từng từ Faster-Whisper
│   ├── video_engine.py            # Module render video Ken Burns & NVENC
│   └── capcut_exporter.py         # Module xuất project CapCut Draft
└── assets/
    ├── bgm/                       # Nhạc nền không bản quyền
    └── sfx/                       # Hiệu ứng âm thanh
```

---

## 📜 Giấy Phép & Tuyên Bố Miễn Trừ Trách Nhiệm
- Dự án được phát hành theo giấy phép **MIT License**.
- Dự án phục vụ mục đích nghiên cứu, học tập và hỗ trợ sáng tạo nội dung hợp pháp. Người sử dụng cần tự chịu trách nhiệm về bản quyền tác phẩm truyện tranh khi xuất bản nội dung công khai.
