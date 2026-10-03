# 📘 Kiến Trúc & Kế Hoạch Dự Án: Manga Recap AI Studio
> **Định hướng:** Tạo video tóm tắt / review truyện tranh tự động phong cách kịch tính & kinh dị (như kênh *Quán Khuya*).  
> **Môi trường vận hành:** Google Colab (GPU NVIDIA T4 - 16GB VRAM).  
> **Mục tiêu:** Cấu trúc dự án chuẩn để đẩy lên GitHub cá nhân, tối ưu hóa chi phí (0 ĐỒNG), thời gian và chất lượng biên tập.

---

## I. Phân Tích Tài Nguyên & Ràng Buộc (Google Colab GPU T4)

| Thành phần | Thông số Colab T4 | Mức tiêu thụ thực tế | Đánh giá khả thi |
| :--- | :--- | :--- | :--- |
| **VRAM GPU** | 15 - 16 GB | ~3.5 - 5.5 GB (VieNeu + Whisper + YOLO) | **Rất an toàn**, không lo tràn VRAM (OOM) |
| **RAM Hệ Thống** | ~12.7 GB | ~4 - 6 GB khi xử lý batch ảnh | Cần giải phóng bộ nhớ định kỳ |
| **Ổ đĩa (Disk)** | ~70 - 100 GB | ~5 - 10 GB cho 1 bộ truyện hoàn chỉnh | Đủ dung lượng, kết nối Google Drive để lưu |
| **GPU Encoder** | Hỗ trợ NVENC | Dùng `h264_nvenc` trong FFmpeg | Render video 1080p cực nhanh (x3-x5 lần CPU) |

---

## II. Phương Án Thiết Kế Hệ Thống Tối Ưu (Pipeline Blueprint)

```
[1. Input: Link Web Manga / Local Zip]
                │
                ▼ (gallery-dl / downloader)
[2. Download & Sắp xếp ảnh gốc Chapter]
                │
                ▼ (YOLO26n / Manga Panel Detector trên GPU T4)
[3. Tách Panels & Đánh số theo thứ tự đọc RTL]
                │
                ▼ (LỰA CHỌN 1: Gemini 3.8 Flash Free API  HOẶC  LỰA CHỌN 2: Qwen2.5-VL-7B 4bit Open-Source)
[4. Chọn lọc Khung Cảnh Đắt Giá (Climax) + Viết Kịch Bản JSON]
                │
        ┌───────┴────────────────────────┐
        ▼                                ▼
[5. VieNeu-TTS-v3-Turbo]        [6. Faster-Whisper (CUDA)]
(Sinh Voice 48kHz kể chuyện)     (Bóc tách Timestamp từng chữ)
        └───────┬────────────────────────┘
                │
                ▼ (FFmpeg NVENC / pyJianYingDraft)
[7. Ghép Video 1080p + Subtitle Nhảy Chữ + SFX + BGM]
                │
                ▼
[8. Thành phẩm MP4 lưu thẳng vào Google Drive]
```

---

## III. Chi Tiết Các Module Thành Phần

### Module 1: Input & Manga Ingestion (Trình tải truyện)
- **Công cụ:** `gallery-dl` kết hợp `cloudscraper`.
- **Chức năng:**
  - Nhập link chapter từ các website truyện tranh (MangaDex, Blogtruyen, Nettruyen...).
  - Tự động bypass bảo vệ chống bot nhẹ, tải toàn bộ ảnh và lưu theo cấu trúc thư mục chuẩn: `raw_pages/page_001.jpg`, `page_002.jpg`...
  - Hỗ trợ fallback: Nhập trực tiếp file `.zip`, `.cbz`, hoặc `.pdf` từ Google Drive.

### Module 2: Panel Extraction & Smart Ordering (Tách khung & Sắp xếp)
- **Mô hình:** `leoxs22/manga-panel-detector-yolo26n` (chạy trên GPU T4 mất ~0.05s/trang).
- **Thuật toán sắp xếp thứ tự đọc (RTL Reading Order):**
  - Manga Nhật Bản đọc từ Phải sang Trái (Right-to-Left) và Trên xuống Dưới (Top-to-Bottom).
  - Phân cụm các panel theo tọa độ `y` (hàng), sau đó sắp xếp theo `x` giảm dần để đảm bảo thứ tự logic theo cốt truyện gốc.
- **Tùy chọn:** Inpainting xóa chữ bóng thoại (Text bubble cleaning) bằng mô hình LaMa nhẹ nếu cần ảnh nền sạch.

### Module 3: Vision Director & Scriptwriter (Đạo diễn & Biên kịch AI)
Hỗ trợ 2 phương án tùy nhu cầu (xem chi tiết mục VI):
- **Phương án 1 (Khuyên dùng):** `gemini-3.8-flash` qua Google AI Studio (Free Tier, 0đ, không cần thẻ tín dụng).
- **Phương án 2 (Mã nguồn mở 100%):** `Qwen2.5-VL-7B-Instruct` lượng tử hóa 4-bit chạy trực tiếp trên GPU T4 của Colab.
- **Cơ chế chọn cảnh & viết lời dẫn:**
  - Đưa chuỗi panel vào AI theo từng đợt (batch 10-15 ảnh).
  - Áp dụng **Structured Outputs (Pydantic Schema)** bắt buộc AI xuất ra JSON chuẩn 100%:
    - `panel_file`: Tên ảnh panel được chọn.
    - `dramatic_score`: Điểm kịch tính (1-10) -> Chỉ giữ lại các panel >= 7 điểm để loại bỏ cảnh tĩnh thừa thãi.
    - `narration`: Lời thoại giọng đọc tiếng Việt theo phong cách Quán Khuya (trầm, rùng rợn, gợi mở, tò mò).
    - `camera_motion`: `zoom_in_slow`, `zoom_out_slow`, `pan_left`, `shake`.
    - `sfx_cue`: Tiếng động gợi ý (`creepy_whisper`, `door_creak`, `impact_jumpscare`).

### Module 4: High-Fidelity Voice & Word Timestamps (Giọng đọc & Phụ đề)
1. **TTS Engine (`VieNeu-TTS-v3-Turbo`):**
   - Chạy trên GPU T4 với backend PyTorch (hỗ trợ batching).
   - Chọn preset giọng nam trầm miền Bắc hoặc Nam (ví dụ giọng `NamMinh`).
   - Hỗ trợ Instant Voice Cloning: Lấy mẫu audio 4 giây từ video mẫu để clone chuẩn chất giọng mong muốn.
   - Hỗ trợ chèn các thẻ cảm xúc `[thở dài]`, `[hắng giọng]` trực tiếp vào kịch bản để tăng độ rùng rợn.
2. **Subtitle Engine (`Faster-Whisper`):**
   - Chạy model `small` hoặc `medium` trên CUDA `float16` của T4.
   - Xuất file `.ass` hoặc `.srt` với timestamp chính xác đến từng từ (`word_timestamps=True`) để làm hiệu ứng chữ nhảy động (karaoke style).

### Module 5: Video Assembly & Rendering Engine (Dựng Video)
- **Phương án A (Tự động hóa hoàn toàn - Headless FFmpeg với GPU NVENC):**
  - Tạo hiệu ứng **Ken Burns** (Zoom/Pan chậm khớp chính xác thời lượng câu thoại).
  - Hiệu ứng **Blurred Background Layer**: Khắc phục hiện tượng viền đen hai bên khi tỷ lệ tranh không khớp 16:9 của YouTube.
  - Tự động lồng nhạc nền rùng rợn (BGM) với Audio Ducking (nhạc tự động nhỏ xuống khi có giọng đọc).
  - Sử dụng encoder phần cứng `h264_nvenc` giúp render cả video 10-15 phút chỉ trong 2-3 phút trên Colab T4.
- **Phương án B (Bán tự động qua CapCut / JianYing):**
  - Dùng `pyJianYingDraft` xuất toàn bộ timeline (voice, panel, sub, sfx) thành file project CapCut.
  - Mở CapCut lên chỉnh sửa thủ công nếu muốn thêm hiệu ứng đặc biệt.

---

## IV. Cấu Trúc Dự Án GitHub (Repository Layout)

```
manga-recap-ai/
├── .gitignore
├── LICENSE
├── README.md                      # Hướng dẫn chi tiết cách chạy trên Colab
├── Manga_Recap_Colab.ipynb        # Notebook chạy 1-click trên Google Colab
├── requirements.txt               # Thư viện phụ thuộc
├── config.yaml                    # Cấu hình API key, độ phân giải, giọng đọc
├── src/
│   ├── __init__.py
│   ├── downloader.py              # Tải ảnh từ web/link manga
│   ├── panel_extractor.py         # YOLO panel segmentation & sorting RTL
│   ├── script_generator.py        # Gemini 3.8 Flash Vision / Qwen2.5-VL
│   ├── tts_engine.py              # VieNeu-TTS-v3-Turbo wrapper
│   ├── subtitle_generator.py      # Faster-Whisper word-level timestamp
│   ├── video_engine.py            # FFmpeg NVENC renderer + Ken Burns
│   └── capcut_exporter.py         # Xuất draft CapCut (tùy chọn)
└── assets/
    ├── bgm/                       # Nhạc nền rùng rợn miễn phí bản quyền
    └── sfx/                       # Hiệu ứng âm thanh (tiếng gõ cửa, tiếng hét...)
```

---

## V. Chiến Lược Triển Khai Thực Tế Trên Google Colab

1. **Gắn kết Google Drive (`drive.mount`):**
   - Đảm bảo đầu vào (manga raw, reference voice) và đầu ra (video thành phẩm, project draft) không bị mất khi hết phiên Colab.
2. **Cơ chế lưu Checkpoint trung gian (Cache Step):**
   - Sau khi cắt panel -> Lưu vào Drive.
   - Sau khi AI tạo kịch bản JSON -> Lưu vào file `timeline.json`.
   - Sau khi sinh audio -> Lưu file `.wav` + `.srt`.
3. **Quản lý Secrets an toàn:**
   - Dùng tính năng *Colab Secrets* để lưu API key (nếu dùng Gemini).

---

## VI. Báo Cáo Nghiên Cứu Chi Phí & Giải Pháp MIỄN PHÍ 100%

### 1. Google AI Studio (`gemini-3.8-flash`) có miễn phí không?
- **TRẢ LỜI: CÓ, HOÀN TOÀN MIỄN PHÍ 100%!**
- **Chính sách Free Tier của Google AI Studio:**
  - **Không yêu cầu thẻ tín dụng/Visa:** Chỉ cần đăng nhập bằng tài khoản Google cá nhân tại `aistudio.google.com` là tạo được API Key ngay lập tức.
  - **Mức phí:** 0 USD (Free of charge) cho model `gemini-3.8-flash` và các bản `flash-lite`.
  - **Hạn mức (Rate Limit):**
    - ~10 - 15 requests/phút (RPM).
    - 250 - 500 requests/ngày (RPD).
  - **Đánh giá thực tế:** 1 video tóm tắt chapter truyện dài 10-15 phút chỉ cần từ **3 - 8 requests API** (mỗi request gửi 1 chùm 10-15 ảnh panel). Hạn mức 250+ request/ngày cho phép bạn làm **20 - 30 video mỗi ngày hoàn toàn miễn phí**!
  - **Lưu ý nhỏ:** Ở gói Free, Google có điều khoản dùng dữ liệu prompt để cải thiện AI. Vì chúng ta chỉ phân tích truyện tranh công khai nên điều này hoàn toàn vô hại.

---

### 2. Phương Án Dự Phòng: MÃ NGUỒN MỞ 100% (Hoàn toàn Offline trên Colab T4)
Nếu bạn không muốn phụ thuộc vào bất kỳ bên thứ 3 nào hoặc không muốn dùng API Key:

- **Mô hình VLM mã nguồn mở số 1 hiện nay:** **`Qwen2.5-VL-7B-Instruct`** (Alibaba Open Source).
- **Khả năng chạy trên Colab GPU T4 (16GB VRAM):**
  - Áp dụng lượng tử hóa 4-bit (`bitsandbytes`, `load_in_4bit=True`).
  - Chiếm khoảng **~5.5 GB VRAM** trên Colab T4 (vẫn dư ~10 GB VRAM cho VieNeu và Whisper).
  - **Ưu điểm:**
    - Hoàn toàn miễn phí, chạy offline cục bộ trên máy ảo Colab.
    - Không bị giới hạn số lần gọi/ngày.
    - Hiểu tiếng Việt khá tốt và đọc ảnh ô tranh sắc nét.
  - **Nhược điểm so với Gemini 3.8 Flash:**
    - Tốc độ sinh text chậm hơn Gemini API (mất khoảng 15-30 giây cho mỗi lượt đọc ảnh).
    - Cần thời gian tải weights model (~5GB) vào Colab lúc khởi động notebook lần đầu.

👉 **Kết luận chiến lược:** Trong code dự án, ta sẽ thiết kế một interface linh hoạt: Mặc định ưu tiên dùng **Gemini 3.8 Flash (Free Tier)** cho tốc độ cao và lời văn mượt mà nhất; đồng thời có sẵn switch chuyển sang **Qwen2.5-VL-7B 4bit (Open-source)** khi không có mạng hoặc không muốn dùng API key.
