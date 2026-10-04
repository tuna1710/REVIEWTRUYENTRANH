# 📋 Danh Sách Mô Hình Gemini API Chính Thức (Google AI for Developers)
> Nguồn tài liệu chính thức: [https://ai.google.dev/gemini-api/docs/models?hl=vi](https://ai.google.dev/gemini-api/docs/models?hl=vi)  
> Cập nhật: Tháng 10/2026

Tài liệu này lưu trữ danh sách các điểm cuối (endpoint) và thông số các mô hình Gemini API để làm tài liệu tham khảo cho dự án `REVIEWTRUYENTRANH`.

---

## 🌟 1. Dòng Mô Hình Gemini 3 (Thế Hệ Mới Nhất - Khuyên Dùng)

| Tên Mô Hình | Endpoint (Model String) | Trạng Thái | Mô Tả & Trường Hợp Sử Dụng |
| :--- | :--- | :--- | :--- |
| **Gemini 3.8 Flash** | `gemini-3.8-flash` | **Mới - Ổn định (Khuyên dùng)** | Mô hình Flash thông minh nhất, tối ưu cho phân tích thị giác, tác nhân tự trị và quy trình suy luận phức tạp. Hỗ trợ Structured Outputs JSON & Vision cực mạnh. |
| **Gemini 3.8 Live** | `gemini-3.8-live` | Mới - Ổn định | Tác nhân giọng nói độ trễ thấp theo thời gian thực. |
| **Gemini 3.8 Live Extended Thinking** | `gemini-3.8-live-extended-thinking` | Mới - Ổn định | Tương tác giọng nói có suy luận sâu trong nền. |
| **Gemini 3.8 Flash TTS** | `gemini-3.8-flash-tts` | Mới - Ổn định | Chuyển văn bản thành giọng nói chuẩn phòng thu, diễn xuất biểu cảm, hỗ trợ 130 ngôn ngữ. |
| **Gemini 3.8 Flash-Lite TTS** | `gemini-3.8-flash-lite-tts` | Mới - Ổn định | TTS tốc độ cao, tiết kiệm chi phí, hỗ trợ 101 ngôn ngữ. |
| **Gemini 3.7 Flash** | `gemini-3.7-flash` | Ổn định | Mô hình Flash thế hệ 3.7, suy luận đa bước ổn định. |
| **Gemini 3.6 Flash** | `gemini-3.6-flash` | Ổn định | Cân bằng tốc độ và khả năng đa phương thức. |
| **Gemini 3.5 Flash** | `gemini-3.5-flash` | Ổn định | Hiệu suất nền tảng cho khối lượng công việc lớn. |
| **Gemini 3.5 Flash-Lite** | `gemini-3.5-flash-lite` | **Ổn định (Siêu tốc)** | Mô hình nhanh nhất, tiết kiệm chi phí nhất dòng 3.5. Rất phù hợp làm fallback tốc độ cao. |
| **Gemini 3.1 Flash-Lite** | `gemini-3.1-flash-lite` | Ổn định | Hiệu suất cao với chi phí thấp. |
| **Gemini 3.1 Pro** | `gemini-3.1-pro-preview` | Xem trước | Trí tuệ nâng cao, suy luận sâu và viết code. |
| **Gemini 3 Flash** | `gemini-3-flash-preview` | Xem trước | Bản xem trước đầu tiên của dòng Flash 3. |
| **Gemini 3.5 Transcribe** | `gemini-3.5-transcribe` / `gemini-3.5-transcribe-live` | Mới - Ổn định | Bóc tách âm thanh thành văn bản kèm dấu thời gian từ (word timestamps). |

---

## ⚡ 2. Dòng Mô Hình Gemini 2.5 (Thế Hệ Trước)
*Lưu ý từ Google:* Google giới hạn các mô hình 2.5 cho người dùng cũ. Đối với dự án mới, Google khuyến nghị dùng `gemini-3.8-flash` hoặc `gemini-3.5-flash-lite`.

| Tên Mô Hình | Endpoint | Trạng Thái | Ghi Chú |
| :--- | :--- | :--- | :--- |
| **Gemini 2.5 Flash** | `gemini-2.5-flash` | Khả dụng (Di chuyển dần) | Hiệu năng tốt cho tác vụ độ trễ thấp thế hệ 2.5. |
| **Gemini 2.5 Flash-Lite**| `gemini-2.5-flash-lite` | Khả dụng (Di chuyển dần) | Bản rút gọn siêu tốc thế hệ 2.5. |
| **Gemini 2.5 Pro** | `gemini-2.5-pro` | Khả dụng (Di chuyển dần) | Mô hình suy luận sâu thế hệ 2.5. |
| **Gemini 2.5 Flash Live**| `gemini-2.5-flash-native-audio-preview-12-2025` | Xem trước | Voice Live API thế hệ 2.5. |
| **Gemini 2.5 Flash TTS** | `gemini-2.5-flash-preview-tts` | Xem trước | TTS thế hệ 2.5. |

---

## 🎨 3. Mô Hình Hình Ảnh, Video & Âm Nhạc (Generative Media)

| Tên Mô Hình | Endpoint | Loại | Chức Năng |
| :--- | :--- | :--- | :--- |
| **Nano Banana 2** | `gemini-3.1-flash-image` | Ảnh | Sinh và chỉnh sửa ảnh tốc độ cao. |
| **Nano Banana 2 Lite** | `gemini-3.1-flash-lite-image` | Ảnh | Sinh ảnh độ trễ cực thấp. |
| **Nano Banana Pro** | `gemini-3-pro-image` | Ảnh | Sinh ảnh 4K chi tiết cao, render chữ chuẩn xác. |
| **Veo 3.1** | `veo-3.1-generate-preview` | Video | Sinh video điện ảnh kèm âm thanh đồng bộ. |
| **Veo 3.1 Lite** | `veo-3.1-lite-generate-preview` | Video | Sinh video tốc độ cao, chi phí tối ưu. |
| **Gemini Omni Flash** | `gemini-omni-1.1-flash` | Video/Audio | Chỉnh sửa video và mở rộng khung hình với âm thanh gốc. |
| **Lyria 3.5** | `lyria-3.5` | Nhạc (BGM) | Tạo bài hát hoàn chỉnh cấu trúc phức tạp. |
| **Lyria 3 Clip** | `lyria-3-clip-preview` | Nhạc (BGM) | Tạo đoạn nhạc ngắn/vòng lặp BGM tới 30 giây. |
| **Lyria RealTime** | `lyria-realtime-exp` | Nhạc (BGM) | Tạo nhạc theo thời gian thực. |

---

## 🤖 4. Mô Hình Tác Nhân & Chuyên Biệt (Agents & Specialized)

| Tên Mô Hình | Endpoint | Chức Năng |
| :--- | :--- | :--- |
| **Gemini Deep Research** | `deep-research-preview-04-2026` | Tự động nghiên cứu tài liệu từ hàng trăm nguồn và viết báo cáo. |
| **Antigravity Agent** | `antigravity-preview-09-2026` | Tác nhân quản lý tự động chạy code và duyệt web an toàn trong sandbox Linux. |
| **Gemini Embedding 2** | `gemini-embedding-2-preview` | Vector embedding đa phương thức (text, image, audio, video, PDF). |

---

## ⚠️ 5. Các Mô Hình Đã Tắt / Ngừng Hoạt Động (Shut Down)
Tuyệt đối **không sử dụng** các endpoint này vì sẽ báo lỗi:
- ❌ `gemini-2.0-flash` (Đã tắt - Shut down)
- ❌ `gemini-2.0-flash-lite` (Đã tắt - Shut down)
- ❌ `gemini-3.1-flash-lite-preview` (Đã tắt - Thay bằng `gemini-3.1-flash-lite`)
- ❌ `gemini-3-pro-preview` (Đã tắt)
- ❌ `imagen-4.0-generate` (Đã tắt)

---

## 🎯 Cấu Hình Khuyến Nghị Cho Dự Án `REVIEWTRUYENTRANH`
1. **Mặc định:** `gemini-3.8-flash` (Đầy đủ tính năng phân tích hình ảnh Manga, Structured Output JSON và tốc độ vượt trội).
2. **Dự phòng tốc độ cao / Tiết kiệm token:** `gemini-3.5-flash-lite`.
3. **Dự phòng thế hệ 2.5:** `gemini-2.5-flash`.
