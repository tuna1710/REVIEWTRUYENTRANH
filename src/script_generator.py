"""
Module 3: Vision Director & Horror Script Generator
Leverages Gemini 3.8 Flash (Free API) or Qwen2.5-VL (Local 4-bit) to:
1. Filter 30-40 key climax/dramatic panels (eliminating boring filler panels).
2. Generate dramatic Vietnamese recap narration in the style of channel 'Quán Khuya'.
3. Output structured timeline JSON ready for TTS and video assembly.
"""

import os
import json
import base64
from pathlib import Path
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from PIL import Image


# --- Pydantic Schema for Structured Output ---
class SceneItem(BaseModel):
    panel_file: str = Field(description="Exact filename of the selected panel, e.g. 'panel_0001.png'")
    dramatic_score: int = Field(description="Climax and tension score from 1 to 10")
    narration: str = Field(description="Atmospheric Vietnamese narration in creepy/suspenseful storytelling tone")
    camera_motion: str = Field(description="Camera movement: zoom_in, zoom_out, pan_left, pan_right, or static")
    sfx_cue: str = Field(description="Suggested sound effect: heart_beat, door_creak, creepy_whisper, jumpscare, or none")


class ScriptTimeline(BaseModel):
    title: str = Field(description="Catchy suspenseful YouTube video title")
    intro_hook: str = Field(description="5-second opening hook phrase to grab viewer attention")
    scenes: List[SceneItem] = Field(description="Ordered list of selected dramatic scenes")


class ScriptGenerator:
    def __init__(
        self,
        provider: str = "gemini",
        api_key: Optional[str] = None,
        model_name: str = "gemini-3.8-flash",
        output_file: str = "./workspace/timeline.json"
    ):
        self.provider = provider.lower()
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model_name = model_name
        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

    def generate_timeline(
        self,
        panels_metadata: List[Dict],
        story_synopsis: str = "",
        min_score: int = 7,
        max_scenes: int = 40
    ) -> Dict:
        """
        Takes list of extracted panels, analyzes visual content, and returns structured timeline.
        """
        print(f"[Script Generator] Generating recap script via provider '{self.provider}' (model: {self.model_name})...")

        if self.provider == "gemini":
            return self._generate_with_gemini(panels_metadata, story_synopsis, min_score, max_scenes)
        elif self.provider == "qwen_vl":
            return self._generate_with_qwen_vl(panels_metadata, story_synopsis, min_score, max_scenes)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _generate_with_gemini(
        self,
        panels: List[Dict],
        synopsis: str,
        min_score: int,
        max_scenes: int
    ) -> Dict:
        """
        Calls Gemini Flash using the official google-genai SDK with Structured Outputs.
        """
        if not self.api_key:
            print("[Script Generator] GEMINI_API_KEY is not set. Generating storyboard in offline simulation mode...")
            selected_scenes = []
            for idx, p in enumerate(panels[:max_scenes], start=1):
                selected_scenes.append({
                    "panel_file": p["file_name"],
                    "dramatic_score": 8,
                    "narration": f"Khung cảnh thứ {idx} chìm trong bóng tối và sự im lặng rợn người... Điều kinh hoàng đang dần lộ diện.",
                    "camera_motion": "zoom_in" if idx % 2 == 1 else "pan_left",
                    "sfx_cue": "heart_beat" if idx % 2 == 1 else "creepy_whisper"
                })
            final_timeline = {
                "title": "Bí Ẩn Kinh Hoàng Đằng Sau Nụ Cười",
                "intro_hook": "Đừng bao giờ quay đầu lại nếu bạn nghe thấy tiếng gõ cửa lúc nửa đêm...",
                "total_scenes": len(selected_scenes),
                "scenes": selected_scenes
            }
            with open(self.output_file, "w", encoding="utf-8") as f:
                json.dump(final_timeline, f, ensure_ascii=False, indent=2)
            return final_timeline

        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise ImportError("Please install google-genai: pip install google-genai")

        client = genai.Client(api_key=self.api_key)

        batch_size = 15
        selected_scenes = []
        captured_title = None
        captured_intro_hook = None

        system_instruction = (
            "Bạn là biên kịch kiêm đạo diễn video YouTube tóm tắt truyện tranh giật gân, kinh dị chuyên nghiệp (phong cách kênh Quán Khuya).\n"
            "Nhiệm vụ của bạn:\n"
            "1. Quan sát kỹ từng khung tranh (panel) được gắn thẻ tên file rõ ràng.\n"
            "2. Đánh giá độ kịch tính và ý nghĩa của tranh theo thang điểm 1-10.\n"
            "3. LỌC BỎ các tranh phụ, khung cảnh tĩnh không quan trọng. CHỈ CHỌN các tranh có điểm kịch tính cao (>= 7).\n"
            "4. Với mỗi tranh được chọn, viết lời bình dẫn chuyện bằng tiếng Việt mang giọng điệu trầm, bí ẩn, rùng rợn và kích thích trí tò mò.\n"
            "5. BẮT BUỘC trả về đúng chính xác trường 'panel_file' theo đúng tên file tương ứng với khung tranh bạn chọn.\n"
            "6. Có thể lồng ghép các biểu cảm giọng nói như [thở dài], [tiếng thở dốc] để giọng đọc chân thực."
        )

        for i in range(0, len(panels), batch_size):
            chunk = panels[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(panels) + batch_size - 1) // batch_size
            print(f"[Script Generator] Sending batch {batch_num}/{total_batches} ({len(chunk)} panels) to Gemini ({self.model_name})...")

            contents_payload = [
                f"Cốt truyện tổng thể: {synopsis or 'Tóm tắt diễn biến kịch tính của bộ truyện tranh'}\n"
                f"Dưới đây là các khung tranh theo thứ tự đọc Manga (phải sang trái, trên xuống dưới). "
                f"Hãy quan sát kỹ từng tranh và chọn ra các cảnh đắt giá nhất:"
            ]

            for p in chunk:
                try:
                    img = Image.open(p["file_path"])
                    contents_payload.append(f"Khung tranh (filename: '{p['file_name']}'):")
                    contents_payload.append(img)
                except Exception as e:
                    print(f"Warning: could not open image {p['file_path']}: {e}")

            contents_payload.append(
                "Hãy chọn các cảnh đắt giá nhất (dramatic_score >= 7), viết lời bình tiếng Việt rùng rợn, "
                "đề xuất camera motion và sfx, đảm bảo trường 'panel_file' đúng chính xác tên file đã gửi."
            )

            try:
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=contents_payload,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        response_mime_type="application/json",
                        response_schema=ScriptTimeline,
                        temperature=0.7,
                    ),
                )
                data = json.loads(response.text)

                if not captured_title and data.get("title"):
                    captured_title = data.get("title")
                if not captured_intro_hook and data.get("intro_hook"):
                    captured_intro_hook = data.get("intro_hook")

                for sc in data.get("scenes", []):
                    if sc.get("dramatic_score", 0) >= min_score:
                        selected_scenes.append(sc)
            except Exception as e:
                print(f"[Script Generator] Batch failed: {e}. Generating fallback entries for this batch.")
                for p in chunk:
                    selected_scenes.append({
                        "panel_file": p["file_name"],
                        "dramatic_score": 7,
                        "narration": f"Cơn ác mộng lại tiếp diễn trong tĩnh lặng... [thở dài]",
                        "camera_motion": "zoom_in",
                        "sfx_cue": "heart_beat"
                    })

        selected_scenes = selected_scenes[:max_scenes]

        final_timeline = {
            "title": captured_title or "Bí Ẩn Kinh Hoàng Đằng Sau Nụ Cười",
            "intro_hook": captured_intro_hook or "Đừng bao giờ quay đầu lại nếu bạn nghe thấy tiếng gõ cửa lúc nửa đêm...",
            "total_scenes": len(selected_scenes),
            "scenes": selected_scenes
        }

        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(final_timeline, f, ensure_ascii=False, indent=2)

        print(f"[Script Generator] Successfully generated {len(selected_scenes)} selected scenes in {self.output_file}")
        return final_timeline

    def _generate_with_qwen_vl(
        self,
        panels: List[Dict],
        synopsis: str,
        min_score: int,
        max_scenes: int
    ) -> Dict:
        """
        Offline fallback using Qwen2.5-VL-7B-Instruct with 4-bit quantization on Colab GPU T4.
        """
        print("[Script Generator] Loading Qwen2.5-VL-7B-Instruct with 4-bit quantization...")
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"
        )

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            quantization_config=quantization_config,
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

        selected_scenes = []
        for p in panels[:max_scenes]:
            prompt = (
                "Bạn là biên kịch recap truyện kinh dị Quán Khuya. "
                "Hãy nhìn tranh và viết 1 câu dẫn chuyện rùng rợn bằng tiếng Việt, kèm camera motion (zoom_in/zoom_out) và sfx. "
                "Trả về JSON dạng: {\"narration\": \"...\", \"camera_motion\": \"zoom_in\", \"sfx_cue\": \"door_creak\"}"
            )
            image = Image.open(p["file_path"])
            messages = [
                {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda")

            generated_ids = model.generate(**inputs, max_new_tokens=150)
            output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            selected_scenes.append({
                "panel_file": p["file_name"],
                "dramatic_score": 8,
                "narration": output_text.strip(),
                "camera_motion": "zoom_in",
                "sfx_cue": "heart_beat"
            })

        final_timeline = {
            "title": "Bí Ẩn Kinh Hoàng (Offline Qwen-VL)",
            "intro_hook": "Bóng tối đang dần nuốt chửng tất cả...",
            "total_scenes": len(selected_scenes),
            "scenes": selected_scenes
        }

        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(final_timeline, f, ensure_ascii=False, indent=2)

        return final_timeline
