"""
Module 3: Vision Director & Horror Script Generator
Leverages Qwen2.5-VL (Local 4-bit GPU on Colab) or Gemini Flash (Free Cloud API) to:
1. Filter key climax/dramatic panels (eliminating boring filler panels).
2. Generate dramatic Vietnamese recap narration in the style of channel 'Quán Khuya'.
3. Output structured timeline JSON ready for TTS and video assembly.
"""

import os
import re
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
        provider: str = "qwen_vl",
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        output_file: str = "./workspace/timeline.json",
        quantization: str = "4bit"
    ):
        raw_model = (model_name or "").strip()
        # Auto-detect provider if model name gives explicit clue
        if "gemini" in raw_model.lower():
            self.provider = "gemini"
            self.model_name = raw_model or "gemini-3.8-flash"
        elif "qwen" in raw_model.lower():
            self.provider = "qwen_vl"
            self.model_name = raw_model or "Qwen/Qwen2.5-VL-7B-Instruct"
        else:
            self.provider = provider.lower()
            if self.provider == "qwen_vl":
                self.model_name = raw_model or "Qwen/Qwen2.5-VL-7B-Instruct"
            else:
                self.model_name = raw_model or "gemini-3.8-flash"

        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.quantization = quantization

    def generate_timeline(
        self,
        panels_metadata: List[Dict],
        story_synopsis: str = "",
        min_score: int = 7,
        max_scenes: int = 35
    ) -> Dict:
        """
        Takes list of extracted panels, analyzes visual content, and returns structured timeline.
        """
        print(f"[Script Generator] Generating recap script via provider '{self.provider}' (model: {self.model_name})...")

        if self.provider == "qwen_vl":
            return self._generate_with_qwen_vl(panels_metadata, story_synopsis, min_score, max_scenes)
        elif self.provider == "gemini":
            return self._generate_with_gemini(panels_metadata, story_synopsis, min_score, max_scenes)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _generate_simulation(
        self,
        panels: List[Dict],
        synopsis: str,
        min_score: int,
        max_scenes: int,
        note: str = "simulation"
    ) -> Dict:
        """Fallback simulation generator when offline or APIs are temporarily down."""
        print(f"[Script Generator] Generating storyboard in fallback {note} mode...")
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
            return self._generate_simulation(panels, synopsis, min_score, max_scenes, note="offline simulation (No API Key)")

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
        Offline local inference using Qwen2.5-VL (3B or 7B-Instruct with 4-bit quantization).
        Optimized for Google Colab GPU T4 (16GB VRAM) and automatically releases VRAM upon completion.
        """
        model_id = self.model_name or "Qwen/Qwen2.5-VL-7B-Instruct"
        print(f"[Script Generator] Initializing local Vision-LLM: {model_id}...")

        try:
            import torch
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        except ImportError:
            print("[Script Generator] Torch or transformers not installed. Falling back to simulation mode...")
            return self._generate_simulation(panels, synopsis, min_score, max_scenes, note="simulation (Missing transformers)")

        has_cuda = torch.cuda.is_available()
        if not has_cuda:
            print("[Script Generator] No CUDA GPU detected. Running large models on CPU may take hours.")
            print("[Script Generator] Using simulation fallback for non-GPU environment...")
            return self._generate_simulation(panels, synopsis, min_score, max_scenes, note="simulation (No CUDA GPU)")

        # Prepare 4-bit quantization config if on GPU
        model_kwargs = {"device_map": "auto"}
        if self.quantization == "4bit":
            try:
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4"
                )
                print("[Script Generator] 4-bit BitsAndBytes quantization enabled (VRAM ~5.5GB).")
            except Exception as e:
                print(f"[Script Generator] Warning: BitsAndBytes unavailable ({e}), using torch.float16.")
                model_kwargs["torch_dtype"] = torch.float16
        else:
            model_kwargs["torch_dtype"] = torch.float16

        print(f"[Script Generator] Loading weights for {model_id} into GPU VRAM...")
        try:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **model_kwargs)
            processor = AutoProcessor.from_pretrained(model_id)
        except Exception as e:
            print(f"[Script Generator] Failed to load {model_id}: {e}. Falling back to simulation...")
            return self._generate_simulation(panels, synopsis, min_score, max_scenes, note=f"simulation (Load error: {e})")

        # Try to import qwen_vl_utils helper
        try:
            from qwen_vl_utils import process_vision_info
            has_qwen_utils = True
        except ImportError:
            has_qwen_utils = False

        # Filter candidate panels (take evenly spaced samples if too many panels)
        if len(panels) > max_scenes:
            step = len(panels) / max_scenes
            candidate_panels = [panels[int(i * step)] for i in range(max_scenes)]
        else:
            candidate_panels = panels

        print(f"[Script Generator] Analyzing {len(candidate_panels)} candidate panels with {model_id}...")
        selected_scenes = []

        system_prompt = (
            "Bạn là biên kịch kiêm đạo diễn video review / recap truyện tranh phong cách kinh dị giật gân (Quán Khuya).\n"
            f"Bối cảnh truyện: {synopsis or 'Không khí căng thẳng, u ám, bí ẩn kinh dị'}.\n"
            "Hãy nhìn kỹ khung tranh và trả về DUY NHẤT một đối tượng JSON có cấu trúc sau:\n"
            "{\n"
            '  "dramatic_score": <số nguyên từ 1 đến 10>,\n'
            '  "narration": "<lời kể dẫn chuyện tiếng Việt rùng rợn, gợi cảm giác hồi hộp, có thể kèm tag [thở dài] hoặc [tiếng thở dốc]>",\n'
            '  "camera_motion": "<chọn 1 trong: zoom_in, zoom_out, pan_left, pan_right>",\n'
            '  "sfx_cue": "<chọn 1 trong: heart_beat, door_creak, creepy_whisper, jumpscare, none>"\n'
            "}"
        )

        for idx, p in enumerate(candidate_panels, start=1):
            try:
                img_path = p["file_path"]
                if not os.path.exists(img_path):
                    continue

                img = Image.open(img_path).convert("RGB")
                # Resize if image is excessively large to save memory and inference time
                max_dim = 1024
                if max(img.size) > max_dim:
                    img.thumbnail((max_dim, max_dim))

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": system_prompt}
                        ]
                    }
                ]

                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

                if has_qwen_utils:
                    image_inputs, video_inputs = process_vision_info(messages)
                    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
                else:
                    inputs = processor(text=[text], images=[img], padding=True, return_tensors="pt")

                inputs = inputs.to("cuda")

                with torch.no_grad():
                    generated_ids = model.generate(**inputs, max_new_tokens=180, do_sample=False)
                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    output_text = processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )[0]

                # Parse JSON output from Qwen
                score = 8
                motion = "zoom_in" if idx % 2 == 1 else "pan_left"
                sfx = "creepy_whisper" if idx % 2 == 1 else "heart_beat"
                narration = ""

                cleaned = output_text.strip()
                # Remove code blocks if present
                if "```json" in cleaned:
                    cleaned = cleaned.split("```json")[-1].split("```")[0].strip()
                elif "```" in cleaned:
                    cleaned = cleaned.split("```")[-1].split("```")[0].strip()

                json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
                if json_match:
                    try:
                        data = json.loads(json_match.group(0))
                        score = int(data.get("dramatic_score", 8))
                        narration = str(data.get("narration", "")).strip()
                        motion = str(data.get("camera_motion", motion)).strip()
                        sfx = str(data.get("sfx_cue", sfx)).strip()
                    except Exception:
                        narration = cleaned
                else:
                    narration = cleaned

                if not narration or len(narration) < 10:
                    narration = f"Khung cảnh thứ {idx} chìm trong bóng tối... Có điều gì đó bất thường đang diễn ra. [thở dài]"

                if score >= min_score or len(selected_scenes) < 5:
                    selected_scenes.append({
                        "panel_file": p["file_name"],
                        "dramatic_score": score,
                        "narration": narration,
                        "camera_motion": motion,
                        "sfx_cue": sfx
                    })
                    print(f"  [Qwen-VL] Scene {len(selected_scenes)}/{max_scenes} (Score {score}): {narration[:60]}...")

                if len(selected_scenes) >= max_scenes:
                    break

            except Exception as e:
                print(f"  [Qwen-VL] Error processing panel {p.get('file_name', '')}: {e}")
                selected_scenes.append({
                    "panel_file": p["file_name"],
                    "dramatic_score": 7,
                    "narration": f"Cơn ác mộng lại tiếp diễn trong tĩnh lặng... [thở dài]",
                    "camera_motion": "zoom_in",
                    "sfx_cue": "heart_beat"
                })

        # CRUCIAL: Unload model and free GPU VRAM for VieNeu-TTS and Faster-Whisper
        print("[Script Generator] Releasing Qwen-VL model from VRAM to preserve memory...")
        del model
        del processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()
        print("✅ [Script Generator] VRAM successfully cleared.")

        final_timeline = {
            "title": f"Bí Ẩn Kinh Hoàng Đằng Sau Nụ Cười ({model_id.split('/')[-1]})",
            "intro_hook": "Bóng tối đang dần nuốt chửng tất cả... Đừng bao giờ ngoảnh đầu lại.",
            "total_scenes": len(selected_scenes),
            "scenes": selected_scenes
        }

        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(final_timeline, f, ensure_ascii=False, indent=2)

        print(f"[Script Generator] Finished writing {len(selected_scenes)} scenes to {self.output_file}")
        return final_timeline
