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


def _is_english_text(text: str) -> bool:
    """
    Detects if text contains primarily English speech bubbles or leaked thoughts.
    """
    if not text:
        return False
    en_words = {
        'the', 'a', 'an', 'she', 'he', 'it', 'they', 'this', 'that', 'these', 'those',
        'is', 'are', 'was', 'were', 'what', 'with', 'from', 'her', 'his', 'my', 'your',
        'can', 'cant', 'cannot', 'could', 'thought', 'shiver', 'spine', 'monster',
        'thing', 'down', 'oh', 'no', 'of', 'in', 'on', 'at', 'to', 'for', 'and', 'but',
        'staring', 'looking', 'walking', 'scared', 'afraid', 'creature', 'behind', 'run',
        'there', 'somebody', 'someone', 'please', 'help', 'why', 'who', 'how', 'me'
    }
    words = [re.sub(r'[^a-zA-Z]', '', w.lower()) for w in text.split()]
    words = [w for w in words if w]
    if not words:
        return False
    en_count = sum(1 for w in words if w in en_words)
    return en_count >= 2 or (len(words) >= 4 and en_count / len(words) >= 0.25)


def _contextual_vietnamese_horror(idx: int, total_scenes: int, synopsis: str = "") -> str:
    """
    Generates contextual, gripping Vietnamese horror narration when an LLM leaks English or truncated JSON.
    Ensures 100% pure Vietnamese storytelling continuity across all scenes.
    """
    ratio = idx / max(1, total_scenes)
    if ratio <= 0.3:
        templates = [
            "Bầu không khí u ám đến nghẹt thở bao trùm lấy không gian... Có điều gì đó bất thường đang âm thầm diễn ra.",
            "Từng bước chân nặng trĩu trong sự im lặng đáng sợ, như thể có một ánh nhìn vô hình đang dõi theo từ trong bóng tối.",
            "Không gian xung quanh dường như đông cứng lại, một luồng khí lạnh buốt bất ngờ ùa tới khiến ai nấy đều rùng mình."
        ]
    elif ratio <= 0.7:
        templates = [
            "Cảm giác ớn lạnh chạy dọc sống lưng... Một hình thù quái đản và dị hợm bất ngờ xuất hiện ngay trong tầm mắt.",
            "Cô gái cố gắng kìm nén hơi thở, tim đập thình thịch và giả vờ như hoàn toàn không nhìn thấy sinh vật ghê rợn kia.",
            "Thực thể ma quái ghé sát lại gần với hơi thở tanh tưởi, chỉ cần để lộ một chút hoảng sợ, hậu quả sẽ không thể lường trước.",
            "Toàn thân cô tê dại vì sợ hãi nhưng đôi mắt vẫn phải nhìn thẳng về phía trước để che giấu sự hoảng loạn."
        ]
    else:
        templates = [
            "Áp lực kinh hoàng đè nặng từng giây từng phút khi quái vật bắt đầu nghi ngờ và tiến sát hơn nữa.",
            "Một khoảnh khắc đối mặt sinh tử đầy nghẹt thở... Ranh giới giữa sự sống và cái chết chưa bao giờ mong manh đến thế.",
            "Cơn ác mộng dường như vẫn chưa thể kết thúc, sự kinh hoàng tột độ vẫn tiếp tục đeo bám trong từng hơi thở."
        ]
    return templates[(idx - 1) % len(templates)]


def _sanitize_vietnamese_text(text: str) -> str:
    """
    Cleans up Chinese leaks, strips all bracket emotion tags (e.g. [thở dài], [tiếng thở dốc]),
    and removes any leaked JSON tokens.
    Ensures 100% pure Vietnamese narration suitable for TTS narration.
    """
    if not text:
        return ""

    # 1. Strip all bracket tags completely (e.g., [thở dài], [tiếng thở dốc], [hắng giọng], etc.)
    text = re.sub(r"\[[^\]]*\]", "", text)

    # 2. Strip any leaked JSON key/syntax markers
    text = re.sub(r'^\s*\{?\s*"dramatic_score"\s*:\s*\d+,?\s*', '', text)
    text = re.sub(r'^\s*"?narration"?\s*:\s*["\']?', '', text)
    text = re.sub(r'["\'\}]+\s*$', '', text)

    # 3. Common Chinese leaks from Qwen visual tokenizer
    replacements = {
        "đồng phục校服": "đồng phục",
        "校服": "đồng phục",
        "áo mưa雨衣": "áo mưa",
        "雨衣": "áo mưa",
        "chiếc ô雨伞": "chiếc ô",
        "cây dù雨伞": "cây dù",
        "雨伞": "chiếc ô",
        "lớp học教室": "lớp học",
        "教室": "lớp học",
        "thầy cô老师": "giáo viên",
        "老师": "giáo viên",
        "học sinh学生": "học sinh",
        "学生": "học sinh",
        "quái vật怪物": "quái vật",
        "怪物": "quái vật",
        "bóng ma幽灵": "bóng ma",
        "幽灵": "bóng ma",
        "dưới mưa雨中": "dưới mưa",
        "雨中": "dưới mưa",
    }
    for zh, vi in replacements.items():
        text = text.replace(zh, vi)

    # 4. Remove any remaining CJK unified ideographs
    text = re.sub(r"[\u4e00-\u9fff]+", "", text)

    # 5. Deduplicate duplicated words
    for phrase in ["đồng phục", "áo mưa", "chiếc ô", "quái vật", "bóng ma", "dưới mưa", "học sinh"]:
        text = text.replace(f"{phrase} {phrase}", phrase).replace(f"{phrase}{phrase}", phrase)

    # 6. Clean up redundant spaces and punctuation spacing
    text = re.sub(r"\s+([,.:;?!])", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip('"\' ')
    return text


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
            self.provider = (provider or "qwen_vl").lower()
            if self.provider == "gemini":
                self.model_name = raw_model or "gemini-3.8-flash"
            else:
                self.model_name = raw_model or "Qwen/Qwen2.5-VL-7B-Instruct"

        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
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
        Orchestrates selection of panels and generation of dramatic Vietnamese recap script.
        """
        print(f"[Script Generator] Generating recap script via provider '{self.provider}' (model: {self.model_name})...")

        if self.provider == "gemini":
            return self._generate_with_gemini(panels_metadata, story_synopsis, min_score, max_scenes)
        elif self.provider == "qwen_vl":
            return self._generate_qwen_vl(panels_metadata, story_synopsis, min_score, max_scenes)
        else:
            print(f"[Script Generator] Unsupported provider '{self.provider}'. Falling back to simulation.")
            return self._generate_simulation(panels_metadata, story_synopsis, min_score, max_scenes, note="simulation (Unsupported provider)")

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
                "narration": _contextual_vietnamese_horror(idx, min(len(panels), max_scenes), synopsis),
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
            "4. Với mỗi tranh được chọn, viết lời bình dẫn chuyện 100% BẰNG TIẾNG VIỆT mang giọng điệu trầm, bí ẩn, rùng rợn và kích thích trí tò mò.\n"
            "   TUYỆT ĐỐI KHÔNG VIẾT TIẾNG ANH dù trong tranh có chữ tiếng Anh hay tiếng Nhật.\n"
            "5. BẮT BUỘC trả về đúng chính xác trường 'panel_file' theo đúng tên file tương ứng với khung tranh bạn chọn.\n"
            "6. Diễn biến câu chuyện phải nối tiếp liền mạch, không lặp lại mô tả ở các cảnh liên tiếp. TUYỆT ĐỐI KHÔNG dùng bất kỳ thẻ cảm xúc nào trong ngoặc vuông như [thở dài] hay [tiếng thở dốc]."
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
                "Hãy chọn các cảnh đắt giá nhất (dramatic_score >= 7), viết lời bình 100% tiếng Việt rùng rợn thuần túy (không dùng tiếng Anh, không kèm tag [thở dài]), "
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
                        raw_narr = sc.get("narration", "")
                        clean_narr = _sanitize_vietnamese_text(raw_narr)
                        if _is_english_text(clean_narr):
                            clean_narr = _contextual_vietnamese_horror(len(selected_scenes) + 1, max_scenes, synopsis)
                        sc["narration"] = clean_narr
                        selected_scenes.append(sc)
            except Exception as e:
                print(f"[Script Generator] Batch failed: {e}. Generating fallback entries for this batch.")
                for p in chunk:
                    selected_scenes.append({
                        "panel_file": p["file_name"],
                        "dramatic_score": 7,
                        "narration": _contextual_vietnamese_horror(len(selected_scenes) + 1, max_scenes, synopsis),
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

    def _generate_qwen_vl(
        self,
        panels: List[Dict],
        synopsis: str,
        min_score: int,
        max_scenes: int
    ) -> Dict:
        """
        Executes local Vision-LLM (Qwen2.5-VL-7B-Instruct) using 4-bit quantization on Colab GPU.
        Outputs 100% pure Vietnamese narration, completely eliminating English leaks and bracket emotion tags.
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
            if 'bitsandbytes' in str(e).lower() and 'quantization_config' in model_kwargs:
                print(f"[Script Generator] 4-bit load failed with bitsandbytes issue: {e}.")
                print("[Script Generator] Attempting fallback to torch.float16 directly without 4-bit...")
                try:
                    fallback_kwargs = {"device_map": "auto", "torch_dtype": torch.float16}
                    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **fallback_kwargs)
                    processor = AutoProcessor.from_pretrained(model_id)
                    print("[Script Generator] Successfully loaded with torch.float16!")
                except Exception as e2:
                    print(f"[Script Generator] Both 4-bit and float16 loading failed ({e2}). Please run: pip install -U 'bitsandbytes>=0.46.1'")
                    return self._generate_simulation(panels, synopsis, min_score, max_scenes, note=f"simulation ({e})")
            else:
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
        previous_narration = ""

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

                # Dynamic storytelling prompt providing context continuity
                if previous_narration:
                    context_instruction = (
                        f"Diễn biến cảnh vừa rồi bạn đã kể: '{previous_narration}'.\n"
                        "Nhiệm vụ cho cảnh này:\n"
                        "1. Quan sát hình ảnh và viết câu diễn biến TIẾP THEO để câu chuyện phát triển liền mạch.\n"
                        "2. BẮT BUỘC 100% TIẾNG VIỆT. TUYỆT ĐỐI KHÔNG VIẾT TIẾNG ANH hay tiếng Trung.\n"
                        "3. TUYỆT ĐỐI KHÔNG sử dụng bất kỳ thẻ biểu cảm nào trong ngoặc vuông như [thở dài], [tiếng thở dốc] hay [hắng giọng].\n"
                        "4. TUYỆT ĐỐI KHÔNG lặp lại việc miêu tả ngoại hình, quần áo, thời tiết mưa nếu cảnh trước đã đề cập.\n"
                        "5. Tập trung vào: hành động mới của nhân vật, nỗi sợ hãi gia tăng, hoặc điều kinh hoàng mới xuất hiện."
                    )
                else:
                    context_instruction = (
                        "Nhiệm vụ: Đây là cảnh mở đầu. Hãy bắt đầu câu chuyện một cách hồi hộp, rùng rợn và kích thích trí tò mò.\n"
                        "- BẮT BUỘC 100% TIẾNG VIỆT HOÀN TOÀN.\n"
                        "- TUYỆT ĐỐI KHÔNG sử dụng bất kỳ thẻ biểu cảm nào như [thở dài] hay dấu ngoặc vuông. Hãy dùng câu dẫn chuyện điện ảnh thuần túy."
                    )

                system_prompt = (
                    "Bạn là biên kịch kiêm đạo diễn video recap truyện tranh kinh dị giật gân (phong cách Quán Khuya).\n"
                    "QUY TẮC BẮT BUỘC SỐNG CÒN:\n"
                    "1. NGÔN NGỮ: 100% TIẾNG VIỆT HOÀN TOÀN. Mặc dù các khung tranh truyện tranh có thể có chữ tiếng Anh hoặc tiếng Nhật, bạn TUYỆT ĐỐI KHÔNG ĐƯỢC viết lời dẫn (narration) bằng tiếng Anh. Mọi suy nghĩ, đối thoại và hành động trong tranh đều PHẢI được chuyển tải thành lời kể chuyện tiếng Việt rùng rợn, lôi cuốn.\n"
                    "2. TUYỆT ĐỐI KHÔNG DÙNG THẺ BIỂU CẢM TRONG NGOẶC VUÔNG (như [thở dài], [tiếng thở dốc], [hắng giọng]). Chỉ viết câu dẫn chuyện điện ảnh thuần túy.\n"
                    "3. CHỈ TRẢ VỀ DUY NHẤT một chuỗi JSON hợp lệ không kèm bất kỳ giải thích bên ngoài nào."
                )

                user_prompt = (
                    f"Bối cảnh tổng thể: {synopsis or 'Không khí căng thẳng, u ám, bí ẩn kinh dị'}.\n"
                    f"{context_instruction}\n"
                    "Nhiệm vụ: Quan sát khung tranh và viết lời dẫn chuyện 'narration' BẰNG TIẾNG VIỆT.\n"
                    "LƯU Ý QUAN TRỌNG: TUYỆT ĐỐI KHÔNG VIẾT TIẾNG ANH (DO NOT USE ENGLISH). Dù tranh có chữ tiếng Anh, trường 'narration' BẮT BUỘC phải là 100% tiếng Việt.\n"
                    "Cấu trúc JSON bắt buộc:\n"
                    "{\n"
                    '  "dramatic_score": <số nguyên từ 1 đến 10>,\n'
                    '  "narration": "<câu kể chuyện tiếng Việt rùng rợn thuần túy, tuyệt đối không có tiếng Anh>",\n'
                    '  "camera_motion": "<zoom_in|zoom_out|pan_left|pan_right>",\n'
                    '  "sfx_cue": "<heart_beat|door_creak|creepy_whisper|jumpscare|none>"\n'
                    "}"
                )

                messages = [
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": system_prompt}
                        ]
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": user_prompt}
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

                # Sampling parameters with repetition penalty to eliminate loop repetition
                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=320,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.85,
                        repetition_penalty=1.2
                    )
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
                        pass

                # If JSON was truncated or malformed, extract narration field cleanly
                if not narration:
                    narr_match = re.search(r'["\']narration["\']\s*:\s*["\']([^"\'\n\r]+)', cleaned)
                    if narr_match:
                        narration = narr_match.group(1).strip()
                    else:
                        narration = cleaned

                # Sanitize Vietnamese text: strip all brackets, JSON leaks, and duplicate words
                narration = _sanitize_vietnamese_text(narration)

                # English leak detector & automatic Vietnamese fallback
                if _is_english_text(narration):
                    vi_replacement = _contextual_vietnamese_horror(idx, len(candidate_panels), synopsis)
                    print(f"  [Qwen-VL] ⚠️ English leak detected in scene {idx} ('{narration[:35]}...'). Replaced with pure Vietnamese narration.")
                    narration = vi_replacement

                if not narration or len(narration) < 10:
                    narration = _contextual_vietnamese_horror(idx, len(candidate_panels), synopsis)

                if score >= min_score or len(selected_scenes) < 5:
                    selected_scenes.append({
                        "panel_file": p["file_name"],
                        "dramatic_score": score,
                        "narration": narration,
                        "camera_motion": motion,
                        "sfx_cue": sfx
                    })
                    previous_narration = narration
                    print(f"  [Qwen-VL] Scene {len(selected_scenes)}/{max_scenes} (Score {score}): {narration[:65]}...")

                if len(selected_scenes) >= max_scenes:
                    break

            except Exception as e:
                print(f"  [Qwen-VL] Error processing panel {p.get('file_name', '')}: {e}")
                selected_scenes.append({
                    "panel_file": p["file_name"],
                    "dramatic_score": 7,
                    "narration": _contextual_vietnamese_horror(idx, len(candidate_panels), synopsis),
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
