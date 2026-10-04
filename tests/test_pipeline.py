"""
Automated Test Suite for Manga Recap AI Studio
Executes complete pipeline on real manga test case: assets/mieruko_chan_ch01.cbz
"""

import os
import sys
import unittest
from pathlib import Path

# Fix httpx NO_PROXY bug with IPv6 brackets
for _k in ["NO_PROXY", "no_proxy"]:
    if _k in os.environ and "::1" in os.environ[_k]:
        os.environ[_k] = os.environ[_k].replace(",[::1]", "").replace(",::1", "").replace("[::1]", "").replace("::1", "")

BASE_DIR = Path(__file__).parent.parent.resolve()
sys.path.append(str(BASE_DIR))

from src.downloader import MangaDownloader
from src.panel_extractor import MangaPanelExtractor
from src.script_generator import ScriptGenerator
from src.tts_engine import TTSEngine
from src.subtitle_generator import SubtitleGenerator
from src.video_engine import VideoEngine
from src.capcut_exporter import CapCutExporter


class TestMangaRecapPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_cbz = BASE_DIR / "assets" / "mieruko_chan_ch01.cbz"
        cls.output_dir = BASE_DIR / "workspace" / "test_run"
        cls.output_dir.mkdir(parents=True, exist_ok=True)
        assert cls.test_cbz.exists(), f"Test manga archive missing at {cls.test_cbz}"

    def test_01_downloader_cbz_ingestion(self):
        """Step 1: Test loading and extracting manga pages from CBZ archive"""
        raw_pages_dir = self.output_dir / "raw_pages"
        downloader = MangaDownloader(output_dir=str(raw_pages_dir))
        pages = downloader.load_from_local_archive(str(self.test_cbz))

        self.assertGreaterEqual(len(pages), 16, "Mieruko-chan Ch1 should have at least 16 pages")
        for p in pages:
            self.assertTrue(os.path.exists(p), f"Page file does not exist: {p}")

    def test_02_panel_extractor_rtl(self):
        """Step 2: Test YOLO/contour RTL panel extraction on real manga pages"""
        raw_pages_dir = self.output_dir / "raw_pages"
        pages = sorted([str(p) for p in raw_pages_dir.glob("*.jpg") or raw_pages_dir.glob("*.png")])
        self.assertTrue(len(pages) > 0, "No raw pages to extract panels from")

        panels_dir = self.output_dir / "panels"
        extractor = MangaPanelExtractor(output_dir=str(panels_dir), reading_order="RTL")
        # Process first 2 pages for fast test execution
        panels_meta = extractor.process_all_pages(pages[:2])

        self.assertGreater(len(panels_meta), 0, "Should detect at least 1 panel from manga pages")
        for p in panels_meta:
            self.assertTrue(os.path.exists(p["file_path"]), f"Panel image missing: {p['file_path']}")
            self.assertIn("panel_id", p)

    def test_03_script_generation(self):
        """Step 3: Test AI / Simulation Script Generator (both Gemini and Qwen-VL)"""
        panels_dir = self.output_dir / "panels"
        panel_files = sorted(list(panels_dir.glob("*.png")))
        self.assertTrue(len(panel_files) > 0, "No panels available for script generation")

        panels_meta = [
            {"page_index": 1, "panel_index": idx, "file_name": pf.name, "file_path": str(pf)}
            for idx, pf in enumerate(panel_files[:3], 1)
        ]

        # 3a. Test Qwen2.5-VL Script Generator
        timeline_qwen_out = self.output_dir / "timeline_qwen.json"
        script_gen_qwen = ScriptGenerator(
            provider="qwen_vl",
            model_name="Qwen/Qwen2.5-VL-7B-Instruct",
            output_file=str(timeline_qwen_out)
        )
        timeline_qwen = script_gen_qwen.generate_timeline(
            panels_metadata=panels_meta,
            story_synopsis="Cô nữ sinh nhìn thấy quái vật kinh dị lúc nửa đêm",
            min_score=7,
            max_scenes=3
        )
        self.assertIn("scenes", timeline_qwen)
        self.assertGreater(len(timeline_qwen["scenes"]), 0)
        self.assertTrue(timeline_qwen_out.exists())

        # 3b. Test Gemini fallback
        timeline_out = self.output_dir / "timeline.json"
        script_gen = ScriptGenerator(
            provider="gemini",
            api_key="",  # Simulation fallback mode
            model_name="gemini-3.8-flash",
            output_file=str(timeline_out)
        )
        timeline = script_gen.generate_timeline(
            panels_metadata=panels_meta,
            story_synopsis="Cô nữ sinh nhìn thấy quái vật kinh dị lúc nửa đêm",
            min_score=7,
            max_scenes=3
        )

        self.assertIn("scenes", timeline)
        self.assertGreater(len(timeline["scenes"]), 0)
        self.assertTrue(timeline_out.exists())

    def test_04_tts_voice_resolution(self):
        """Step 4: Test TTS engine voice resolution and preview synthesis"""
        audio_dir = self.output_dir / "audio"
        tts = TTSEngine(voice_preset="Thiện Minh (Trầm ấm, rùng rợn - Chuẩn Quán Khuya)", output_dir=str(audio_dir))
        
        resolved_voice = tts._resolve_voice(tts.voice_preset)
        self.assertEqual(resolved_voice, "Thiện Minh")

        # Test preview audition
        sample_audio = tts.preview_voice("Đừng quay đầu lại, có thứ gì đó đang đứng sau lưng bạn...")
        self.assertTrue(os.path.exists(sample_audio), "Sample preview audio was not created")
        self.assertGreater(os.path.getsize(sample_audio), 1000)

    def test_05_subtitles_responsive(self):
        """Step 5: Test Faster-Whisper / ASS Subtitles with 9:16 safe margins"""
        sub_dir = self.output_dir / "subtitles"
        audio_dir = self.output_dir / "audio"
        preview_audio = str(audio_dir / "voice_sample_preview.wav")

        sub_gen_916 = SubtitleGenerator(output_dir=str(sub_dir), aspect_ratio="9:16")
        res = sub_gen_916.transcribe_and_generate_sub(
            scene_id=1,
            audio_path=preview_audio,
            reference_text="Một bóng đen xuất hiện trong đêm tối..."
        )
        self.assertTrue(os.path.exists(res["ass_path"]))
        with open(res["ass_path"], "r", encoding="utf-8") as f:
            ass_content = f.read()
        self.assertIn("PlayResX: 1080", ass_content)
        self.assertIn("PlayResY: 1920", ass_content)

    def test_06_video_scene_and_capcut_export(self):
        """Step 6: Test Video rendering and CapCut Project Draft Export"""
        panels_dir = self.output_dir / "panels"
        panel_file = next(panels_dir.glob("*.png"))
        audio_dir = self.output_dir / "audio"
        audio_file = audio_dir / "voice_sample_preview.wav"
        sub_dir = self.output_dir / "subtitles"
        ass_file = sub_dir / "scene_0001.ass"

        scenes_dir = self.output_dir / "rendered_scenes"
        video_engine = VideoEngine(
            output_dir=str(scenes_dir),
            final_output_path=str(self.output_dir / "final_test_video.mp4"),
            aspect_ratio="9:16"
        )
        scene_mp4 = video_engine.render_scene(
            scene_id=1,
            panel_path=str(panel_file),
            audio_path=str(audio_file),
            sub_ass_path=str(ass_file),
            camera_motion="zoom_in",
            duration=3.0
        )
        self.assertTrue(os.path.exists(scene_mp4), "Rendered scene MP4 missing")
        self.assertGreater(os.path.getsize(scene_mp4), 10000)

        # CapCut Draft Export
        capcut_dir = self.output_dir / "capcut_draft"
        capcut_exp = CapCutExporter(output_dir=str(capcut_dir))
        mock_timeline = {
            "title": "Mieruko-chan Test Recap",
            "scenes": [{"panel_file": panel_file.name, "narration": "Cảnh thử nghiệm kinh dị", "camera_motion": "zoom_in"}]
        }
        draft_root = capcut_exp.export_draft(
            timeline_data=mock_timeline,
            panels_dir=str(panels_dir),
            audio_dir=str(audio_dir),
            aspect_ratio="9:16"
        )
        self.assertTrue(os.path.exists(Path(draft_root) / "draft_content.json"))
        self.assertTrue(os.path.exists(Path(draft_root) / "draft_meta_info.json"))


if __name__ == "__main__":
    unittest.main()
