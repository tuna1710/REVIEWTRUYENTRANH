"""
Module 1: Manga Downloader & Ingestion
Handles downloading chapters from online manga URLs or extracting from local archive files (.zip, .cbz, .pdf).
"""

import os
import shutil
import zipfile
import subprocess
from pathlib import Path
from typing import List, Optional
import requests

try:
    import cloudscraper
    HAS_CLOUDSCRAPER = True
except ImportError:
    HAS_CLOUDSCRAPER = False


class MangaDownloader:
    def __init__(self, output_dir: str = "./workspace/raw_pages"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_from_url(self, url: str) -> List[str]:
        """
        Downloads manga images from a URL using gallery-dl or fallback scraper.
        Returns a sorted list of file paths.
        """
        print(f"[Downloader] Downloading chapter from: {url}")
        
        # 1. Try gallery-dl first (supports 1000+ comic/manga websites)
        try:
            cmd = [
                "gallery-dl",
                "--dest", str(self.output_dir),
                "--filename", "page_{num:03d}.{extension}",
                url
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                pages = self.get_sorted_pages()
                if pages:
                    print(f"[Downloader] Successfully fetched {len(pages)} pages via gallery-dl.")
                    return pages
        except Exception as e:
            print(f"[Downloader] gallery-dl notice: {e}, falling back to custom scraper...")

        # 2. Fallback scraper for custom websites
        pages = self._fallback_scraper(url)
        return pages

    def load_from_local_archive(self, file_path: str) -> List[str]:
        """
        Extracts images from a .zip, .cbz, .pdf, or directory.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        print(f"[Downloader] Processing local source: {file_path}")

        # If it's a directory, copy and standardize images
        if path.is_dir():
            image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
            raw_files = sorted([f for f in path.iterdir() if f.suffix.lower() in image_extensions])
            for idx, f in enumerate(raw_files, start=1):
                dest = self.output_dir / f"page_{idx:03d}{f.suffix.lower()}"
                shutil.copy(f, dest)
            return self.get_sorted_pages()

        # If it's a PDF document
        if path.suffix.lower() == ".pdf":
            return self._extract_pdf(path)

        # If it's a zip or cbz archive
        if path.suffix.lower() in [".zip", ".cbz"]:
            temp_extract = self.output_dir / "temp_extract"
            temp_extract.mkdir(exist_ok=True)
            with zipfile.ZipFile(path, 'r') as zip_ref:
                zip_ref.extractall(temp_extract)

            # Collect extracted images and move to output_dir
            image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
            extracted = sorted([
                p for p in temp_extract.rglob("*")
                if p.is_file() and p.suffix.lower() in image_extensions
            ])
            for idx, img_path in enumerate(extracted, start=1):
                dest = self.output_dir / f"page_{idx:03d}{img_path.suffix.lower()}"
                shutil.move(str(img_path), str(dest))

            shutil.rmtree(temp_extract, ignore_errors=True)
            return self.get_sorted_pages()

        raise ValueError(f"Unsupported local format: {path.suffix}. Must be .zip, .cbz, .pdf, or directory.")

    def _extract_pdf(self, pdf_path: Path) -> List[str]:
        """
        Extracts high-resolution page images from a PDF file.
        Uses pypdfium2 if available, or PyMuPDF (fitz) as fallback.
        """
        try:
            import pypdfium2 as pdfium
            pdf = pdfium.PdfDocument(str(pdf_path))
            print(f"[Downloader] Extracting {len(pdf)} pages from PDF using pypdfium2...")
            for idx, page in enumerate(pdf, start=1):
                dest = self.output_dir / f"page_{idx:03d}.png"
                # Render at 2x scale for crisp manga panels
                img = page.render(scale=2.0).to_pil()
                img.save(str(dest), "PNG")
            return self.get_sorted_pages()
        except ImportError:
            pass

        try:
            import fitz
            doc = fitz.open(str(pdf_path))
            print(f"[Downloader] Extracting {len(doc)} pages from PDF using PyMuPDF...")
            for idx, page in enumerate(doc, start=1):
                dest = self.output_dir / f"page_{idx:03d}.png"
                pix = page.get_pixmap(dpi=150)
                pix.save(str(dest))
            return self.get_sorted_pages()
        except ImportError:
            raise ImportError(
                "PDF support requires 'pypdfium2' or 'PyMuPDF'. Please install: pip install pypdfium2 pillow"
            )

    def _fallback_scraper(self, url: str) -> List[str]:
        """
        Lightweight fallback scraper using cloudscraper/requests.
        """
        scraper = cloudscraper.create_scraper() if HAS_CLOUDSCRAPER else requests.Session()
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        res = scraper.get(url, headers=headers)
        if res.status_code != 200:
            raise RuntimeError(f"Failed to fetch webpage (status {res.status_code})")

        # Basic HTML regex to find manga image tags
        import re
        img_urls = re.findall(r'<img[^>]+(?:src|data-src|data-original)=["\'](https?://[^"\']+\.(?:jpg|jpeg|png|webp))["\']', res.text, re.IGNORECASE)
        img_urls = list(dict.fromkeys(img_urls)) # Remove duplicates while preserving order

        if not img_urls:
            raise RuntimeError("No comic image elements discovered on page.")

        print(f"[Downloader] Discovered {len(img_urls)} comic page links. Downloading...")
        saved_paths = []
        for idx, img_url in enumerate(img_urls, start=1):
            ext = img_url.split(".")[-1].split("?")[0].lower()
            if ext not in ["jpg", "jpeg", "png", "webp"]:
                ext = "jpg"
            dest = self.output_dir / f"page_{idx:03d}.{ext}"
            img_res = scraper.get(img_url, headers=headers)
            if img_res.status_code == 200:
                with open(dest, "wb") as f:
                    f.write(img_res.content)
                saved_paths.append(str(dest))

        return sorted(saved_paths)

    def get_sorted_pages(self) -> List[str]:
        """
        Returns all valid page images sorted in natural order.
        """
        exts = {".jpg", ".jpeg", ".png", ".webp"}
        files = [str(f) for f in self.output_dir.iterdir() if f.is_file() and f.suffix.lower() in exts]
        return sorted(files)
