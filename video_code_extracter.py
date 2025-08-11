#!/usr/bin/env python3
"""
video_code_extractor.py

Automated pipeline to extract code from a YouTube video using OCR
and auto-fix it with the free Google Gemini API (via AI Studio).

Features:
- Downloads a YouTube video (yt-dlp)
- Extracts frames (ffmpeg)
- Optional ROI crop (focus on code area)
- Deduplicates similar frames (imagehash)
- Runs OCR on frames (pytesseract)
- Merges OCR text into raw_ocr.txt
- Sends OCR output to Gemini 2.5 Flash (free tier) for fixing

Requirements:
    pip install pillow pytesseract imagehash requests tqdm google-genai

System dependencies:
    yt-dlp (https://github.com/yt-dlp/yt-dlp)
    ffmpeg (https://ffmpeg.org/)
    Tesseract OCR (https://github.com/tesseract-ocr/tesseract)

Usage:
    export GEMINI_API_KEY="your_google_ai_studio_key_here"
    python3 video_code_extractor.py --url "YOUTUBE_URL" --outdir ./output --fps 1 --use_gemini
"""

import os
import argparse
import subprocess
import shutil
from pathlib import Path
from PIL import Image
import pytesseract
import imagehash
from tqdm import tqdm
from google import genai
from google.genai import types


# ----------------------------- Helpers -----------------------------

def run_cmd(cmd):
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{res.stderr}")
    return res.stdout


def ensure_prog_exists(name):
    if shutil.which(name) is None:
        raise RuntimeError(f"Required binary '{name}' not found in PATH.")


# ----------------------------- Pipeline -----------------------------

def download_video(url, outdir):
    ensure_prog_exists("yt-dlp")
    Path(outdir).mkdir(parents=True, exist_ok=True)
    target = str(Path(outdir) / "video.%(ext)s")
    run_cmd(["yt-dlp", "-f", "bestvideo+bestaudio", "--merge-output-format", "mp4", "-o", target, url])
    mp4s = sorted(Path(outdir).glob("*.mp4"), key=lambda p: p.stat().st_size, reverse=True)
    if not mp4s:
        raise RuntimeError("No mp4 downloaded.")
    return str(mp4s[0])


def extract_frames(video_path, frames_dir, fps=1):
    ensure_prog_exists("ffmpeg")
    Path(frames_dir).mkdir(parents=True, exist_ok=True)
    run_cmd(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", video_path, "-vf", f"fps={fps}", str(Path(frames_dir) / "frame_%06d.png")])
    return sorted(Path(frames_dir).glob("*.png"))


def crop_image(in_path, out_path, roi):
    im = Image.open(in_path)
    x, y, w, h = roi
    im.crop((x, y, x + w, y + h)).save(out_path)


def deduplicate_frames(frames, threshold=5):
    kept, hashes = [], []
    for f in tqdm(frames, desc="Deduplicating"):
        h = imagehash.dhash(Image.open(f).convert("RGB"))
        if not any(h - eh <= threshold for eh in hashes):
            hashes.append(h)
            kept.append(f)
    return kept


def ocr_frames(frames, lang='eng', psm=6):
    all_text = {}
    config = f"--psm {psm} -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_[](){{}}<>:;.,+-=*/\\\"'#%&|@!?^~ \t"
    for f in tqdm(frames, desc="OCR"):
        text = pytesseract.image_to_string(Image.open(f), lang=lang, config=config)
        all_text[str(f)] = text
    return all_text


def merge_ocr(all_text, out_file):
    items = sorted(all_text.items(), key=lambda kv: kv[0])
    merged = []
    for fname, text in items:
        if len(text.strip()) > 2:
            merged.append(text.strip() + "\n")
    combined = "\n".join(merged)
    Path(out_file).write_text(combined, encoding="utf8")
    return combined


# ----------------------------- Gemini Integration -----------------------------

def send_to_generative_model(code_text, model="gemini-2.5-flash", api_key=None):
    client = genai.Client(api_key=api_key)
    tool = types.Tool(code_execution=types.ToolCodeExecution())
    config = types.GenerateContentConfig(tools=[tool], temperature=0)
    prompt = (
        "You are an expert software engineer. The text below is OCR output from a coding video.\n"
        "OCR may cause syntax errors, wrong characters (0/O, 1/l/I, ;/:), or bad indentation.\n"
        "Fix the code so it is correct and functional. Preserve logic.\n"
        "Return only the corrected code.\n\n"
        f"{code_text}"
    )
    resp = client.models.generate_content(model=model, contents=prompt, config=config)
    fixed = ""
    for part in resp.candidates[0].content.parts:
        if part.executable_code:
            fixed += part.executable_code.code + "\n"
        elif part.text:
            fixed += part.text + "\n"
    return fixed


# ----------------------------- Main -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="YouTube URL")
    ap.add_argument("--outdir", default="./output", help="Output folder")
    ap.add_argument("--fps", type=float, default=1.0, help="Frames per second to capture")
    ap.add_argument("--roi", default=None, help="Crop region x,y,w,h")
    ap.add_argument("--dedup_threshold", type=int, default=5, help="Image hash threshold for deduplication")
    ap.add_argument("--use_gemini", action="store_true", help="Auto-fix with Gemini")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("[1] Downloading video...")
    video_path = download_video(args.url, outdir)

    print("[2] Extracting frames...")
    frames = extract_frames(video_path, outdir / "frames", fps=args.fps)

    if args.roi:
        roi = tuple(int(x) for x in args.roi.split(","))
        cropped_dir = outdir / "frames_cropped"
        cropped_dir.mkdir(exist_ok=True)
        for f in tqdm(frames, desc="Cropping"):
            crop_image(f, cropped_dir / Path(f).name, roi)
        frames = sorted(cropped_dir.glob("*.png"))

    print("[3] Deduplicating frames...")
    frames = deduplicate_frames(frames, threshold=args.dedup_threshold)

    print("[4] Running OCR...")
    ocr_texts = ocr_frames(frames)

    print("[5] Merging OCR outputs...")
    merged_text = merge_ocr(ocr_texts, outdir / "raw_ocr.txt")

    if args.use_gemini:
        print("[6] Sending to Gemini for fixing...")
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Please set GEMINI_API_KEY environment variable.")
        fixed_code = send_to_generative_model(merged_text, api_key=api_key)
        Path(outdir / "fixed_by_gemini.txt").write_text(fixed_code, encoding="utf8")
        print("Gemini-fixed code saved to fixed_by_gemini.txt")

    print("Done. See output folder.")


if __name__ == "__main__":
    main()
