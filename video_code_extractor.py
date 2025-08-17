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
import cv2
import argparse
import subprocess
import shutil
from pathlib import Path
from PIL import Image
import pytesseract
import imagehash
import shutil
from tqdm import tqdm
from google import genai
from google.genai import types
from yt_dlp import YoutubeDL
import numpy as np



def download_youtube_video(url: str, outdir: str = "./output") -> str:
    """
    Downloads a YouTube video in the best quality available as .mp4
    Returns the path of the downloaded video file.
    """
    # Check if ffmpeg is in the system's PATH
    if not shutil.which("ffmpeg"):
        print("Warning: ffmpeg is not found in your system's PATH. This is required to merge best quality video and audio. The script will continue but may download a lower-quality video.")
    
    os.makedirs(outdir, exist_ok=True)

    ydl_opts = {
        "format": "bestvideo+bestaudio/best",  # get best video + audio merged
        "merge_output_format": "mp4",          # always output .mp4
        "outtmpl": os.path.join(outdir, "video.%(ext)s"),
        "noplaylist": True,                    # only one video, no playlists
        "quiet": False,                        # show progress
        "ignoreerrors": True
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        if info is None:
            raise RuntimeError("Failed to download video.")

        # Get final file path
        return ydl.prepare_filename(info) #.replace(".webm", ".mp4").replace(".mkv", ".mp4")


def extract_frames(video_path: str, outdir: str, fps: int = 1) -> str:
    """
    Extracts frames from a video and saves them in a 'frames' folder inside outdir.
    :param video_path: Path to the video file (.mp4)
    :param outdir: Base output directory
    :param fps: Frames per second to extract
    :return: Path to frames folder
    """
    frames_dir = os.path.join(outdir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(round(video_fps / fps))  # how many frames to skip

    count, saved = 0, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_interval == 0:
            frame_path = os.path.join(frames_dir, f"frame_{saved:05d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved += 1

        count += 1

    cap.release()
    print(f"Extracted {saved} frames to {frames_dir}")
    return frames_dir

# A more robust version of the filter function that checks for VS Code-like UI elements in frames:
# def filter_vs_code_frames(frames_dir, output_dir, keywords=None):
#     """
#     Filters frames to keep only those containing a VS Code-like interface
#     by checking a combination of header text, file explorer content, and a unique color profile.
    
#     Args:
#         frames_dir (str | Path): The directory containing the extracted frames.
#         output_dir (str | Path): The directory to save the filtered frames.
#         keywords (list[str]): Keywords to look for in OCR text.
#     """
#     frames_path = Path(frames_dir)
#     output_path = Path(output_dir)
#     # Clear the output directory to avoid old files
#     if output_path.exists():
#         shutil.rmtree(output_path)
#     output_path.mkdir(parents=True, exist_ok=True)
    
#     if keywords is None:
#         # A comprehensive list including header menus and file explorer items
#         keywords = [
#             "File", "Edit", "Selection", "View", "Go", "Run", "Help",
#             "src", "public", "node_modules", "package.json", "package-lock.json",
#             "App.css", "App.js", "App.jsx", "index.css", "index.html",
#             "components", "assets", "public", ".gitignore", "vite.config.js"
#         ]

#     kept, removed = 0, 0
#     total = len(list(frames_path.glob("*.jpg")))
    
#     print(f"Starting to filter {total} frames. This may take a while.")

#     for frame_path in frames_path.glob("*.jpg"):
#         img = cv2.imread(str(frame_path))
#         if img is None:
#             continue

#         # Check 1: OCR for header and sidebar keywords
#         roi_left = img[:, :int(img.shape[1] * 0.25)] # Left 25% for sidebar
#         roi_top = img[0:int(img.shape[0] * 0.05), :] # Top 5% for header
        
#         ocr_result_left = pytesseract.image_to_string(roi_left)
#         ocr_result_top = pytesseract.image_to_string(roi_top)
        
#         text_found = any(kw.lower() in ocr_result_left.lower() or kw.lower() in ocr_result_top.lower() for kw in keywords)

#         # Check 2: Color analysis for a dark theme signature
#         h, w, _ = img.shape
#         # Sample a few key regions
#         samples = [
#             img[h-20:h, w-100:w],  # Bottom right status bar
#             img[100:150, 0:50]     # Top left activity bar
#         ]
        
#         # Check for dark pixel percentages
#         is_dark_theme = False
#         for sample in samples:
#             if sample.size > 0:
#                 gray_sample = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
#                 dark_pixels = np.sum(gray_sample < 50) # Very dark pixels
#                 if dark_pixels > sample.size * 0.5:
#                     is_dark_theme = True
#                     break

#         # Check 3: Look for a file explorer-like structure (lines and spacing)
#         # This is more advanced and can be added later if needed.
#         # For now, the first two checks are sufficient and fast.
        
#         # Keep the frame if either of the main checks pass
#         if text_found or is_dark_theme:
#             save_path = output_path / frame_path.name
#             cv2.imwrite(str(save_path), img)
#             kept += 1
#         else:
#             removed += 1
            
#     print(f"✅ Filtering complete. Total frames processed: {total}. Kept: {kept}, Removed: {removed}")




def filter_vs_code_frames(frames_dir, output_dir, keywords=None):
    """
    Filters frames to keep only those containing VS Code UI elements
    by checking for a combination of header text and a unique color-based icon.
    
    Args:
        frames_dir (str | Path): Directory with extracted frames.
        output_dir (str | Path): Directory to save filtered frames.
        keywords (list[str]): Keywords to look for in OCR text.
    """
    frames_dir = Path(frames_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if keywords is None:
        # A comprehensive list including header menus and file explorer items
        keywords = [
            "File", "Edit", "Selection", "View", "Go", "Run", "Help",
            "src", "public", "node_modules", "package.json", "App.js", "App.jsx",
            "index.css", ".gitignore", "vite.config.js"
        ]

    kept, removed = 0, 0
    
    for frame_path in frames_dir.glob("*.jpg"):
        img = cv2.imread(str(frame_path))
        if img is None:
            continue

        # Check 1: OCR for header keywords
        # This ROI targets the top menu bar
        roi_header = img[0:int(img.shape[0] * 0.05), :]
        ocr_result_header = pytesseract.image_to_string(roi_header)
        
        header_found = any(kw.lower() in ocr_result_header.lower() for kw in keywords)
        
        # Check 2: Look for the unique explorer icon color
        # This ROI targets the activity bar, where the explorer icon is located
        roi_icon = img[int(img.shape[0] * 0.1):int(img.shape[0] * 0.15), 0:int(img.shape[1] * 0.05)]
        
        # Define a common VS Code dark theme icon color (a type of teal/blue)
        # BGR format
        lower_blue = np.array([120, 50, 20])
        upper_blue = np.array([255, 100, 50])
        
        # Create a mask for the color
        mask = cv2.inRange(roi_icon, lower_blue, upper_blue)
        
        # Check if a significant portion of the ROI matches the color
        icon_found = np.count_nonzero(mask) > 100 # a small threshold
        
        # If either the header text or the icon is found, keep the frame
        if header_found or icon_found:
            save_path = output_dir / frame_path.name
            cv2.imwrite(str(save_path), img)
            kept += 1
        else:
            removed += 1
            
    print(f"✅ Filtering complete. Kept: {kept}, Removed: {removed}")


def clean_vs_code_frames(frames_dir, output_dir, min_conf=50):
    """
    Filters VS Code frames:
    1. Keeps only frames with a file open (tab detected).
    2. Removes frames where first 5 lines of code area are empty.
    
    Args:
        frames_dir (str): Input folder with filtered VS Code frames.
        output_dir (str): Output folder for cleaned frames.
        min_conf (int): OCR confidence (default=50).
    """
    frames_dir = Path(frames_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    kept, removed = 0, 0

    for frame_path in frames_dir.glob("*.jpg"):
        img = cv2.imread(str(frame_path))
        if img is None:
            continue

        h, w, _ = img.shape

        # ------------------------
        # Step 1: Check for open file tab (top bar area ~ 10-15% height)
        # ------------------------
        tab_area = img[0:int(h*0.15), 0:w]   # top 15%
        tab_text = pytesseract.image_to_string(tab_area)

        if not any(ext in tab_text for ext in [".py", ".js", ".java", ".cpp", ".md", ".txt"]):
            removed += 1
            continue  # no file open, discard

        # ------------------------
        # Step 2: Check code area (middle ~ 70% of screen)
        # ------------------------
        code_area = img[int(h*0.15):int(h*0.85), 0:w]   # skip tabs & bottom panels
        code_text = pytesseract.image_to_string(code_area)

        # Split lines, take first 5
        code_lines = [line.strip() for line in code_text.split("\n") if line.strip() != ""]

        if len(code_lines) < 5:  
            removed += 1
            continue  # not enough visible code, discard

        # ------------------------
        # Step 3: Keep frame
        # ------------------------
        save_path = output_dir / frame_path.name
        cv2.imwrite(str(save_path), img)
        kept += 1

    print(f"✅ Cleaning complete. Kept: {kept}, Removed: {removed}")


if __name__ == "__main__":
    video_url = "https://youtu.be/wFh0SJVDM9E"
    outdir = "./output"
    # video_path = "./output/Complete React Portfolio Website Project Tutorial - Create Personal Portfolio Website with React JS.mp4"
    # video_path = download_youtube_video(video_url, outdir)
    # print("Downloaded video saved at:", video_path)
    # frames_folder = extract_frames(video_path, outdir, fps=1)
    # print("Frames saved in:", frames_folder)
    # filter_vs_code_frames("./output/frames", "./output/vscode_frames")
    clean_vs_code_frames("./output/vscode_frames", "./output/clean_frames")





# # ----------------------------- Global Fallbacks -----------------------------
# # Set your fallback executable paths here if not in PATH
# YTDLP_FALLBACK = r"C:\Users\DELL\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\Scripts\yt-dlp.exe"
# FFMPEG_FALLBACK = r"C:\ffmpeg\bin\ffmpeg.exe"  # <-- Update to your actual ffmpeg.exe path
# TESSERACT_FALLBACK = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # <-- Update if different

# # ----------------------------- Helpers -----------------------------

# def run_cmd(cmd):
#     res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
#     if res.returncode != 0:
#         raise RuntimeError(f"Command failed: {' '.join(map(str, cmd))}\n{res.stderr}")
#     return res.stdout


# def ensure_prog_exists(name):
#     """
#     Ensure a program exists, with fallbacks for yt-dlp, ffmpeg, tesseract.
#     Returns the executable path to use.
#     """
#     exe_path = shutil.which(name)
#     if exe_path:
#         return exe_path

#     if name.lower() == "yt-dlp" and Path(YTDLP_FALLBACK).is_file():
#         return YTDLP_FALLBACK
#     if name.lower() == "ffmpeg" and Path(FFMPEG_FALLBACK).is_file():
#         return FFMPEG_FALLBACK
#     if name.lower() == "tesseract" and Path(TESSERACT_FALLBACK).is_file():
#         return TESSERACT_FALLBACK

#     raise RuntimeError(f"Required binary '{name}' not found in PATH or fallback location.")


# # ----------------------------- Pipeline -----------------------------

# def download_video(url, outdir):
#     yt_dlp_path = ensure_prog_exists("yt-dlp")
#     Path(outdir).mkdir(parents=True, exist_ok=True)
#     target = str(Path(outdir) / "video.%(ext)s")
#     run_cmd([
#         yt_dlp_path,
#         "-f", "bestvideo+bestaudio",
#         "--merge-output-format", "mp4",
#         "--no-keep-video",  # remove temp files
#         "-o", target,
#         url
#     ])
#     mp4s = sorted(Path(outdir).glob("*.mp4"), key=lambda p: p.stat().st_size, reverse=True)
#     if not mp4s:
#         raise RuntimeError("No mp4 downloaded.")
#     return str(mp4s[0])


# def extract_frames(video_path, frames_dir, fps=1):
#     ffmpeg_path = ensure_prog_exists("ffmpeg")
#     Path(frames_dir).mkdir(parents=True, exist_ok=True)
#     run_cmd([
#         ffmpeg_path,
#         "-hide_banner", "-loglevel", "error",
#         "-i", video_path,
#         "-vf", f"fps={fps}",
#         str(Path(frames_dir) / "frame_%06d.png")
#     ])
#     return sorted(Path(frames_dir).glob("*.png"))


# def crop_image(in_path, out_path, roi):
#     im = Image.open(in_path)
#     x, y, w, h = roi
#     im.crop((x, y, x + w, y + h)).save(out_path)


# def deduplicate_frames(frames, threshold=5):
#     kept, hashes = [], []
#     for f in tqdm(frames, desc="Deduplicating"):
#         h = imagehash.dhash(Image.open(f).convert("RGB"))
#         if not any(h - eh <= threshold for eh in hashes):
#             hashes.append(h)
#             kept.append(f)
#     return kept


# def ocr_frames(frames, lang='eng', psm=6):
#     tesseract_path = ensure_prog_exists("tesseract")
#     pytesseract.pytesseract.tesseract_cmd = tesseract_path
#     all_text = {}
#     config = f"--psm {psm} -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_[](){{}}<>:;.,+-=*/\\\"'#%&|@!?^~ \t"
#     for f in tqdm(frames, desc="OCR"):
#         text = pytesseract.image_to_string(Image.open(f), lang=lang, config=config)
#         all_text[str(f)] = text
#     return all_text


# def merge_ocr(all_text, out_file):
#     items = sorted(all_text.items(), key=lambda kv: kv[0])
#     merged = []
#     for fname, text in items:
#         if len(text.strip()) > 2:
#             merged.append(text.strip() + "\n")
#     combined = "\n".join(merged)
#     Path(out_file).write_text(combined, encoding="utf8")
#     return combined


# # ----------------------------- Gemini Integration -----------------------------

# def send_to_generative_model(code_text, model="gemini-2.5-flash", api_key=None):
#     client = genai.Client(api_key=api_key)
#     tool = types.Tool(code_execution=types.ToolCodeExecution())
#     config = types.GenerateContentConfig(tools=[tool], temperature=0)
#     prompt = (
#         "You are an expert software engineer. The text below is OCR output from a coding video.\n"
#         "OCR may cause syntax errors, wrong characters (0/O, 1/l/I, ;/:), or bad indentation.\n"
#         "Fix the code so it is correct and functional. Preserve logic.\n"
#         "Return only the corrected code.\n\n"
#         f"{code_text}"
#     )
#     resp = client.models.generate_content(model=model, contents=prompt, config=config)
#     fixed = ""
#     for part in resp.candidates[0].content.parts:
#         if part.executable_code:
#             fixed += part.executable_code.code + "\n"
#         elif part.text:
#             fixed += part.text + "\n"
#     return fixed


# # ----------------------------- Main -----------------------------

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--url", required=True, help="YouTube URL")
#     ap.add_argument("--outdir", default="./output", help="Output folder")
#     ap.add_argument("--fps", type=float, default=1.0, help="Frames per second to capture")
#     ap.add_argument("--roi", default=None, help="Crop region x,y,w,h")
#     ap.add_argument("--dedup_threshold", type=int, default=5, help="Image hash threshold for deduplication")
#     ap.add_argument("--use_gemini", action="store_true", help="Auto-fix with Gemini")
#     args = ap.parse_args()

#     outdir = Path(args.outdir)
#     outdir.mkdir(parents=True, exist_ok=True)

#     print("[1] Downloading video...")
#     video_path = download_video(args.url, outdir)

#     print("[2] Extracting frames...")
#     frames = extract_frames(video_path, outdir / "frames", fps=args.fps)

#     if args.roi:
#         roi = tuple(int(x) for x in args.roi.split(","))
#         cropped_dir = outdir / "frames_cropped"
#         cropped_dir.mkdir(exist_ok=True)
#         for f in tqdm(frames, desc="Cropping"):
#             crop_image(f, cropped_dir / Path(f).name, roi)
#         frames = sorted(cropped_dir.glob("*.png"))

#     print("[3] Deduplicating frames...")
#     frames = deduplicate_frames(frames, threshold=args.dedup_threshold)

#     print("[4] Running OCR...")
#     ocr_texts = ocr_frames(frames)

#     print("[5] Merging OCR outputs...")
#     merged_text = merge_ocr(ocr_texts, outdir / "raw_ocr.txt")

#     if args.use_gemini:
#         print("[6] Sending to Gemini for fixing...")
#         api_key = os.environ.get("GEMINI_API_KEY")
#         if not api_key:
#             raise RuntimeError("Please set GEMINI_API_KEY environment variable.")
#         fixed_code = send_to_generative_model(merged_text, api_key=api_key)
#         Path(outdir / "fixed_by_gemini.txt").write_text(fixed_code, encoding="utf8")
#         print("Gemini-fixed code saved to fixed_by_gemini.txt")

#     print("Done. See output folder.")


# if __name__ == "__main__":
#     main()
