import os
import cv2
import pytesseract
import numpy as np
from pathlib import Path
import shutil
import imagehash
from PIL import Image

# Make sure pytesseract knows where to find the tesseract executable

def extract_code_frames(video_path: str, output_dir: str, keywords=None):
    """
    Extracts and filters unique VS Code frames from a video stream.
    This function is optimized for speed and avoids redundant processing.
    
    Args:
        video_path (str): Path to the video file (.mp4).
        output_dir (str): Directory to save the filtered frames.
        keywords (list[str]): Keywords to look for in OCR text.
    """
    output_path = Path(output_dir)
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if keywords is None:
        keywords = [
            "File", "Edit", "Selection", "View", "Go", "Run", "Help",
            "src", "public", "node_modules", "package.json", "package-lock.json",
            "App.css", "App.js", "App.jsx", "index.css", "index.html",
            "components", "assets", "public", ".gitignore", "vite.config.js"
        ]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    # Initialize a set to store hashes of unique frames to avoid re-processing
    seen_hashes = set()
    kept, removed = 0, 0
    frame_count = 0
    
    print("Starting to process video stream...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Step 1: Check if the image is mostly empty (e.g., a black screen)
        # This is the fastest check, so we do it first.
        if np.mean(frame) < 10:
            removed += 1
            continue
            
        # Step 2: Use a quick perceptual hash to skip duplicate frames instantly
        try:
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            current_hash = imagehash.phash(pil_img)
        except Exception:
            removed += 1
            continue
            
        if current_hash in seen_hashes:
            removed += 1
            continue
        
        # Step 3: Identify if the frame is a VS Code-like interface using OCR and color checks
        is_vs_code = False
        
        # Check 3.1: OCR for header and sidebar keywords (on ROIs)
        roi_left = frame[:, :int(frame.shape[1] * 0.25)]
        roi_top = frame[0:int(frame.shape[0] * 0.05), :]
        ocr_result_left = pytesseract.image_to_string(roi_left)
        ocr_result_top = pytesseract.image_to_string(roi_top)
        
        if any(kw.lower() in ocr_result_left.lower() or kw.lower() in ocr_result_top.lower() for kw in keywords):
            is_vs_code = True
        
        # Check 3.2: Color analysis for a dark theme signature
        if not is_vs_code:
            h, w, _ = frame.shape
            samples = [
                frame[h-20:h, w-100:w],  # Bottom right status bar
                frame[100:150, 0:50]     # Top left activity bar
            ]
            
            for sample in samples:
                if sample.size > 0:
                    gray_sample = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
                    dark_pixels = np.sum(gray_sample < 50) 
                    if dark_pixels > sample.size * 0.5:
                        is_vs_code = True
                        break

        # If the frame is identified as a VS Code window
        if is_vs_code:
            # We add its hash to our set and save the frame
            seen_hashes.add(current_hash)
            save_path = output_path / f"frame_{kept:05d}.jpg"
            cv2.imwrite(str(save_path), frame)
            kept += 1
        else:
            removed += 1
            
    cap.release()
    print(f"✅ Filtering complete. Total frames processed: {frame_count}. Kept: {kept}, Removed: {removed}")


extract_code_frames("./output/video.mp4", "./output/vscode2_frames")