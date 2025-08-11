# 📹 Video Code Extractor with Gemini Integration

Extract code snippets from YouTube videos using OCR and automatically fix syntax errors using Google Gemini's free API.

---

## 🚀 Features

- 📥 Download videos using `yt-dlp`
- 🎞️ Extract frames with `ffmpeg`
- ✂️ Optional crop region (ROI) to focus on code area
- 🧠 Deduplicate similar frames using perceptual hashing
- 🔍 OCR text extraction via `pytesseract`
- 📄 Merge all extracted text into a single file
- 🤖 Auto-fix syntax and OCR errors using Gemini 2.5 Flash

---

## 🛠️ Requirements

### 🔧 System Tools

- [`yt-dlp`](https://github.com/yt-dlp/yt-dlp)
- [`ffmpeg`](https://ffmpeg.org/)
- [`Tesseract OCR`](https://github.com/tesseract-ocr/tesseract)

### 📦 Python Packages

Install required packages:

```bash
pip install pillow pytesseract imagehash requests tqdm google-genai
```

### 🔐 Gemini API Setup

- Visit Google AI Studio
- Create a free API key (no billing required)
- Export the key in your terminal:

```bash
export GEMINI_API_KEY="your_api_key_here"
```

---

## 📦 Usage Examples

### ▶️ Basic Extraction (No Gemini Fix)
```bash
python3 video_code_extractor.py \
  --url "https://www.youtube.com/watch?v=VIDEO_ID" \
  --outdir ./output \
  --fps 1
```

### 🧠 With Crop Region and Gemini Auto-Fix
```bash
export GEMINI_API_KEY="your_api_key_here"

python3 video_code_extractor.py \
  --url "https://www.youtube.com/watch?v=VIDEO_ID" \
  --outdir ./output \
  --fps 1 \
  --roi 200,150,1200,600 \
  --use_gemini
```
📐 ROI format: x,y,w,h in pixel

---

## 📁 Output Files

| File Name | Description | 
| raw_ocr.txt | Merged OCR text from all frames | 
| fixed_by_gemini.txt | Gemini-corrected code (only if --use_gemini) | 

---

## ⚠️ Notes

- OCR accuracy depends on video quality, font style, and frame capture rate.
- Gemini helps repair syntax but may infer logic—manual review is recommended.
- Always respect copyright and licensing of video content.

---

## 💡 Tips

For best results, use high-resolution videos with clear code formatting and consistent backgrounds.

---

## 🧑‍💻 Author

Vibe coded by Sabbas
