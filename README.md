# 📹 YouTube Video Code Extractor with Google Gemini AI Auto-Fix

**Easily extract and clean up code from YouTube programming tutorials** using advanced OCR (Tesseract) and AI-powered syntax correction from **Google Gemini 2.5 Flash** — all in a single Python tool.

---

## 🚀 Key Features

- 📥 **Download videos** directly with [`yt-dlp`](https://github.com/yt-dlp/yt-dlp)  
- 🎞️ **Frame extraction** using [`ffmpeg`](https://ffmpeg.org/)  
- 📐 **Custom crop/ROI** to focus on the exact code region  
- 🧠 **Duplicate frame detection** via perceptual hashing for faster OCR  
- 🔍 **Text-to-code extraction** with [`Tesseract OCR`](https://github.com/tesseract-ocr/tesseract)  
- 📄 **Automatic code merging** into a single file  
- 🤖 **Google Gemini AI auto-fix** for syntax and OCR errors (free API tier)  
- ⚡ Works on **Python 3.8+** and is fully **cross-platform**  

---

## 🛠️ System Requirements

### 📌 Tools You Need Installed
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) — YouTube video downloader
- [ffmpeg](https://ffmpeg.org/) — Frame extraction tool
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) — Optical Character Recognition

### 📦 Python Dependencies
```bash
pip install pillow pytesseract imagehash requests tqdm google-genai
```

---

## 🔐 Google Gemini API Setup

1. Go to **[Google AI Studio](https://aistudio.google.com/)**  
2. Create a **free API key** (no billing required)  
3. Export it in your terminal:  
```bash
export GEMINI_API_KEY="your_api_key_here"
```

---

## 📦 How to Use

### ▶️ Extract Code (No AI Fix)
```bash
python3 video_code_extractor.py   --url "https://www.youtube.com/watch?v=VIDEO_ID"   --outdir ./output   --fps 1
```

### 🧠 Extract & Auto-Fix with Google Gemini
```bash
export GEMINI_API_KEY="your_api_key_here"

python3 video_code_extractor.py   --url "https://www.youtube.com/watch?v=VIDEO_ID"   --outdir ./output   --fps 1   --roi 200,150,1200,600   --use_gemini
```
📐 **ROI format:** `x,y,w,h` in pixels

---

## 📂 Output Files

| File Name              | Description |
|------------------------|-------------|
| `raw_ocr.txt`          | Merged OCR text from frames |
| `fixed_by_gemini.txt`  | AI-corrected, syntax-fixed code (if `--use_gemini` is enabled) |

---

## ⚠️ Important Notes

- **OCR accuracy** depends on video resolution, font clarity, and frame rate  
- **AI fixes** syntax and typos but may make assumptions about logic—manual review is advised  
- Respect **copyright laws** when using this tool  

---

## 💡 Pro Tips for Best Results
- Use **1080p or higher** video quality  
- Keep code background **solid and high contrast**  
- Increase `--fps` for fast-changing code screens  

---

## 🎯 SEO Keywords
`YouTube code extractor` `Extract code from YouTube video` `AI code fixer`  
`Google Gemini API OCR` `Python YouTube code scraper` `Tesseract OCR code recognition`  
`Auto fix code errors` `Gemini 2.5 Flash API` `Convert YouTube tutorial to code`  

---

## 🧑‍💻 Author
Vibe-Coded by **Sabbas** — making developers’ lives easier, one script at a time.
