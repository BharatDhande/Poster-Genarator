import os
import io
import re
import json
import base64
import hashlib
import colorsys
import requests
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Tuple

from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
from flask import Flask, render_template, request, session, jsonify
from dotenv import load_dotenv

# -------------------------
# Load env & config
# -------------------------
load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY", "dev")

OPENAI_API_KEY       = os.getenv("OPENAI_API_KEY")
UNSPLASH_ACCESS_KEY  = os.getenv("UNSPLASH_ACCESS_KEY")
PEXELS_API_KEY       = os.getenv("PEXELS_API_KEY")
LOGO_PATH            = os.getenv("LOGO_PATH", "logo.png")

BRAND_NAME_DEFAULT    = os.getenv("BRAND_NAME", "Your Company")
BRAND_WEBSITE_DEFAULT = os.getenv("BRAND_WEBSITE", "https://yourcompany.com")
BRAND_CONTACT_DEFAULT = os.getenv("BRAND_CONTACT", "+1 234 567 890")

POSTERS_DIR = os.path.join("static", "posters")
os.makedirs(POSTERS_DIR, exist_ok=True)

# -------------------------
# Flask
# -------------------------
app = Flask(__name__)
app.secret_key = SECRET_KEY

# -------------------------
# OpenAI client (optional)
# -------------------------
try:
    from openai import OpenAI
    openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
except Exception:
    openai_client = None

# -------------------------
# Utils
# -------------------------
def timestamped_filename(ext="png"):
    return datetime.now(timezone.utc).strftime(f"poster_%Y%m%d_%H%M%S_%f.{ext}")

def try_truetype(font_candidates, size):
    for f in font_candidates:
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    return ImageFont.load_default()

def pick_fonts(w):
    h = max(64, int(w * 0.075))
    s = max(42, int(w * 0.045))
    b = max(32, int(w * 0.032))
    br = max(28, int(w * 0.028))
    candidates_bold = ["Montserrat-Bold.ttf", "Poppins-Bold.ttf", "arialbd.ttf", "arial.ttf"]
    candidates_semi = ["Montserrat-SemiBold.ttf", "Poppins-SemiBold.ttf", "arial.ttf"]
    candidates_reg  = ["OpenSans-Regular.ttf", "Poppins-Regular.ttf", "arial.ttf"]
    return (
        try_truetype(candidates_bold, h),
        try_truetype(candidates_semi, s),
        try_truetype(candidates_reg, b),
        try_truetype(candidates_reg, br),
    )

def wrap_text(draw, text, font, max_width):
    words = text.split()
    lines, cur = [], []
    for w in words:
        test = " ".join(cur + [w])
        if draw.textlength(test, font=font) <= max_width:
            cur.append(w)
        else:
            if cur:
                lines.append(" ".join(cur))
            cur = [w]
    if cur:
        lines.append(" ".join(cur))
    return "\n".join(lines)

def text_block(draw, xy, text, font, max_width, fill=(255,255,255), line_spacing=1.2):
    text = wrap_text(draw, text, font, max_width)
    x, y = xy
    ascent, descent = font.getmetrics()
    line_h = ascent + descent
    for line in text.split("\n"):
        draw.text((x, y), line, font=font, fill=fill)
        y += int(line_h * line_spacing)
    return y

def apply_vignette(img: Image.Image, strength=0.28):
    w, h = img.size
    vignette = Image.new("L", (w, h), 0)
    grad = ImageDraw.Draw(vignette)
    max_r = int((w**2 + h**2) ** 0.5 / 2)
    for i in range(max_r, 0, -1):
        alpha = int(255 * strength * (1 - i/max_r))
        grad.ellipse([w/2 - i, h/2 - i, w/2 + i, h/2 + i], fill=alpha)
    vignette = vignette.resize((w, h), Image.LANCZOS)
    overlay = Image.new("RGBA", (w, h), (0,0,0,0))
    overlay.putalpha(vignette)
    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGBA")

def gradient_overlay_left(width, height, color=(0,0,0), max_alpha=220, fraction=0.48):
    overlay = Image.new("RGBA", (width, height), (0,0,0,0))
    draw = ImageDraw.Draw(overlay)
    grad_w = int(width * fraction)
    for x in range(grad_w):
        alpha = int(max_alpha * (1 - x / grad_w))
        draw.line([(x, 0), (x, height)], fill=(color[0], color[1], color[2], alpha))
    return overlay

def extract_brand_regex(prompt: str):
    def grab(key):
        m = re.search(rf"{key}\s*=\s*([^\n\r]+)", prompt, re.I)
        return m.group(1).strip() if m else ""
    name = grab("BRAND_NAME")
    website = grab("BRAND_WEBSITE")
    contact = grab("BRAND_CONTACT")

    if not website:
        m = re.search(r"(https?://[^\s]+)", prompt, re.I)
        website = m.group(1).strip() if m else ""
    if not contact:
        m = re.search(r"(\+?\d[\d\s\-]{7,})", prompt)
        contact = m.group(1).strip() if m else ""

    return {
        "brand_name": name or BRAND_NAME_DEFAULT,
        "brand_website": website or BRAND_WEBSITE_DEFAULT,
        "brand_contact": contact or BRAND_CONTACT_DEFAULT
    }

# ---------- Auto logo ----------
def brand_color_from_name(name: str) -> tuple[int, int, int]:
    if not name:
        name = "Brand"
    h = int(hashlib.md5(name.encode("utf-8")).hexdigest()[:6], 16) % 360
    s = 0.55
    v = 0.85
    r, g, b = colorsys.hsv_to_rgb(h / 360.0, s, v)
    return int(r * 255), int(g * 255), int(b * 255)

def initials_from_name(name: str) -> str:
    if not name:
        return "B"
    parts = re.split(r"\s+", name.strip())
    if len(parts) == 1:
        return parts[0][:2].upper()
    return (parts[0][:1] + parts[-1][:1]).upper()

def create_brand_logo(brand_name: str, size: int = 256) -> Image.Image:
    bg_color = brand_color_from_name(brand_name)
    initials = initials_from_name(brand_name)
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse([0, 0, size, size], fill=bg_color + (255,))
    for ttf in ["Montserrat-Bold.ttf", "Poppins-Bold.ttf", "arialbd.ttf", "arial.ttf"]:
        try:
            font = ImageFont.truetype(ttf, int(size * 0.42))
            break
        except OSError:
            font = ImageFont.load_default()
    tw = draw.textlength(initials, font=font)
    ascent, descent = font.getmetrics()
    th = ascent + descent
    cx = (size - tw) / 2
    cy = (size - th) / 2
    draw.text((cx + 2, cy + 2), initials, font=font, fill=(0, 0, 0, 90))
    draw.text((cx, cy), initials, font=font, fill=(255, 255, 255, 255))
    ring_w = max(2, size // 40)
    draw.ellipse([ring_w//2, ring_w//2, size - ring_w//2, size - ring_w//2],
                 outline=(255, 255, 255, 35), width=ring_w)
    return img

def paste_logo(base: Image.Image, brand_name: str, pad: int):
    w, h = base.size
    try:
        if os.path.exists(LOGO_PATH):
            raw = Image.open(LOGO_PATH).convert("RGBA")
        else:
            raw = create_brand_logo(brand_name, size=int(w * 0.18))

        max_logo_w = int(w * 0.18)
        lw, lh = raw.size
        ratio = max_logo_w / lw
        logo = raw.resize((int(lw * ratio), int(lh * ratio)), Image.LANCZOS)
        base.alpha_composite(logo, (w - logo.width - pad, h - logo.height - pad))
    except Exception as e:
        print("logo error:", e)

# ---------- Background providers ----------
def fetch_stock_image(keywords: str, size=(1080, 1350)) -> Image.Image | None:
    if UNSPLASH_ACCESS_KEY:
        try:
            url = "https://api.unsplash.com/photos/random"
            params = {"query": keywords, "orientation": "portrait", "count": 1}
            headers = {"Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"}
            r = requests.get(url, params=params, headers=headers, timeout=10)
            if r.ok:
                data = r.json()
                if isinstance(data, list) and data:
                    img_url = data[0]["urls"]["regular"]
                    img = Image.open(io.BytesIO(requests.get(img_url, timeout=10).content)).convert("RGB")
                    return img.resize(size, Image.LANCZOS)
        except Exception as e:
            print("Unsplash error:", e)

    if PEXELS_API_KEY:
        try:
            url = "https://api.pexels.com/v1/search"
            params = {"query": keywords, "orientation": "portrait", "per_page": 1}
            headers = {"Authorization": PEXELS_API_KEY}
            r = requests.get(url, params=params, headers=headers, timeout=10)
            if r.ok:
                data = r.json()
                photos = data.get("photos", [])
                if photos:
                    img_url = photos[0]["src"]["large"]
                    img = Image.open(io.BytesIO(requests.get(img_url, timeout=10).content)).convert("RGB")
                    return img.resize(size, Image.LANCZOS)
        except Exception as e:
            print("Pexels error:", e)

    return None

def generate_ai_background(prompt: str, size=(1080, 1350)) -> Image.Image | None:
    if not openai_client:
        return None
    try:
        w, h = size
        # OpenAI image models usually accept square/standard sizes; if your plan supports custom size use it.
        # We'll request 1024x1280 (close to 4:5) if allowed, else 1024x1024 and resize.
        target = f"{min(1024, w)}x{min(1024, h)}"
        gen_prompt = (
            "A professional vertical marketing poster background (no real text), "
            "clean futuristic healthcare theme, gradients, abstract tech shapes, "
            "room for left text overlay, high contrast for readability."
        )
        resp = openai_client.images.generate(
            model="gpt-image-1",
            prompt=gen_prompt,
            size=target,
            n=1
        )
        b64 = resp.data[0].b64_json
        img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
        return img.resize(size, Image.LANCZOS)
    except Exception as e:
        print("OpenAI image error:", e)
        return None

# ---------- LLM copy writer ----------
def llm_analyze_and_write(prompt: str) -> dict:
    fb = extract_brand_regex(prompt)

    if openai_client:
        system = (
            "You are a marketing copy expert for healthcare AI posters. "
            "Return valid JSON only."
        )
        user = f"""
Input: {prompt}

Task:
1) Extract key_phrases.
2) headline (<= 10 words)
3) subheadline (<= 20 words)
4) 3 short bullet points (<= 8 words each)
5) badge_text if accuracy exists
6) social caption + 3-4 hashtags
7) Extract brand_name, brand_website, brand_contact (empty if not provided)

Return JSON:
{{
  "key_phrases": [],
  "headline": "",
  "subheadline": "",
  "bullets": [],
  "badge_text": "",
  "caption": "",
  "hashtags": [],
  "brand_name": "",
  "brand_website": "",
  "brand_contact": ""
}}
"""
        try:
            resp = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                response_format={"type": "json_object"}
            )
            data = json.loads(resp.choices[0].message.content)
            if not data.get("brand_name"):
                data["brand_name"] = fb["brand_name"]
            if not data.get("brand_website"):
                data["brand_website"] = fb["brand_website"]
            if not data.get("brand_contact"):
                data["brand_contact"] = fb["brand_contact"]
            return data
        except Exception as e:
            print("LLM parse error:", e)

    # ---- Fallback (no LLM) ----
    accuracy = None
    m = re.search(r"(\d{2,3}%\s*accuracy)", prompt, re.I)
    if m:
        accuracy = m.group(1)
    diseases = re.findall(r"(diabetes|heart disease|cancer|stroke|alzheimer's)", prompt, re.I)
    diseases = list({d.lower() for d in diseases})
    key_phrases = re.findall(r'"([^"]+)"', prompt) or []
    if not key_phrases:
        words = re.findall(r"[A-Za-z%]+", prompt)
        key_phrases = list({w for w in words if len(w) > 5})[:6]

    badge = accuracy.title() if accuracy else ""
    caption = "Introducing our AI-powered medical diagnosis system"
    if accuracy:
        caption += f" – delivering {accuracy} results"
    if diseases:
        caption += f" for {', '.join(diseases)}"
    caption += " within minutes."
    hashtags = ["#AIHealthcare", "#DiagnosisRevolution", "#MedTech"]

    return {
        "key_phrases": key_phrases,
        "headline": "AI-Based Medical Diagnosis System",
        "subheadline": "Instant, accurate results you can trust.",
        "bullets": [
            f"{accuracy} accuracy" if accuracy else "High accuracy",
            "Covers diabetes & heart disease" if diseases else "Broad disease coverage",
            "Fast & secure results"
        ],
        "badge_text": badge,
        "caption": caption,
        "hashtags": hashtags,
        "brand_name": fb["brand_name"],
        "brand_website": fb["brand_website"],
        "brand_contact": fb["brand_contact"]
    }

# ---------- Main poster generator ----------
def generate_poster_with_background_and_text(prompt: str, size=(1080, 1350)) -> dict:
    gen = llm_analyze_and_write(prompt)

    w, h = size

    # 1) Background (AI -> stock -> gradient fallback)
    bg = generate_ai_background(prompt, size) or \
         fetch_stock_image(" ".join(gen.get("key_phrases") or []) or "healthcare ai technology", size=size) or \
         None

    if bg is None:
        # Gradient fallback
        bg = Image.new("RGB", size, (20, 35, 70))
        grad = Image.linear_gradient("L").resize((w, h))
        grad = ImageOps.colorize(grad, (10, 20, 40), (30, 90, 140))
        bg = Image.blend(bg, grad, 0.6)

    base = bg.convert("RGBA")
    base = apply_vignette(base, strength=0.26)

    # left gradient panel for text
    base = Image.alpha_composite(base, gradient_overlay_left(w, h, (0, 0, 0), 220, 0.48))
    draw = ImageDraw.Draw(base)

    # 2) Fonts
    f_h, f_s, f_b, f_brand = pick_fonts(w)

    # 3) Layout
    pad = int(w * 0.06)
    text_x = pad
    text_y = pad
    text_w = int(w * 0.42) - 2 * pad

    # Headline
    text_y = text_block(draw, (text_x, text_y), gen["headline"], f_h, text_w, fill=(255,255,255), line_spacing=1.15)
    text_y += int(w * 0.015)

    # Subheadline
    text_y = text_block(draw, (text_x, text_y), gen["subheadline"], f_s, text_w, fill=(215,230,255), line_spacing=1.18)
    text_y += int(w * 0.02)

    # Bullets
    for b in gen["bullets"][:3]:
        text_y = text_block(draw, (text_x, text_y), f"• {b}", f_b, text_w, fill=(245,245,245), line_spacing=1.18)
        text_y += int(w * 0.009)

    # Branding
    brand_block_y = int(h * 0.84)
    brand_txt = f"{gen['brand_name']}\n🌐 {gen['brand_website']}\n📞 {gen['brand_contact']}"
    text_block(draw, (pad, brand_block_y), brand_txt, f_brand, text_w, fill=(220,230,240), line_spacing=1.2)

    # Auto / real logo
    paste_logo(base, gen['brand_name'], pad)

    # Save
    fname = timestamped_filename("png")
    path = os.path.join(POSTERS_DIR, fname)
    base.convert("RGB").save(path, "PNG")

    caption = (gen["caption"] + " " + " ".join(gen.get("hashtags", []))).strip()

    return {
        "poster_url": f"/static/posters/{fname}",
        "caption": caption,
        "analysis": gen
    }

# -------------------------
# Routes
# -------------------------
@app.route("/", methods=["GET"])
def index():
    if "history" not in session:
        session["history"] = []
    return render_template("index.html", history=session["history"])

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json() or {}
        prompt = data.get("prompt", "").strip()
        ratio  = data.get("ratio", "1080x1350")
        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400

        try:
            w, h = map(int, ratio.lower().split("x"))
            size = (w, h)
        except Exception:
            size = (1080, 1350)

        res = generate_poster_with_background_and_text(prompt, size=size)

        history = session.get("history", [])
        history.append({
            "prompt": prompt,
            "poster_url": res["poster_url"],
            "caption": res["caption"]
        })
        session["history"] = history

        return jsonify(res)
    except Exception as e:
        print("Error in /chat:", e)
        return jsonify({"error": str(e)}), 500
    
from flask import send_file

@app.route("/download_poster/<filename>")
def download_poster(filename):
    poster_path = os.path.join(POSTERS_DIR, filename)
    if os.path.exists(poster_path):
        return send_file(poster_path, as_attachment=True)
    return "File not found", 404

@app.route("/clear_history", methods=["POST"])
def clear_history():
    session['history'] = []  # If you use Flask sessions
    return {"status": "ok"}


# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)
