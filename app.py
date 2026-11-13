"""
AI Judiciary Vision - Clean, Minimal Web UI
Antigravity design: whitespace, typography, focus
"""

import streamlit as st
import base64
import json
import os
import time
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, asdict
from io import BytesIO

import sys
sys.path.insert(0, str(Path(__file__).parent))

from agents import (
    VisionAdvocateAgent,
    VisionOpposerAgent,
    VisionRebuttalAgent,
    VisionJudgeAgent,
)
from llm import call_llm_vision
from PIL import Image

# =============================================================================
# CONFIGURATION
# =============================================================================

APP_TITLE = "AI Judiciary Vision"
MAX_IMAGE_SIZE_MB = 20
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
HISTORY_FILE = Path(__file__).parent / "data" / "history.json"
HISTORY_FILE.parent.mkdir(exist_ok=True)

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Product:
    name: str = ""
    price: str = ""
    currency: str = "$"
    images: List[Dict[str, str]] = None
    url: str = ""
    source: str = "manual"
    metadata: Dict = None
    
    def __post_init__(self):
        if self.images is None:
            self.images = []
        if self.metadata is None:
            self.metadata = {}

@dataclass
class DebateResult:
    id: str
    timestamp: str
    question: str
    products: List[Dict]
    verdict: str
    pro_scores: Dict
    con_scores: Dict
    pro_arguments: List
    con_arguments: List
    pro_rebuttal: Dict
    con_rebuttal: Dict
    reasoning: str
    strongest_pro: str
    strongest_con: str
    panels_agree: bool
    mode: str = "single"
    
    def to_dict(self) -> Dict:
        return asdict(self)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def init_session_state():
    defaults = {
        "history": [],
        "current_result": None,
        "debate_running": False,
        "products": [],
        "comparison_mode": False,
        "theme": "light",
        "selected_product_idx": 0,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val
    
    if not st.session_state.history and HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, "r") as f:
                st.session_state.history = json.load(f)
        except Exception:
            st.session_state.history = []

def save_history():
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump(st.session_state.history[-50:], f, indent=2)
    except Exception:
        pass

def generate_id() -> str:
    return hashlib.md5(f"{time.time()}{os.urandom(8).hex()}".encode()).hexdigest()[:12]

def image_to_base64(uploaded_file) -> Tuple[str, str]:
    bytes_data = uploaded_file.read()
    img = Image.open(BytesIO(bytes_data))
    img.verify()
    img = Image.open(BytesIO(bytes_data))
    
    if img.mode in ("RGBA", "LA", "P"):
        if img.mode == "P":
            img = img.convert("RGBA")
        background = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "RGBA":
            background.paste(img, mask=img.split()[-1])
        else:
            background.paste(img)
        img = background
    
    max_dim = 2048
    if max(img.size) > max_dim:
        img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    
    buffered = BytesIO()
    fmt = "PNG" if uploaded_file.name.lower().endswith(".png") else "JPEG"
    img.save(buffered, format=fmt, quality=85, optimize=True)
    b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    mime_map = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp"}
    ext = uploaded_file.name.split(".")[-1].lower()
    mime_type = mime_map.get(ext, "image/png")
    return b64, mime_type

def format_score(score: float) -> str:
    """Format a numeric score for display (e.g., 8.0 -> '8', 8.5 -> '8.5')."""
    if score is None:
        return "N/A"
    try:
        val = float(score)
        if val.is_integer():
            return str(int(val))
        return f"{val:.1f}"
    except (TypeError, ValueError):
        return "N/A"


def score_value(score) -> float:
    """Extract numeric value from score for calculations. Returns 0.0 for invalid scores."""
    if score is None:
        return 0.0
    try:
        return float(score)
    except (TypeError, ValueError):
        return 0.0

def create_pdf_report(result: DebateResult) -> bytes:
    try:
        from fpdf import FPDF
    except ImportError:
        return None
    
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()
    
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 12, "AI Judiciary Vision - Analysis Report", ln=True, align="C")
    pdf.ln(5)
    
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, f"Generated: {datetime.fromisoformat(result.timestamp).strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.cell(0, 6, f"Question: {result.question}", ln=True)
    pdf.cell(0, 6, f"Mode: {result.mode.capitalize()}", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 10, "Products Analyzed", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(0, 0, 0)
    for i, p in enumerate(result.products, 1):
        pdf.cell(0, 6, f"  {i}. {p.get('name', 'Unknown')} - {p.get('price', 'N/A')}", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 10, "Final Verdict", ln=True)
    pdf.set_font("Helvetica", "B", 12)
    
    verdict_colors = {"pro": (34, 197, 94), "con": (239, 68, 68), "draw": (107, 114, 128), "split": (245, 158, 11)}
    color = verdict_colors.get(result.verdict, (107, 114, 128))
    pdf.set_text_color(*color)
    verdict_label = {"pro": "FOR (Advocate Wins)", "con": "AGAINST (Opposer Wins)", "draw": "DRAW", "split": "SPLIT DECISION"}.get(result.verdict, result.verdict)
    pdf.cell(0, 10, f"Result: {verdict_label}", ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Panels Agree: {'Yes' if result.panels_agree else 'No (Split Decision)'}", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 10, "Scores", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(0, 0, 0)
    
    pro_total = sum(score_value(v) for v in result.pro_scores.values())
    con_total = sum(score_value(v) for v in result.con_scores.values())
    
    pdf.cell(0, 6, f"Advocate (FOR): {format_score(pro_total)}/30", ln=True)
    for k, v in result.pro_scores.items():
        pdf.cell(0, 6, f"  {k.capitalize()}: {format_score(v)}/10", ln=True)
    pdf.ln(2)
    pdf.cell(0, 6, f"Opposer (AGAINST): {format_score(con_total)}/30", ln=True)
    for k, v in result.con_scores.items():
        pdf.cell(0, 6, f"  {k.capitalize()}: {format_score(v)}/10", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 10, "Arguments", ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(0, 0, 0)
    
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "Advocate (FOR):", ln=True)
    pdf.set_font("Helvetica", "", 9)
    for i, arg in enumerate(result.pro_arguments, 1):
        pdf.multi_cell(0, 5, f"  [{i}] {arg.get('point', '')}")
        pdf.multi_cell(0, 5, f"      Reason: {arg.get('reason', '')[:200]}")
        pdf.multi_cell(0, 5, f"      Evidence: {arg.get('evidence', '')[:200]}")
        pdf.ln(2)
    
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "Opposer (AGAINST):", ln=True)
    pdf.set_font("Helvetica", "", 9)
    for i, arg in enumerate(result.con_arguments, 1):
        pdf.multi_cell(0, 5, f"  [{i}] {arg.get('point', '')}")
        pdf.multi_cell(0, 5, f"      Reason: {arg.get('reason', '')[:200]}")
        pdf.multi_cell(0, 5, f"      Evidence: {arg.get('evidence', '')[:200]}")
        pdf.ln(2)
    
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 10, "Rebuttals", ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 6, f"Advocate: {result.pro_rebuttal.get('rebuttal', '')[:300]}", ln=True)
    pdf.ln(2)
    pdf.cell(0, 6, f"Opposer: {result.con_rebuttal.get('rebuttal', '')[:300]}", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 10, "Judicial Reasoning", ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(0, 0, 0)
    pdf.multi_cell(0, 5, result.reasoning)
    
    return pdf.output(dest="S").encode("latin-1")

# =============================================================================
# DEBATE ENGINE WITH PARALLEL EXECUTION
# =============================================================================

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Any
import time


@dataclass
class AgentTiming:
    """Timing information for agent calls."""
    advocate: float = 0.0
    opposer: float = 0.0
    advocate_rebuttal: float = 0.0
    opposer_rebuttal: float = 0.0
    judge_panel1: float = 0.0
    judge_panel2: float = 0.0
    total: float = 0.0


def _run_with_timing(func: Callable, *args, **kwargs) -> tuple:
    """Run a function and return (result, elapsed_time)."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


def run_debate(question: str, image_b64: str, mime_type: str, progress_callback=None, max_workers: int = 2) -> Dict[str, Any]:
    """
    Run the debate with bounded parallel execution.
    
    Phases:
    1. Advocate + Opposer (parallel)
    2. Both rebuttals (parallel, after phase 1)
    3. Both judge panels (parallel, after phase 2)
    4. Compile verdict (sequential)
    """
    timings = AgentTiming()
    total_start = time.perf_counter()
    
    def update_progress(step: int, message: str):
        if progress_callback:
            progress_callback(step, message)
    
    # Phase 1: Advocate and Opposer in parallel
    update_progress(1, "Advocate + Opposer analyzing image (parallel)...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_advocate = executor.submit(VisionAdvocateAgent.argue, question, image_b64, mime_type)
        future_opposer = executor.submit(VisionOpposerAgent.argue, question, image_b64, mime_type)
        
        pro_case, timings.advocate = _run_with_timing(future_advocate.result)
        con_case, timings.opposer = _run_with_timing(future_opposer.result)
    
    pro_arguments = pro_case.get("arguments", [])
    con_arguments = con_case.get("arguments", [])
    
    # Phase 2: Both rebuttals in parallel
    update_progress(2, "Advocate + Opposer rebuttals (parallel)...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_pro_rebuttal = executor.submit(
            VisionRebuttalAgent.rebut, "Advocate", question, "in favor", con_arguments, image_b64, mime_type
        )
        future_con_rebuttal = executor.submit(
            VisionRebuttalAgent.rebut, "Opposer", question, "against", pro_arguments, image_b64, mime_type
        )
        
        pro_rebuttal, timings.advocate_rebuttal = _run_with_timing(future_pro_rebuttal.result)
        con_rebuttal, timings.opposer_rebuttal = _run_with_timing(future_con_rebuttal.result)
    
    # Phase 3: Both judge panels in parallel
    update_progress(3, "Judge Panel 1 + Panel 2 deliberating (parallel)...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_judge1 = executor.submit(
            VisionJudgeAgent.evaluate, question, pro_arguments, con_arguments,
            pro_rebuttal, con_rebuttal, image_b64, mime_type, True
        )
        future_judge2 = executor.submit(
            VisionJudgeAgent.evaluate, question, pro_arguments, con_arguments,
            pro_rebuttal, con_rebuttal, image_b64, mime_type, False
        )
        
        run1, timings.judge_panel1 = _run_with_timing(future_judge1.result)
        run2, timings.judge_panel2 = _run_with_timing(future_judge2.result)
    
    # Phase 4: Compile verdict
    update_progress(4, "Compiling verdict...")
    
    v1 = run1["mapped"]["verdict"].lower()
    v2 = run2["mapped"]["verdict"].lower()
    
    if v1 == v2:
        r1_pro = run1["mapped"]["pro_scores"]
        r2_pro = run2["mapped"]["pro_scores"]
        r1_con = run1["mapped"]["con_scores"]
        r2_con = run2["mapped"]["con_scores"]
        
        avg_pro_scores = {k: (r1_pro.get(k, 0) + r2_pro.get(k, 0)) / 2.0 for k in ("evidence", "logic", "relevance")}
        avg_con_scores = {k: (r1_con.get(k, 0) + r2_con.get(k, 0)) / 2.0 for k in ("evidence", "logic", "relevance")}
        
        verdict = v1
        reasoning = run1["mapped"]["reasoning"]
        strongest_pro = run1["mapped"]["strongest_pro"]
        strongest_con = run1["mapped"]["strongest_con"]
        panels_agree = True
    else:
        avg_pro_scores = run1["mapped"]["pro_scores"]
        avg_con_scores = run1["mapped"]["con_scores"]
        verdict = "split"
        reasoning = f"Split decision: Panel 1 ruled '{v1}', Panel 2 ruled '{v2}'"
        strongest_pro = run1["mapped"]["strongest_pro"]
        strongest_con = run1["mapped"]["strongest_con"]
        panels_agree = False
    
    timings.total = time.perf_counter() - total_start
    
    return {
        "question": question,
        "pro_arguments": pro_arguments,
        "con_arguments": con_arguments,
        "pro_rebuttal": pro_rebuttal,
        "con_rebuttal": con_rebuttal,
        "verdict": verdict,
        "pro_scores": avg_pro_scores,
        "con_scores": avg_con_scores,
        "reasoning": reasoning,
        "strongest_pro": strongest_pro,
        "strongest_con": strongest_con,
        "panels_agree": panels_agree,
        "run1": run1,
        "run2": run2,
        "timings": timings,
    }


def run_comparison_debate(question: str, product_a: Product, product_b: Product, progress_callback=None, max_workers: int = 2) -> Dict[str, Any]:
    img_a = product_a.images[0] if product_a.images else None
    img_b = product_b.images[0] if product_b.images else None
    
    if not img_a or not img_b:
        raise ValueError("Both products must have at least one image")
    
    combined_q = (
        f"Compare these two products:\n"
        f"Product A: {product_a.name} ({product_a.price})\n"
        f"Product B: {product_b.name} ({product_b.price})\n\n"
        f"Question: {question}\n\n"
        f"Which product is better? Analyze both images and argue for one over the other."
    )
    
    return run_debate(combined_q, img_a["b64"], img_a["mime"], progress_callback, max_workers)

def run_comparison_debate(question: str, product_a: Product, product_b: Product, progress_callback=None) -> Dict[str, Any]:
    img_a = product_a.images[0] if product_a.images else None
    img_b = product_b.images[0] if product_b.images else None
    
    if not img_a or not img_b:
        raise ValueError("Both products must have at least one image")
    
    combined_q = (
        f"Compare these two products:\n"
        f"Product A: {product_a.name} ({product_a.price})\n"
        f"Product B: {product_b.name} ({product_b.price})\n\n"
        f"Question: {question}\n\n"
        f"Which product is better? Analyze both images and argue for one over the other."
    )
    
    return run_debate(combined_q, img_a["b64"], img_a["mime"], progress_callback)

# =============================================================================
# UI COMPONENTS - CLEAN, MINIMAL DESIGN
# =============================================================================

def inject_css():
    theme = st.session_state.get("theme", "light")
    
    if theme == "dark":
        bg_primary = "#0f172a"
        bg_secondary = "#1e293b"
        bg_tertiary = "#334155"
        text_primary = "#f1f5f9"
        text_secondary = "#94a3b8"
        text_muted = "#64748b"
        accent = "#3b82f6"
        accent_hover = "#2563eb"
        border = "#334155"
        card_bg = "#1e293b"
        success = "#22c55e"
        error = "#ef4444"
        warning = "#f59e0b"
        info = "#3b82f6"
    else:
        bg_primary = "#ffffff"
        bg_secondary = "#f8fafc"
        bg_tertiary = "#f1f5f9"
        text_primary = "#0f172a"
        text_secondary = "#475569"
        text_muted = "#94a3b8"
        accent = "#2563eb"
        accent_hover = "#1d4ed8"
        border = "#e2e8f0"
        card_bg = "#ffffff"
        success = "#16a34a"
        error = "#dc2626"
        warning = "#d97706"
        info = "#2563eb"
    
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    :root {{
        --bg-primary: {bg_primary};
        --bg-secondary: {bg_secondary};
        --bg-tertiary: {bg_tertiary};
        --text-primary: {text_primary};
        --text-secondary: {text_secondary};
        --text-muted: {text_muted};
        --accent: {accent};
        --accent-hover: {accent_hover};
        --border: {border};
        --card-bg: {card_bg};
        --success: {success};
        --error: {error};
        --warning: {warning};
        --info: {info};
    }}
    
    .stApp {{
        background: var(--bg-primary);
        color: var(--text-primary);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}
    
    .main .block-container {{
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 900px;
    }}
    
    /* Hide Streamlit branding */
    #MainMenu, footer, header {{ visibility: hidden; }}
    
    /* Typography */
    h1 {{ font-weight: 700; letter-spacing: -0.02em; font-size: 2.5rem; line-height: 1.2; margin-bottom: 0.5rem; }}
    h2 {{ font-weight: 600; letter-spacing: -0.01em; font-size: 1.5rem; line-height: 1.3; margin-top: 2rem; margin-bottom: 1rem; }}
    h3 {{ font-weight: 600; font-size: 1.1rem; line-height: 1.4; margin-top: 1.5rem; margin-bottom: 0.75rem; }}
    p {{ color: var(--text-secondary); line-height: 1.7; }}
    
    /* Cards */
    .card {{
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: box-shadow 0.2s, border-color 0.2s;
    }}
    .card:hover {{
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border-color: var(--accent);
    }}
    
    /* Buttons */
    .stButton > button {{
        background: var(--accent);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.2s;
        width: 100%;
    }}
    .stButton > button:hover {{
        background: var(--accent-hover);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
        transform: translateY(-1px);
    }}
    .stButton > button:disabled {{
        background: var(--bg-tertiary);
        color: var(--text-muted);
        cursor: not-allowed;
    }}
    .stButton > button[kind="secondary"] {{
        background: var(--bg-tertiary);
        color: var(--text-primary);
        border: 1px solid var(--border);
    }}
    .stButton > button[kind="secondary"]:hover {{
        background: var(--border);
        box-shadow: none;
    }}
    
    /* Inputs */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > div {{
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 8px;
        color: var(--text-primary);
    }}
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {{
        border-color: var(--accent);
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15);
    }}
    
    /* File uploader */
    .stFileUploader > div {{
        background: var(--card-bg);
        border: 2px dashed var(--border);
        border-radius: 12px;
        padding: 2rem;
        transition: all 0.2s;
    }}
    .stFileUploader > div:hover {{
        border-color: var(--accent);
        background: var(--bg-secondary);
    }}
    .stFileUploader > div > div {{
        color: var(--text-secondary);
    }}
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
        background: var(--bg-tertiary);
        padding: 4px;
        border-radius: 10px;
        margin-bottom: 1.5rem;
    }}
    .stTabs [data-baseweb="tab"] {{
        padding: 0.6rem 1.2rem;
        border-radius: 8px;
        font-weight: 500;
        font-size: 0.9rem;
        color: var(--text-secondary);
        background: transparent;
    }}
    .stTabs [aria-selected="true"] {{
        background: var(--card-bg) !important;
        color: var(--accent) !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }}
    
    /* Progress */
    .stProgress > div > div > div > div {{
        background: linear-gradient(90deg, var(--accent) 0%, var(--accent-hover) 100%);
        border-radius: 4px;
    }}
    
    /* Metrics */
    [data-testid="metric-container"] {{
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.25rem;
    }}
    [data-testid="metric-container"] label {{
        color: var(--text-secondary) !important;
        font-size: 0.85rem !important;
    }}
    [data-testid="metric-container"] [data-testid="stMetricValue"] {{
        color: var(--text-primary) !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }}
    
    /* Expander */
    .streamlit-expanderHeader {{
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 8px;
        font-weight: 600;
        color: var(--text-primary);
    }}
    .streamlit-expanderContent {{
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-top: none;
        border-radius: 0 0 8px 8px;
        padding: 1rem;
    }}
    
    /* Alerts */
    .stAlert {{
        border-radius: 10px;
        border: none;
    }}
    
    /* Scrollbar */
    ::-webkit-scrollbar {{ width: 8px; height: 8px; }}
    ::-webkit-scrollbar-track {{ background: var(--bg-tertiary); }}
    ::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 4px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: var(--text-muted); }}
    
    /* Custom components */
    .verdict-badge {{
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.75rem 1.5rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1rem;
    }}
    .verdict-pro {{ background: rgba(34, 197, 94, 0.15); color: var(--success); border: 1px solid var(--success); }}
    .verdict-con {{ background: rgba(239, 68, 68, 0.15); color: var(--error); border: 1px solid var(--error); }}
    .verdict-draw {{ background: rgba(107, 114, 128, 0.15); color: var(--text-muted); border: 1px solid var(--text-muted); }}
    .verdict-split {{ background: rgba(245, 158, 11, 0.15); color: var(--warning); border: 1px solid var(--warning); }}
    
    .score-bar {{
        height: 6px;
        background: var(--bg-tertiary);
        border-radius: 3px;
        overflow: hidden;
        margin-top: 0.5rem;
    }}
    .score-fill {{
        height: 100%;
        border-radius: 3px;
        transition: width 0.5s ease;
    }}
    .score-fill-pro {{ background: linear-gradient(90deg, var(--success), #4ade80); }}
    .score-fill-con {{ background: linear-gradient(90deg, var(--error), #f87171); }}
    
    .argument-card {{
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }}
    .argument-title {{
        font-weight: 600;
        font-size: 1rem;
        margin-bottom: 0.75rem;
        color: var(--text-primary);
    }}
    .argument-meta {{
        font-size: 0.85rem;
        color: var(--text-muted);
        margin-bottom: 0.5rem;
    }}
    .argument-content {{
        font-size: 0.95rem;
        line-height: 1.7;
        color: var(--text-secondary);
    }}
    
    .rebuttal-card {{
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        border-left: 3px solid var(--accent);
        border-radius: 8px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }}
    
    .product-preview {{
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 1rem;
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 10px;
    }}
    .product-preview img {{
        width: 80px;
        height: 80px;
        object-fit: cover;
        border-radius: 8px;
    }}
    .product-info {{
        flex: 1;
    }}
    .product-name {{ font-weight: 600; color: var(--text-primary); }}
    .product-price {{ color: var(--accent); font-weight: 500; }}
    .product-meta {{ font-size: 0.8rem; color: var(--text-muted); }}
    
    .history-item {{
        padding: 1rem;
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 10px;
        margin-bottom: 0.75rem;
        transition: all 0.2s;
    }}
    .history-item:hover {{
        border-color: var(--accent);
    }}
    .history-question {{ font-weight: 500; color: var(--text-primary); margin-bottom: 0.25rem; }}
    .history-meta {{ font-size: 0.8rem; color: var(--text-muted); }}
    
    .empty-state {{
        text-align: center;
        padding: 4rem 2rem;
        color: var(--text-muted);
    }}
    .empty-state svg {{
        width: 80px;
        height: 80px;
        margin-bottom: 1.5rem;
        opacity: 0.5;
    }}
    
    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background: var(--bg-secondary);
        border-right: 1px solid var(--border);
    }}
    section[data-testid="stSidebar"] .stButton > button {{
        background: var(--bg-tertiary);
        color: var(--text-primary);
        border: 1px solid var(--border);
    }}
    section[data-testid="stSidebar"] .stButton > button:hover {{
        background: var(--border);
        box-shadow: none;
    }}
    
    /* Selectbox label fix */
    .stSelectbox label {{ display: none; }}
    
    /* Animation */
    @keyframes fadeIn {{ from {{ opacity: 0; transform: translateY(10px); }} to {{ opacity: 1; transform: translateY(0); }} }}
    .animate-in {{ animation: fadeIn 0.4s ease-out; }}
    
    /* Debate step indicator */
    .step-indicator {{
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 1rem;
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 10px;
        margin-bottom: 1rem;
    }}
    .step-icon {{
        width: 36px;
        height: 36px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.1rem;
        background: var(--bg-tertiary);
        color: var(--text-muted);
    }}
    .step-icon.active {{
        background: var(--accent);
        color: white;
    }}
    .step-icon.done {{
        background: var(--success);
        color: white;
    }}
    .step-text {{ flex: 1; }}
    .step-label {{ font-weight: 500; color: var(--text-primary); }}
    .step-desc {{ font-size: 0.85rem; color: var(--text-muted); }}
    </style>
    """, unsafe_allow_html=True)

def render_header():
    col1, col2 = st.columns([4, 1])
    
    with col1:
        st.markdown(f"""
        <div class="animate-in">
            <h1 style="margin: 0; background: linear-gradient(135deg, var(--accent) 0%, #06b6d4 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                ⚖️ AI Judiciary Vision
            </h1>
            <p style="margin: 0.5rem 0 0; font-size: 1.1rem;">AI-Powered Product Analysis & Debate</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        theme_options = ["☀️ Light", "🌙 Dark"]
        current_idx = 0 if st.session_state.theme == "light" else 1
        theme = st.selectbox(
            "Theme",
            theme_options,
            index=current_idx,
            label_visibility="collapsed",
            key="theme_selector"
        )
        st.session_state.theme = "dark" if "Dark" in theme else "light"

def render_product_section():
    st.markdown("### Product Images")
    
    tabs = st.tabs(["📤 Upload", "🔗 Fetch URL", "⚖️ Compare"])
    
    with tabs[0]:
        uploaded_files = st.file_uploader(
            "Drag & drop product images",
            type=list(ALLOWED_EXTENSIONS),
            accept_multiple_files=True,
            help=f"Max {MAX_IMAGE_SIZE_MB}MB each. PNG, JPG, WebP.",
            label_visibility="collapsed"
        )
        
        if uploaded_files:
            new_products = []
            cols = st.columns(min(len(uploaded_files), 4))
            for idx, f in enumerate(uploaded_files):
                with cols[idx % 4]:
                    try:
                        b64, mime = image_to_base64(f)
                        st.image(f, use_container_width=True)
                        product = Product(
                            name=f.name,
                            images=[{"b64": b64, "mime": mime, "url": ""}],
                            source="manual"
                        )
                        new_products.append(product)
                    except Exception as e:
                        st.error(f"Failed: {e}")
            
            if new_products:
                st.session_state.products = new_products
                st.success(f"Loaded {len(new_products)} product(s)")
    
    with tabs[1]:
        try:
            import requests
            from bs4 import BeautifulSoup
            url_fetch_available = True
        except ImportError:
            url_fetch_available = False
        
        if not url_fetch_available:
            st.warning("Install `requests` and `beautifulsoup4` for URL fetching")
        else:
            url = st.text_input("Product URL", placeholder="https://amazon.com/dp/... or Shopify store", label_visibility="collapsed")
            
            if st.button("Fetch Product", type="primary", disabled=not url, use_container_width=True):
                if url:
                    with st.spinner("Fetching..."):
                        product = fetch_product_from_url(url)
                        if product:
                            st.session_state.products = [product]
                            st.success(f"Fetched: {product.name or 'Product'}")
                            st.rerun()
                        else:
                            st.error("Failed to fetch. Try manual upload.")
            
            if st.session_state.products and st.session_state.products[0].source != "manual":
                p = st.session_state.products[0]
                if p.url:
                    st.markdown("---")
                    render_product_preview(p)
    
    with tabs[2]:
        st.session_state.comparison_mode = st.checkbox("Enable Comparison Mode", value=st.session_state.comparison_mode)
        
        if st.session_state.comparison_mode:
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown("**Product A**")
                files_a = st.file_uploader("Upload", type=list(ALLOWED_EXTENSIONS), accept_multiple_files=True, key="comp_a", label_visibility="collapsed")
                if files_a:
                    b64, mime = image_to_base64(files_a[0])
                    st.image(files_a[0], use_container_width=True)
                    if len(st.session_state.products) >= 2:
                        st.session_state.products[0] = Product(name=files_a[0].name, images=[{"b64": b64, "mime": mime}], source="manual")
                    else:
                        st.session_state.products = [Product(name=files_a[0].name, images=[{"b64": b64, "mime": mime}], source="manual")]
            
            with col_b:
                st.markdown("**Product B**")
                files_b = st.file_uploader("Upload", type=list(ALLOWED_EXTENSIONS), accept_multiple_files=True, key="comp_b", label_visibility="collapsed")
                if files_b:
                    b64, mime = image_to_base64(files_b[0])
                    st.image(files_b[0], use_container_width=True)
                    if len(st.session_state.products) >= 2:
                        st.session_state.products[1] = Product(name=files_b[0].name, images=[{"b64": b64, "mime": mime}], source="manual")
                    else:
                        st.session_state.products.append(Product(name=files_b[0].name, images=[{"b64": b64, "mime": mime}], source="manual"))
            
            if len(st.session_state.products) >= 2:
                st.success("Ready to compare")

def render_product_preview(p: Product):
    col1, col2 = st.columns([1, 3])
    with col1:
        if p.images:
            st.image(f"data:{p.images[0]['mime']};base64,{p.images[0]['b64']}", width=150)
    with col2:
        st.write(f"**{p.name or 'Unknown'}**")
        st.write(f"**Price:** {p.price or 'Unknown'}")
        st.write(f"**Source:** {p.source.title()}")
        st.write(f"**Images:** {len(p.images)}")

def render_question_section():
    st.markdown("### Question")
    
    templates = {
        "Custom": "",
        "Quality Check": "Is this product well-made and good quality based on the images?",
        "Value Assessment": "Is this product worth the price shown?",
        "Defect Detection": "Are there any visible defects, damage, or quality issues?",
        "Authenticity": "Does this product look authentic/genuine or potentially counterfeit?",
        "Durability": "Does this product look durable and long-lasting?",
        "Comparison": "Which product is better quality and value?",
    }
    
    selected = st.selectbox("Quick templates", list(templates.keys()), key="template_select", label_visibility="collapsed")
    default_q = templates[selected]
    
    question = st.text_area(
        "What would you like to know?",
        value=default_q if default_q else "",
        placeholder="e.g., Is this laptop worth $800? Does this jacket look durable? Any red flags?",
        height=100,
        key="question_input",
        label_visibility="collapsed"
    )
    
    with st.expander("💰 Price Context (optional)"):
        pc1, pc2 = st.columns(2)
        with pc1:
            price = st.number_input("Price", min_value=0.0, step=0.01, format="%.2f", key="price_input")
        with pc2:
            currency = st.selectbox("Currency", ["$", "€", "£", "¥", "₹", "Other"], key="currency_select")
    
    return question.strip(), price if price > 0 else None, currency

def render_debate_progress():
    phases = [
        (1, "🤖", "Phase 1: Opening Arguments", "Advocate + Opposer analyzing image in parallel"),
        (2, "⚔️", "Phase 2: Rebuttals", "Both sides delivering counter-arguments in parallel"),
        (3, "🏛️", "Phase 3: Judge Panels", "Panel 1 + Panel 2 deliberating in parallel (swapped labels)"),
        (4, "📋", "Phase 4: Verdict", "Compiling final verdict and scores"),
    ]
    
    progress_container = st.empty()
    status_container = st.empty()
    
    def update_ui(step: int, message: str):
        progress = step / len(phases)
        progress_container.progress(progress)
        
        phase_html = ""
        for i, (phase_num, icon, title, desc) in enumerate(phases, 1):
            if i < step:
                cls = "done"
                icon_html = "✓"
            elif i == step:
                cls = "active"
                icon_html = icon
            else:
                cls = ""
                icon_html = icon
            
            phase_html += f"""
            <div class="step-indicator">
                <div class="step-icon {cls}">{icon_html}</div>
                <div class="step-text">
                    <div class="step-label">{title}</div>
                    <div class="step-desc">{desc}</div>
                </div>
            </div>
            """
        
        status_container.markdown(phase_html, unsafe_allow_html=True)
    
    return update_ui

def render_results(result: Dict[str, Any], products: List[Product]):
    st.markdown("---")
    st.markdown("## Analysis Results")
    
    # Question & Products
    st.info(f"**Question:** {result['question']}")
    
    if products:
        cols = st.columns(len(products))
        for i, p in enumerate(products):
            with cols[i]:
                render_product_preview(p)
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Arguments", "Rebuttals", "Scores", "Verdict", "Export", "Timing"])
    
    with tab1:
        render_arguments_tab(result)
    
    with tab2:
        render_rebuttals_tab(result)
    
    with tab3:
        render_scores_tab(result)
    
    with tab4:
        render_verdict_tab(result)
    
    with tab5:
        render_export_tab(result, products)
    
    with tab6:
        render_timing_tab(result)

def render_arguments_tab(result: Dict):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Advocate (FOR)")
        for i, arg in enumerate(result["pro_arguments"], 1):
            st.markdown(f"""
            <div class="argument-card animate-in" style="animation-delay: {i*0.1}s;">
                <div class="argument-title">[{i}] {arg.get('point', '').strip()}</div>
                <div class="argument-meta">Reasoning</div>
                <div class="argument-content">{arg.get('reason', '').strip()}</div>
                <div class="argument-meta" style="margin-top: 0.75rem;">Visual Evidence</div>
                <div class="argument-content">{arg.get('evidence', '').strip()}</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Opposer (AGAINST)")
        for i, arg in enumerate(result["con_arguments"], 1):
            st.markdown(f"""
            <div class="argument-card animate-in" style="animation-delay: {i*0.1}s;">
                <div class="argument-title">[{i}] {arg.get('point', '').strip()}</div>
                <div class="argument-meta">Reasoning</div>
                <div class="argument-content">{arg.get('reason', '').strip()}</div>
                <div class="argument-meta" style="margin-top: 0.75rem;">Visual Evidence</div>
                <div class="argument-content">{arg.get('evidence', '').strip()}</div>
            </div>
            """, unsafe_allow_html=True)

def render_rebuttals_tab(result: Dict):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Advocate Rebuttal")
        reb = result["pro_rebuttal"]
        st.markdown(f"""
        <div class="rebuttal-card animate-in">
            <div class="argument-meta">Targeted Opponent Point</div>
            <div class="argument-title">{reb.get('target_point', '').strip()}</div>
            <div class="argument-meta" style="margin-top: 1rem;">Counter-Argument</div>
            <div class="argument-content">{reb.get('rebuttal', '').strip()}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Opposer Rebuttal")
        reb = result["con_rebuttal"]
        st.markdown(f"""
        <div class="rebuttal-card animate-in">
            <div class="argument-meta">Targeted Opponent Point</div>
            <div class="argument-title">{reb.get('target_point', '').strip()}</div>
            <div class="argument-meta" style="margin-top: 1rem;">Counter-Argument</div>
            <div class="argument-content">{reb.get('rebuttal', '').strip()}</div>
        </div>
        """, unsafe_allow_html=True)

def render_scores_tab(result: Dict):
    pro_scores = result["pro_scores"]
    con_scores = result["con_scores"]
    pro_total = sum(score_value(v) for v in pro_scores.values())
    con_total = sum(score_value(v) for v in con_scores.values())
    
    st.markdown("### Judicial Scorecard")
    
    col1, col2 = st.columns(2)
    
    with col1:
        delta_val = pro_total - con_total
        st.metric("Advocate (FOR)", f"{format_score(pro_total)}/30", delta=f"{delta_val:+.1f}")
        for k, v in pro_scores.items():
            pct = score_value(v) / 10
            st.markdown(f"""
            <div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                    <span style="font-weight: 500;">{k.capitalize()}</span>
                    <span style="color: var(--success); font-weight: 600;">{format_score(v)}/10</span>
                </div>
                <div class="score-bar">
                    <div class="score-fill score-fill-pro" style="width: {pct*100}%"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        delta_val = con_total - pro_total
        st.metric("Opposer (AGAINST)", f"{format_score(con_total)}/30", delta=f"{delta_val:+.1f}")
        for k, v in con_scores.items():
            pct = score_value(v) / 10
            st.markdown(f"""
            <div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                    <span style="font-weight: 500;">{k.capitalize()}</span>
                    <span style="color: var(--error); font-weight: 600;">{format_score(v)}/10</span>
                </div>
                <div class="score-bar">
                    <div class="score-fill score-fill-con" style="width: {pct*100}%"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("#### Detailed Breakdown")
    
    score_data = []
    for k in ["evidence", "logic", "relevance"]:
        pro_val = score_value(pro_scores.get(k, 0))
        con_val = score_value(con_scores.get(k, 0))
        winner = "Advocate" if pro_val > con_val else "Opposer" if con_val > pro_val else "Tie"
        score_data.append({
            "Criterion": k.capitalize(),
            "Advocate": f"{format_score(pro_val)}/10",
            "Opposer": f"{format_score(con_val)}/10",
            "Winner": winner
        })
    st.table(score_data)

def render_verdict_tab(result: Dict):
    verdict = result["verdict"]
    pro_scores = result["pro_scores"]
    con_scores = result["con_scores"]
    pro_total = sum(score_value(v) for v in pro_scores.values())
    con_total = sum(score_value(v) for v in con_scores.values())
    score_gap = abs(pro_total - con_total)
    
    if score_gap < 4:
        margin = "Narrow"
    elif score_gap < 10:
        margin = "Clear"
    else:
        margin = "Decisive"
    
    verdict_classes = {
        "pro": "verdict-pro",
        "con": "verdict-con",
        "draw": "verdict-draw",
        "split": "verdict-split"
    }
    
    verdict_labels = {
        "pro": ("🟢 ADVOCATE (FOR) PREVAILS", "verdict-pro"),
        "con": ("🔴 OPPOSER (AGAINST) PREVAILS", "verdict-con"),
        "draw": ("⚖️ DEBATE DECLARED A DRAW", "verdict-draw"),
        "split": ("⚠️ SPLIT DECISION", "verdict-split"),
    }
    
    label, cls = verdict_labels.get(verdict, (verdict, "verdict-draw"))
    
    st.markdown(f"""
    <div class="verdict-badge {cls} animate-in" style="margin-bottom: 1.5rem;">
        {label}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"**Margin:** {margin} | **Score Gap:** {format_score(score_gap)} points")
    
    if not result["panels_agree"] and verdict != "split":
        st.warning("⚠️ Judge panels disagreed. Verdict reflects Panel 1 only.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Advocate's Strongest Point**")
        st.info(result["strongest_pro"])
    with col2:
        st.markdown("**Opposer's Strongest Point**")
        st.info(result["strongest_con"])
    
    with st.expander("📜 Full Judicial Reasoning", expanded=True):
        st.markdown(result["reasoning"])

def render_timing_tab(result: Dict):
    """Render timing information for the debate."""
    timings = result.get("timings")
    if not timings:
        st.info("Timing data not available for this analysis.")
        return
    
    st.markdown("### Execution Timing")
    
    # Total time
    st.metric("Total Analysis Time", f"{timings.total:.2f}s")
    
    st.markdown("---")
    
    # Phase breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Phase 1: Opening Arguments")
        st.metric("Advocate", f"{timings.advocate:.2f}s")
        st.metric("Opposer", f"{timings.opposer:.2f}s")
        st.metric("Phase 1 Total (parallel)", f"{max(timings.advocate, timings.opposer):.2f}s")
        
        st.markdown("#### Phase 2: Rebuttals")
        st.metric("Advocate Rebuttal", f"{timings.advocate_rebuttal:.2f}s")
        st.metric("Opposer Rebuttal", f"{timings.opposer_rebuttal:.2f}s")
        st.metric("Phase 2 Total (parallel)", f"{max(timings.advocate_rebuttal, timings.opposer_rebuttal):.2f}s")
    
    with col2:
        st.markdown("#### Phase 3: Judge Panels")
        st.metric("Judge Panel 1", f"{timings.judge_panel1:.2f}s")
        st.metric("Judge Panel 2", f"{timings.judge_panel2:.2f}s")
        st.metric("Phase 3 Total (parallel)", f"{max(timings.judge_panel1, timings.judge_panel2):.2f}s")
        
        st.markdown("#### Summary")
        sequential_estimate = (
            timings.advocate + timings.opposer + 
            timings.advocate_rebuttal + timings.opposer_rebuttal + 
            timings.judge_panel1 + timings.judge_panel2
        )
        st.metric("Sequential Estimate", f"{sequential_estimate:.2f}s")
        st.metric("Parallel Actual", f"{timings.total:.2f}s")
        speedup = sequential_estimate / timings.total if timings.total > 0 else 0
        st.metric("Speedup", f"{speedup:.1f}x")
    
    st.markdown("---")
    st.markdown("#### Performance Notes")
    st.markdown("""
    - Phases 1, 2, and 3 run in parallel with configurable concurrency (default: 2 workers)
    - Actual parallel time = max of parallel tasks in each phase
    - Speedup depends on API latency variance and rate limits
    - Rate limiting may reduce effective parallelism
    """)


def render_export_tab(result: Dict, products: List[Product]):
    st.markdown("### Export Report")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        export_data = {
            "question": result["question"],
            "timestamp": datetime.now().isoformat(),
            "verdict": result["verdict"],
            "pro_scores": result["pro_scores"],
            "con_scores": result["con_scores"],
            "pro_arguments": result["pro_arguments"],
            "con_arguments": result["con_arguments"],
            "pro_rebuttal": result["pro_rebuttal"],
            "con_rebuttal": result["con_rebuttal"],
            "reasoning": result["reasoning"],
            "strongest_pro": result["strongest_pro"],
            "strongest_con": result["strongest_con"],
            "panels_agree": result["panels_agree"],
            "products": [{"name": p.name, "price": p.price, "source": p.source} for p in products],
        }
        st.download_button(
            "📥 Download JSON",
            data=json.dumps(export_data, indent=2),
            file_name=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col2:
        try:
            from fpdf import FPDF
            pdf_available = True
        except ImportError:
            pdf_available = False
        
        if pdf_available:
            pdf_bytes = create_pdf_report(DebateResult(
                id=generate_id(),
                timestamp=datetime.now().isoformat(),
                question=result["question"],
                products=[{"name": p.name, "price": p.price} for p in products],
                verdict=result["verdict"],
                pro_scores=result["pro_scores"],
                con_scores=result["con_scores"],
                pro_arguments=result["pro_arguments"],
                con_arguments=result["con_arguments"],
                pro_rebuttal=result["pro_rebuttal"],
                con_rebuttal=result["con_rebuttal"],
                reasoning=result["reasoning"],
                strongest_pro=result["strongest_pro"],
                strongest_con=result["strongest_con"],
                panels_agree=result["panels_agree"],
            ))
            if pdf_bytes:
                st.download_button(
                    "📄 Download PDF",
                    data=pdf_bytes,
                    file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
        else:
            st.button("📄 PDF (install fpdf2)", disabled=True, use_container_width=True)
            st.caption("Run: `pip install fpdf2`")
    
    with col3:
        if st.button("💾 Save to History", use_container_width=True):
            save_to_history(result, products)
            st.success("Saved!")

def save_to_history(result: Dict, products: List[Product]):
    entry = DebateResult(
        id=generate_id(),
        timestamp=datetime.now().isoformat(),
        question=result["question"],
        products=[{"name": p.name, "price": p.price, "source": p.source} for p in products],
        verdict=result["verdict"],
        pro_scores=result["pro_scores"],
        con_scores=result["con_scores"],
        pro_arguments=result["pro_arguments"],
        con_arguments=result["con_arguments"],
        pro_rebuttal=result["pro_rebuttal"],
        con_rebuttal=result["con_rebuttal"],
        reasoning=result["reasoning"],
        strongest_pro=result["strongest_pro"],
        strongest_con=result["strongest_con"],
        panels_agree=result["panels_agree"],
        mode="comparison" if st.session_state.comparison_mode else "single"
    )
    st.session_state.history.insert(0, entry.to_dict())
    save_history()

def render_history_sidebar():
    with st.sidebar:
        st.markdown("## History")
        
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.history = []
            save_history()
            st.rerun()
        
        if not st.session_state.history:
            st.caption("No analyses yet")
        else:
            for entry in st.session_state.history[:10]:
                with st.container(border=True):
                    st.markdown(f"""
                    <div class="history-item">
                        <div class="history-question">{entry['question'][:60]}{'...' if len(entry['question']) > 60 else ''}</div>
                        <div class="history-meta">{entry['timestamp'][:16]} • {entry['verdict'].upper()} • {sum(entry['pro_scores'].values()):.1f} vs {sum(entry['con_scores'].values()):.1f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    if st.button("Load", key=f"load_{entry['id']}", use_container_width=True):
                        st.session_state.current_result = entry
                        st.rerun()

def fetch_product_from_url(url: str) -> Optional[Product]:
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        return None
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "lxml")
        
        source = "generic"
        if "amazon." in url:
            source = "amazon"
        elif "shopify" in resp.text.lower() or "myshopify" in url:
            source = "shopify"
        
        selectors = {
            "amazon": {"image": ["#landingImage", "#imgBlkFront", ".a-dynamic-image"], "title": ["#productTitle"], "price": [".a-price .a-offscreen", "#priceblock_ourprice"]},
            "shopify": {"image": [".product-media-container img", ".product-featured-media img"], "title": ["h1.product-title", ".product__title"], "price": [".price-item--regular", ".product-price"]},
            "generic": {"image": ["img[src*='product']", ".product-image", "main img"], "title": ["h1", ".product-title", "[itemprop='name']"], "price": [".price", "[itemprop='price']", ".product-price"]},
        }.get(source, {"image": ["main img"], "title": ["h1"], "price": [".price"]})
        
        title = ""
        for sel in selectors["title"]:
            el = soup.select_one(sel)
            if el:
                title = el.get_text(strip=True)
                break
        
        price = ""
        for sel in selectors["price"]:
            el = soup.select_one(sel)
            if el:
                price = el.get_text(strip=True)
                break
        
        images = []
        seen = set()
        for sel in selectors["image"]:
            els = soup.select(sel)
            for el in els:
                src = el.get("src") or el.get("data-src") or el.get("data-large")
                if src and src.startswith("http"):
                    if src.startswith("{"):
                        try:
                            data = json.loads(src)
                            src = list(data.keys())[0] if data else ""
                        except:
                            continue
                    if src and src not in seen:
                        seen.add(src)
                        try:
                            img_resp = requests.get(src, headers=headers, timeout=10)
                            if img_resp.status_code == 200:
                                mime = img_resp.headers.get("Content-Type", "image/jpeg")
                                b64 = base64.b64encode(img_resp.content).decode("utf-8")
                                images.append({"b64": b64, "mime": mime, "url": src})
                                if len(images) >= 5:
                                    break
                        except:
                            pass
            if len(images) >= 5:
                break
        
        product = Product(name=title, price=price, images=images, url=url, source=source, metadata={"fetched_at": datetime.now().isoformat()})
        return product if (title or images) else None
    except Exception:
        return None

# =============================================================================
# MAIN APP
# =============================================================================

def main():
    st.set_page_config(
        page_title="AI Judiciary Vision",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    inject_css()
    init_session_state()
    render_header()
    render_history_sidebar()
    
    # Main layout
    left_col, right_col = st.columns([1.1, 1], gap="large")
    
    with left_col:
        with st.container(border=True):
            render_product_section()
        
        with st.container(border=True):
            question, price, currency = render_question_section()
        
        can_run = bool(question and st.session_state.products and any(p.images for p in st.session_state.products))
        
        if st.button(
            "🚀 Start Analysis" + (" (Compare)" if st.session_state.comparison_mode else ""),
            type="primary",
            disabled=not can_run or st.session_state.debate_running,
            use_container_width=True
        ):
            if can_run:
                st.session_state.debate_running = True
                
                update_progress = render_debate_progress()
                
                try:
                    primary_product = st.session_state.products[0]
                    primary_image = primary_product.images[0] if primary_product.images else None
                    
                    if not primary_image:
                        st.error("No images available")
                        st.session_state.debate_running = False
                        st.rerun()
                    
                    full_question = question
                    if price:
                        full_question += f"\n\nPrice context: {currency}{price:.2f}"
                    
                    if st.session_state.comparison_mode and len(st.session_state.products) >= 2:
                        with st.spinner("Running comparison analysis..."):
                            result = run_comparison_debate(full_question, st.session_state.products[0], st.session_state.products[1], update_progress, max_workers=2)
                    else:
                        with st.spinner("Running analysis..."):
                            result = run_debate(full_question, primary_image["b64"], primary_image["mime"], update_progress, max_workers=2)
                    
                    st.session_state.current_result = result
                    save_to_history(result, st.session_state.products)
                    st.session_state.debate_running = False
                    time.sleep(0.3)
                    st.rerun()
                    
                except Exception as e:
                    st.session_state.debate_running = False
                    st.error(f"Analysis failed: {str(e)}")
    
    with right_col:
        with st.container(border=True):
            if st.session_state.current_result:
                render_results(st.session_state.current_result, st.session_state.products)
            else:
                st.markdown("""
                <div class="empty-state animate-in">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                        <path d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
                    </svg>
                    <h3 style="margin: 0 0 0.5rem; color: var(--text-secondary);">Ready to Analyze</h3>
                    <p style="margin: 0; max-width: 320px; margin-left: auto; margin-right: auto;">
                        Upload product images or fetch from a URL, ask your question, and watch AI agents debate the answer.
                    </p>
                </div>
                """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: var(--text-muted); font-size: 0.8rem; padding: 1.5rem;">
        ⚖️ AI Judiciary Vision • Blinded dual-judge panels with swap test • 
        <a href="https://github.com/sibirajan777/ai-judiciary" target="_blank" style="color: var(--accent); text-decoration: none;">GitHub</a>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()