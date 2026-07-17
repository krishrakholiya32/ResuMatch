"""ResuMatch — Resume <-> Job Description fit checker, Streamlit app.

Scores how well a resume matches a job description across 3 classes
(No Fit / Potential Fit / Good Fit) using a self-trained DistilBERT
classifier, fine-tuned on the cnamuangtoun/resume-job-description-fit
dataset (8,000 real resume-JD pairs).

Scope: text-based fit scoring from a trained classifier — not a
guarantee of interview success, and not a substitute for a human reviewer.
"""

import json
import re
from pathlib import Path

import numpy as np
import streamlit as st
from tokenizers import Tokenizer

_FOCUS_STOPWORDS = {
    "a", "an", "the", "is", "it", "only", "for", "and", "or", "of", "to", "in", "on",
    "area", "areas", "from", "this", "that", "with", "please", "just", "also",
    "focus", "consider", "part", "parts", "section", "sections", "relevant",
}

MODEL_PATH = Path("models/resume_fit_distilbert.onnx")
TOKENIZER_PATH = Path("models/tokenizer.json")
LABELS_PATH = Path("models/labels.json")

# Must match training/configs/train_config.yaml — resumes front-load the most
# relevant info, so a fixed per-side token budget beats naive tail truncation.
RESUME_MAX_TOKENS = 350
JD_MAX_TOKENS = 150


@st.cache_resource(show_spinner=False)
def _load_session():
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(MODEL_PATH), sess_options=opts, providers=["CPUExecutionProvider"])


@st.cache_resource(show_spinner=False)
def _load_tokenizer() -> Tokenizer:
    return Tokenizer.from_file(str(TOKENIZER_PATH))


@st.cache_resource(show_spinner=False)
def _load_labels() -> list[str]:
    with open(LABELS_PATH) as f:
        return json.load(f)


def _extract_text(uploaded) -> str:
    suffix = Path(uploaded.name).suffix.lower()
    if suffix == ".pdf":
        import pdfplumber
        with pdfplumber.open(uploaded) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    if suffix == ".docx":
        from docx import Document
        doc = Document(uploaded)
        return "\n".join(p.text for p in doc.paragraphs)
    return uploaded.read().decode("utf-8", errors="ignore")


def _filter_jd_by_focus(jd_text: str, focus: str) -> tuple[str, bool]:
    """Keyword filter, not language understanding: pulls out JD sentences/lines containing
    any word from `focus` (e.g. "AI ML and Python" -> keywords {ai, ml, python}). Not an LLM,
    so it can't interpret nuanced instructions -- only literal keyword matches. Returns
    (filtered_text, matched) -- falls back to the full JD if nothing matched, since a
    misspelled or obscure keyword shouldn't silently produce an empty/broken JD.
    """
    keywords = [w for w in re.findall(r"[A-Za-z][A-Za-z0-9+#.]*", focus) if w.lower() not in _FOCUS_STOPWORDS]
    if not keywords:
        return jd_text, False

    pattern = re.compile("|".join(re.escape(k) for k in keywords), re.IGNORECASE)
    lines = re.split(r"(?<=[.!?])\s+|\n+", jd_text)
    matched_lines = [line for line in lines if pattern.search(line)]

    if not matched_lines:
        return jd_text, False
    return "\n".join(matched_lines), True


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def _chunk(ids: list[int], size: int) -> list[list[int]]:
    if not ids:
        return [[]]
    return [ids[i:i + size] for i in range(0, len(ids), size)]


def predict(resume_text: str, jd_text: str) -> list[tuple[str, float]]:
    """Return (label, probability) pairs in label order.

    The model was trained on a fixed 350/150-token front slice of each document (DistilBERT's
    512-token limit leaves no room to raise that budget directly). Rather than silently
    dropping everything past that slice, both documents are split into token-budget-sized
    chunks and every resume-chunk/JD-chunk pairing is scored, then averaged -- so the full
    text is actually read, not just the first ~350/150 tokens. Chunks past the first one are
    somewhat out-of-distribution for the model (it never saw "resume, paragraph 3" during
    training), so treat this as a genuine mitigation, not a full fix for the token limit.
    """
    session = _load_session()
    tokenizer = _load_tokenizer()
    labels = _load_labels()

    cls_id = tokenizer.token_to_id("[CLS]")
    sep_id = tokenizer.token_to_id("[SEP]")

    resume_chunks = _chunk(tokenizer.encode(resume_text, add_special_tokens=False).ids, RESUME_MAX_TOKENS)
    jd_chunks = _chunk(tokenizer.encode(jd_text, add_special_tokens=False).ids, JD_MAX_TOKENS)

    all_probs = []
    for i in range(max(len(resume_chunks), len(jd_chunks))):
        r_ids = resume_chunks[i % len(resume_chunks)]
        j_ids = jd_chunks[i % len(jd_chunks)]
        input_ids = [cls_id] + r_ids + [sep_id] + j_ids + [sep_id]
        input_ids_arr = np.array([input_ids], dtype=np.int64)
        attention_mask = np.ones_like(input_ids_arr)
        logits = session.run(["logits"], {"input_ids": input_ids_arr, "attention_mask": attention_mask})[0][0]
        all_probs.append(_softmax(logits))

    avg_probs = np.mean(all_probs, axis=0)
    return list(zip(labels, (float(p) for p in avg_probs)))


def _verdict_style(top_label: str):
    return {"No Fit": ("error", "🚨"), "Potential Fit": ("warning", "🟡"), "Good Fit": ("success", "✅")}[top_label]


# ── UI ──────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="ResuMatch", page_icon="🎯", layout="centered")

st.title("🎯 ResuMatch — Resume/JD Fit Scorer")
st.markdown(
    "Upload your resume and paste a job description to get a **fit score** "
    "(No Fit / Potential Fit / Good Fit) before you apply."
)

with st.expander("ℹ️ About this model & limitations"):
    st.markdown(
        "- Self-trained **DistilBERT** classifier fine-tuned on 8,000 real resume-JD pairs "
        "([cnamuangtoun/resume-job-description-fit](https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit))\n"
        "- 3-class fit scoring with class-weighted loss to handle label imbalance\n\n"
        "**Important limitations:**\n"
        "- The model itself only sees 350 resume tokens / 150 JD tokens per pass (DistilBERT's "
        "512-token limit) — the app works around this by chunking your full resume/JD and "
        "averaging across chunks, but chunks past the first are somewhat out-of-distribution "
        "for the model (it was only trained on front-of-document text)\n"
        "- A text-similarity signal, not a hiring decision — always tailor your actual application\n"
        "- Trained on English-language resumes/JDs only\n"
        "- The \"focus on\" field is **keyword matching, not an instruction the model understands** "
        "— this isn't an LLM, so it can't reason about what you type. It just keeps JD lines "
        "containing those literal words (e.g. \"AI ML Python\" keeps lines mentioning those terms)"
    )

col1, col2 = st.columns(2)
with col1:
    resume_file = st.file_uploader("Upload your resume", type=["pdf", "docx", "txt"])
with col2:
    jd_file = st.file_uploader("Upload the JD (optional)", type=["pdf", "docx", "txt"])
    jd_text = st.text_area("...or paste the job description", height=150, placeholder="Paste the full JD text here…")

focus_text = st.text_input(
    "Focus on specific areas (optional)",
    placeholder="e.g. AI, ML, Python — keeps only JD lines mentioning these words",
)

check = st.button("🎯 Analyze", use_container_width=True, type="primary")

if check:
    if not resume_file:
        st.warning("Upload a resume first.")
    elif not (jd_file or jd_text.strip()):
        st.warning("Upload a JD file or paste the job description text.")
    else:
        resume_text = _extract_text(resume_file)
        jd_source = "\n\n".join(t for t in [_extract_text(jd_file) if jd_file else "", jd_text] if t.strip())
        if not resume_text.strip():
            st.error("Couldn't extract any text from that resume file.")
        elif not jd_source.strip():
            st.error("Couldn't extract any text from that JD file.")
        else:
            if focus_text.strip():
                jd_source, matched = _filter_jd_by_focus(jd_source, focus_text)
                if matched:
                    st.caption(f"🔎 Filtered JD to lines matching: {focus_text.strip()}")
                else:
                    st.caption(f"⚠️ No JD lines matched \"{focus_text.strip()}\" — scoring against the full JD instead")

            with st.spinner("Scoring fit…"):
                predictions = predict(resume_text, jd_source)

            st.divider()

            top_label, top_conf = max(predictions, key=lambda p: p[1])
            kind, emoji = _verdict_style(top_label)
            getattr(st, kind)(f"### {emoji} {top_label} — {top_conf:.1%} confidence")

            st.subheader("Class probabilities")
            for label, conf in predictions:
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.progress(conf, text=label)
                with col_b:
                    st.write(f"**{conf:.1%}**")

st.divider()
st.caption(
    "Self-trained DistilBERT + ONNX Runtime · "
    "[GitHub](https://github.com/krishrakholiya32/ResuMatch)"
)
