"""ResuMatch — Resume <-> Job Description fit checker, Streamlit app.

Scores how well a resume matches a job description across 3 classes
(No Fit / Potential Fit / Good Fit) using a self-trained DistilBERT
classifier, fine-tuned on the cnamuangtoun/resume-job-description-fit
dataset (8,000 real resume-JD pairs).

Scope: text-based fit scoring from a trained classifier — not a
guarantee of interview success, and not a substitute for a human reviewer.
"""

import json
from pathlib import Path

import numpy as np
import streamlit as st
from tokenizers import Tokenizer

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


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def predict(resume_text: str, jd_text: str) -> list[tuple[str, float]]:
    """Return (label, probability) pairs in label order."""
    session = _load_session()
    tokenizer = _load_tokenizer()
    labels = _load_labels()

    cls_id = tokenizer.token_to_id("[CLS]")
    sep_id = tokenizer.token_to_id("[SEP]")

    resume_ids = tokenizer.encode(resume_text, add_special_tokens=False).ids[:RESUME_MAX_TOKENS]
    jd_ids = tokenizer.encode(jd_text, add_special_tokens=False).ids[:JD_MAX_TOKENS]
    input_ids = [cls_id] + resume_ids + [sep_id] + jd_ids + [sep_id]

    input_ids_arr = np.array([input_ids], dtype=np.int64)
    attention_mask = np.ones_like(input_ids_arr)

    logits = session.run(["logits"], {"input_ids": input_ids_arr, "attention_mask": attention_mask})[0][0]
    probs = _softmax(logits)
    return list(zip(labels, (float(p) for p in probs)))


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
        "- Only reads the first ~350 tokens of your resume and ~150 tokens of the JD — "
        "put your most relevant info up front\n"
        "- A text-similarity signal, not a hiring decision — always tailor your actual application\n"
        "- Trained on English-language resumes/JDs only"
    )

col1, col2 = st.columns(2)
with col1:
    resume_file = st.file_uploader("Upload your resume", type=["pdf", "docx", "txt"])
with col2:
    jd_text = st.text_area("Paste the job description", height=200, placeholder="Paste the full JD text here…")

check = st.button("🎯 Check Fit", use_container_width=True, type="primary",
                   disabled=not (resume_file and jd_text.strip()))

if check:
    resume_text = _extract_text(resume_file)
    if not resume_text.strip():
        st.error("Couldn't extract any text from that resume file.")
    else:
        with st.spinner("Scoring fit…"):
            predictions = predict(resume_text, jd_text)

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
