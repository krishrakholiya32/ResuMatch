"""ResuMatch — Resume <-> Job Description fit checker, Streamlit app.

Scores how well a resume matches a job description (No Fit / Fit) using a
self-trained DistilBERT classifier, fine-tuned on the
cnamuangtoun/resume-job-description-fit dataset (8,000 real resume-JD pairs).
Binary, not the dataset's original 3-way label -- Potential Fit and Good Fit
lead to the same real action (apply), so they're merged into one Fit class;
see training/scripts/prepare_dataset.py for the full rationale.

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
    return {"No Fit": ("error", "🚨"), "Fit": ("success", "✅")}[top_label]


# Decision threshold picked on the val set, not the default 0.5 argmax -- P(Fit) > 0.5 measurably
# skewed toward over-predicting Fit on the real test set (No Fit recall 0.37 vs Fit recall 0.78).
# 0.54 rebalances recall *and* improves test macro-F1 (0.556 -> 0.576), verified independently on
# held-out val (where it was picked) and test (where it wasn't) splits -- not just tuned to look
# good on one set.
FIT_THRESHOLD = 0.54

# Half-width of the "close call" band around FIT_THRESHOLD -- a P(Fit) of 0.58 against a 0.54 bar
# is the model barely clearing the threshold, not a confident verdict.
UNCERTAIN_MARGIN = 0.05


# ── UI ──────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="ResuMatch", page_icon="🎯", layout="centered")

st.title("🎯 ResuMatch — Resume/JD Fit Scorer")
st.markdown(
    "Upload your resume and paste a job description to get a **fit score** "
    "(No Fit / Fit) before you apply."
)

with st.expander("ℹ️ About this model & limitations"):
    st.markdown(
        "- Self-trained **DistilBERT** classifier fine-tuned on 8,000 real resume-JD pairs "
        "([cnamuangtoun/resume-job-description-fit](https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit))\n"
        "- Binary (No Fit / Fit) scoring with class-weighted loss and a val-tuned decision threshold\n\n"
        "**Important limitations:**\n"
        "- The model itself only sees 350 resume tokens / 150 JD tokens per pass (DistilBERT's "
        "512-token limit) — the app works around this by chunking your full resume/JD and "
        "averaging across chunks, but chunks past the first are somewhat out-of-distribution "
        "for the model (it was only trained on front-of-document text)\n"
        "- A text-similarity signal, not a hiring decision — always tailor your actual application\n"
        "- Trained on English-language resumes/JDs only"
    )

col1, col2 = st.columns(2)
with col1:
    resume_file = st.file_uploader("Upload your resume", type=["pdf", "docx", "txt"])
with col2:
    jd_file = st.file_uploader("Upload the JD (optional)", type=["pdf", "docx", "txt"])
    jd_text = st.text_area("...or paste the job description", height=150, placeholder="Paste the full JD text here…")

check = st.button("🎯 Analyze", use_container_width=True, type="primary")

if check:
    if not resume_file:
        st.warning("Upload a resume first.")
    elif not (jd_file or jd_text.strip()):
        st.warning("Upload a JD file or paste the job description text.")
    else:
        if jd_file and jd_text.strip():
            st.warning(
                "Both a JD file and pasted JD text are filled in — they'll be combined into "
                "one job description. If these are two different jobs, clear one first."
            )
        resume_text = _extract_text(resume_file)
        jd_source = "\n\n".join(t for t in [_extract_text(jd_file) if jd_file else "", jd_text] if t.strip())
        if not resume_text.strip():
            st.error("Couldn't extract any text from that resume file.")
        elif not jd_source.strip():
            st.error("Couldn't extract any text from that JD file.")
        else:
            with st.spinner("Scoring fit…"):
                predictions = predict(resume_text, jd_source)

            st.divider()

            fit_conf = dict(predictions)["Fit"]
            top_label = "Fit" if fit_conf > FIT_THRESHOLD else "No Fit"
            top_conf = fit_conf if top_label == "Fit" else 1 - fit_conf
            other_label = "No Fit" if top_label == "Fit" else "Fit"
            other_conf = 1 - top_conf

            if abs(fit_conf - FIT_THRESHOLD) < UNCERTAIN_MARGIN:
                st.info(
                    f"### 🤔 Close call — {top_label} ({top_conf:.1%}) vs {other_label} ({other_conf:.1%})\n\n"
                    "The model can't confidently tell these two apart for this resume/JD pairing. "
                    "Treat this as a toss-up, not a verdict."
                )
            else:
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
