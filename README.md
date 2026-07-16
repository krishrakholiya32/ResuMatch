# 🎯 FitCheck

A Streamlit web app that scores how well a resume matches a job description — **No Fit / Potential Fit / Good Fit** — using a self-trained DistilBERT classifier.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red)
![License](https://img.shields.io/badge/License-MIT-green)

> **Status: trained and ready to run.** See [Model Performance](#-model-performance) below —
> including a real data-leakage finding worth reading before trusting the headline number.

---

## ✨ Features

- 📄 Upload a resume (PDF / DOCX / TXT), paste a job description
- 🎯 3-class fit score with confidence bars
- ⚡ Runs on CPU — no GPU needed for inference

---

## 🧠 Scope Note

Fine-tuned on [cnamuangtoun/resume-job-description-fit](https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit)
(8,000 real resume-JD pairs, 3-class labels). This is a text-similarity signal from a trained
classifier, not a hiring decision or a guarantee of interview success — always tailor your
actual application.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| [Streamlit](https://streamlit.io) | Web app framework |
| DistilBERT (ONNX, int8-quantized) | Resume/JD fit classification |
| [ONNX Runtime](https://onnxruntime.ai) | CPU inference |
| [tokenizers](https://github.com/huggingface/tokenizers) | Lightweight tokenization (no torch at inference) |
| pdfplumber / python-docx | Resume text extraction |
| PyTorch + transformers | Training (Kaggle GPU) |

---

## 📈 Model Performance

| Metric | Value |
|--------|-------|
| Test macro-F1 (deployed int8 model) | **0.372** |
| Test macro-F1 (fp32, pre-quantization) | 0.387 |
| Val macro-F1 (during training — inflated, see below) | 0.647 |
| Naive "always predict majority class" baseline | ~0.22 |
| Model | DistilBERT, int8-quantized ONNX (67MB) |
| Training data | 5,304 rows (Kaggle CPU, ~99 min/epoch, 4 epochs) |
| Test data | 1,759 rows, held out by the dataset's original authors |

**A genuine finding, not just a caveat:** during training, validation macro-F1 climbed to 0.647 —
but the real held-out test score is only 0.372. The gap isn't a bug; it's leakage baked into how
the source dataset is structured. The published `train.csv` only contains **279 unique job
descriptions** reused across 5,304 rows (each JD paired with many different resumes). My val split
was a random 15% slice of `train.csv`, so **231 of its 232 unique job descriptions had already been
seen during training** (just paired with a different resume) — val was measuring "generalize to a
new resume against an already-seen JD," an easier task than the real one. The dataset's official
`test.csv`, by contrast, shares **zero job descriptions** with train — a genuinely unseen-JD
evaluation, which is what the deployed model actually faces (a user pasting a JD it's never seen).

0.372 macro-F1 is real, above-baseline signal (~1.7x the naive majority-class baseline) but modest
— treat this as an honest first pass, not a polished production classifier. A proper fix would
re-split validation by job description (group-based split, no JD overlap with train) so the
training-time metric reflects true generalization instead of this optimistic number.

---

## 🚀 Getting Started

### 1. Train the model (Kaggle notebook, GPU + internet enabled)

```bash
cd training/scripts
python prepare_dataset.py --output_dir /kaggle/working/data
python train.py --config ../configs/train_config.yaml
python evaluate.py --config ../configs/train_config.yaml \
    --checkpoint /kaggle/working/runs/resumefit_distilbert_v1/checkpoints/best
python export_model.py \
    --checkpoint /kaggle/working/runs/resumefit_distilbert_v1/checkpoints/best \
    --output ../../models/resume_fit_distilbert.onnx
```

Download the resulting `models/resume_fit_distilbert.onnx`, `models/tokenizer.json`, and
`models/labels.json` back into this repo.

### 2. Run the app

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## ⚙️ How It Works

1. Upload a resume — text is extracted (PDF/DOCX/TXT) and truncated to the first 350 tokens
   (resumes front-load the most relevant info: summary, recent experience)
2. Paste a job description — truncated to the first 150 tokens
3. Both are combined as `[CLS] resume [SEP] jd [SEP]` and run through the ONNX model
4. The 3-class probability distribution is shown, with the top class as the verdict

---

## 📝 License

MIT — feel free to use, modify, and share.
