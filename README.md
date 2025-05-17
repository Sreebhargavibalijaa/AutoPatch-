# AutoPatch+

**AutoPatch+** is a hallucination-aware verification and self-correction framework for Large Language Models (LLMs). It integrates retrieval-augmented generation (RAG), hallucination detection, and correction pipelines using interpretable models like Neural Additive Models (NAMs) to ensure response trustworthiness across QA and summarization tasks.

---

## 🔍 Key Features

- ✅ **Hallucination Detection**: Uses both training-based (NAMs, LIME) and sampling-based (SelfCheckGPT) methods.
- 🔁 **Self-Correction**: Automatically rewrites hallucinated outputs using retrieval context.
- 📊 **Interpretable Verification**: Uses interpretable models (e.g., NAMs) to justify answer correctness.
- 🔗 **Modular RAG Pipeline**: Supports various retrievers, vector stores, and LLM backends.
- 🌐 **Web UI**: Streamlit-based demo app to visualize detection and correction in real time.

---

## 🏗️ Architecture

User Input --> RAG Pipeline --> LLM Output
|
Hallucination Detector
|
┌──────────┴──────────┐
Detected False Detected True
→ Accept → → Rewrite →

markdown
Copy
Edit

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- [transformers](https://github.com/huggingface/transformers)
- [streamlit](https://streamlit.io/)
- [scikit-learn](https://scikit-learn.org/)
- [openai](https://pypi.org/project/openai/)
- [langchain](https://github.com/langchain-ai/langchain)

Install dependencies:

```bash
pip install -r requirements.txt
Run the App
bash
Copy
Edit
streamlit run appfinal.py
Or for a backend evaluation without UI:

bash
Copy
Edit
python evaluate_hallucination.py
🧠 Hallucination Detection Modes
SelfCheck Sampling: Compare multiple sampled LLM outputs for consistency.

NAM-Based Detector: Trains interpretable models on labeled hallucination data.

Cluster-Aware Verifier: Uses answer clustering to assign best detector dynamically.

📁 Repository Structure
bash
Copy
Edit
.
├── appfinal.py                 # Streamlit frontend
├── detector/
│   ├── selfcheck.py            # Sampling-based hallucination detector
│   └── nam_verifier.py         # Interpretable NAM-based verifier
├── rag_pipeline/
│   ├── retriever.py            # Vector/keyword hybrid search
│   └── context_builder.py      # Document retrieval interface
├── rewrite/
│   └── rewriter.py             # LLM-based rewriting for detected hallucinations
├── data/                       # Datasets & QA examples
├── eval/                       # Evaluation scripts & metrics
└── README.md
💡 Example Use Case
Input a factual question.

AutoPatch+ retrieves relevant documents via RAG.

LLM answers the question.

Detector checks if answer is hallucinated.

If hallucinated, the pipeline rewrites using facts.

🧪 Citation
If you use AutoPatch+ in your research, please cite:

bibtex
Copy
Edit
@inprogress{autopatch2025,
  title={AutoPatch+: A Hallucination-Aware Verification and Correction Framework for Large Language Models},
  author={Sree Bhargavi Balija},
  year={2025},
  note={Under submission}
}
🤝 Contributing
We welcome contributions! Please open issues or submit PRs for improvements. See CONTRIBUTING.md for details.
