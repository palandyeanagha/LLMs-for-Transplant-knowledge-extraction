# LLMs for Transplant Knowledge Extraction

> **A bibliometrics-enhanced RAG pipeline for transparent, citation-anchored knowledge synthesis in transplantation science. Built in collaboration with NYU Langone Health.**

## 🎯 Problem

Medical researchers reviewing transplantation literature face a growing challenge: the volume of published studies far exceeds what any individual can read and synthesize. Existing LLM-based summarization tools often **hallucinate citations** or generate claims without traceable evidence — a critical problem in clinical research where every claim must be verifiable.

This project builds a RAG (Retrieval-Augmented Generation) pipeline that produces **citation-anchored summaries** of transplant research, where every generated claim is tied to a specific source document.

## 🏗️ Approach

The pipeline combines bibliometric analysis with modern retrieval and generation techniques:

```
Scientific Papers (PDFs)
        ↓
   Text Extraction & OCR
        ↓
   RPYS-based Bibliometric Filtering (identify seminal works)
        ↓
   Chunking & Embedding (MiniLM)
        ↓
   Dense Retrieval + Cross-Encoder Reranking
        ↓
   Citation-Aware Prompting → LLM Generation
        ↓
   NLI-based Validation (faithfulness checking)
        ↓
   Citation-Anchored Summary with Source Attribution
```

### Key Design Decisions

- **RPYS (Reference Publication Year Spectroscopy)** filters the corpus to surface historically significant papers, improving retrieval relevance
- **Cross-encoder reranking** after initial dense retrieval significantly improves precision
- **Citation-aware prompting** instructs the LLM to attribute every claim to a specific retrieved chunk
- **NLI-based validation** post-generation checks whether generated statements are entailed by their cited sources

## 📊 Key Results

| Model | Faithfulness | Citation Accuracy |
|-------|-------------|-------------------|
| **Qwen2.5-32B-Instruct** | **89.4%** | **67.7%** |
| GPT-4o | Baseline comparison | Baseline comparison |

- Qwen2.5-32B-Instruct achieved higher faithfulness and citation accuracy than GPT-4o on this task
- Evaluation used **RAGAS** framework + custom citation verification + NLI-based validation
- The pipeline successfully reduces hallucinations by grounding generation in retrieved evidence

## 📁 Repository Structure

```
├── text_extraction_and_OCR/    # PDF processing and text extraction
├── chunking/                   # Document chunking strategies
├── code_files/                 # Core pipeline code (retrieval, generation, evaluation)
├── Summary and Evaluation/     # Output summaries and evaluation results
└── README.md
```

## 🛠️ Tech Stack

- **Retrieval:** MiniLM (sentence-transformers), cross-encoder reranking
- **Generation:** Qwen2.5-32B-Instruct, GPT-4o
- **Evaluation:** RAGAS, custom NLI-based citation validator
- **Bibliometrics:** RPYS-based publication filtering
- **OCR:** Tesseract / PDF text extraction
- **Languages:** Python, Jupyter Notebooks

## 🔗 Context

This project was conducted as a **research collaboration with NYU Langone Health** (Sep–Dec 2025), focused on applying LLM-powered knowledge synthesis to clinical transplantation literature.

## 👥 Team

Built collaboratively as part of NYU's Capstone research project. Contributors include team members focused on text extraction/OCR, chunking strategies, and summarization evaluation.
