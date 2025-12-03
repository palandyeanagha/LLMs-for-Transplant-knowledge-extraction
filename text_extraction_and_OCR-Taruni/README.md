# Text Extraction and OCR Pipeline

This folder contains the text extraction and OCR (Optical Character Recognition) pipeline developed for extracting text from scientific research papers (PDFs) related to organ transplantation studies.

## Folder Structure

```
text_extraction_and_OCR-Taruni/
├── README.md
├── Text_Extraction_Tesseract_OCR_v1_liver.ipynb    # Version 1 notebook
├── Text_Extraction_DeepSeek_OCR_v2.ipynb           # Version 2 notebook
├── Text_Extraction_v3.ipynb                         # Version 3 notebook (Final)
├── pdf_metadata_v1_liver.csv                        # Metadata for V1
├── metadata_new_v2.csv                              # Metadata for V2
├── metadata_new_final_v3.csv                        # Metadata for V3 (Final)
├── extracted_text_v1_liver/                         # Extracted text files (V1)
│   └── [1764 .txt files]
└── text_extraction/                                 # Extracted text files (V2/V3)
    └── [3746 .txt files]
```

---

## Version History

### Version 1 (`Text_Extraction_Tesseract_OCR_v1_liver.ipynb`)

**Scope:** Liver Transplant Papers Only

This was the initial version focused exclusively on **liver transplant** research papers.

#### Features:
- Basic text extraction using `pypdfium2`
- Tesseract OCR support for scanned documents
- Simple metadata extraction

#### Metadata File: `pdf_metadata_v1_liver.csv`

| Column | Description |
|--------|-------------|
| `pdf_title` | Title extracted from the PDF |
| `doi` | Digital Object Identifier |
| `file_size_mb` | File size in megabytes |
| `text_length` | Length of extracted text (characters) |
| `is_scanned` | Boolean - whether the PDF is scanned |
| `needs_ocr` | Boolean - whether OCR is required |
| `extraction_method` | Method used for extraction |

#### Output:
- **1,764 text files** in `extracted_text_v1_liver/`

---

### Version 2 (`Text_Extraction_DeepSeek_OCR_v2.ipynb`)

**Scope:** All Transplant Types (Liver, Lung, Heart, Kidney)

Expanded to include all organ transplant categories.

#### Key Changes from V1:
- Extended to multiple transplant categories (liver, lung, heart, kidney)
- Attempted integration with **DeepSeek OCR** from DeepSeek documentation via Google Colab
- Added `category` field to metadata to identify transplant type

#### Important Finding:
> **No scanned PDFs were detected** in the dataset, so OCR functionality was not utilized in this version.

#### Metadata File: `metadata_new_v2.csv`

| Column | Description |
|--------|-------------|
| `pdf_title` | Title extracted from the PDF |
| `doi` | Digital Object Identifier |
| `file_size_mb` | File size in megabytes |
| `text_length` | Length of extracted text (characters) |
| `is_scanned` | Boolean - whether the PDF is scanned |
| `needs_ocr` | Boolean - whether OCR is required |
| `extraction_method` | Method used for extraction |
| `year` | Publication year |
| `citation_count` | Number of citations |
| `publication` | Publication/Journal name |
| `category` | Transplant category (liver/lung/heart/kidney) |

---

### Version 3 (`Text_Extraction_v3.ipynb`) **Final Version**

**Scope:** All Transplant Types with Enhanced Extraction & Metadata

This is the most comprehensive and refined version with intelligent extraction methods and enriched metadata.

#### Key Improvements:

##### 1. Multi-Method PDF Extraction
Uses **3 different PDF extraction libraries** and dynamically selects the best result:

| Library | Description |
|---------|-------------|
| **pypdfium2** | Google's PDFium-based extractor |
| **PyPDF2** | Pure Python PDF library |
| **PyMuPDF (Fitz)** | MuPDF-based extractor with high accuracy |

The system compares extraction results from all three methods and selects the one that produces the **best quality text** (based on text length and content quality).

##### 2. Enhanced Metadata via External APIs
Integrates with multiple scholarly APIs to enrich metadata:

| API | Purpose |
|-----|---------|
| **Semantic Scholar API** | Citation counts, publication details |
| **CrossRef API** | DOI resolution, publication metadata |
| **OpenAlex API** | Open scholarly data, year of publication |

#### Metadata File: `metadata_new_final_v3.csv`

| Column | Description |
|--------|-------------|
| `pdf_title` | Title extracted from the PDF |
| `doi` | Digital Object Identifier |
| `file_size_mb` | File size in megabytes |
| `text_length` | Length of extracted text (characters) |
| `is_scanned` | Boolean - whether the PDF is scanned |
| `needs_ocr` | Boolean - whether OCR is required |
| `extraction_method` | Method used for extraction |
| `year` | Publication year (from APIs) |
| `citation_count` | Number of citations (from APIs) |
| `publication` | Publication/Journal name (from APIs) |
| `category` | Transplant category (liver/lung/heart/kidney/unassigned) |

#### Output:
- **3,746 text files** in `text_extraction/`
- **3,845 metadata records** in `metadata_new_final_v3.csv`

---

## Version Comparison Summary

| Feature | V1 | V2 | V3 |
|---------|----|----|-----|
| **Transplant Types** | Liver only | All (Liver, Lung, Heart, Kidney) | All (Liver, Lung, Heart, Kidney) |
| **PDF Count** | ~1,764 | ~3,769 | ~3,845 |
| **Extraction Methods** | pypdfium2 | pypdfium2 | pypdfium2 + PyPDF2 + PyMuPDF |
| **OCR Support** | Tesseract | DeepSeek (unused) | - |
| **Scanned PDFs Found** | Yes | No | - |
| **API Integration** | None | Partial | Full (Semantic Scholar, CrossRef, OpenAlex) |
| **Metadata Fields** | 7 | 11 | 11 |

---

## Dependencies

```python
# Core PDF Libraries
pypdfium2      # Google's PDFium wrapper
PyMuPDF        # aka 'fitz' - MuPDF wrapper
PyPDF2         # Pure Python PDF library

# Data Processing
pandas
tqdm

# API Requests
requests

# OCR (V1 only)
pytesseract    # Tesseract OCR wrapper
```

---

## Usage Notes

1. **Run notebooks in Google Colab** for best compatibility with Google Drive integration
2. **V3 is recommended** for new extractions due to its multi-method approach
3. Metadata CSV files can be used for filtering and analysis of the paper corpus
4. Text files are named using DOI format: `{doi_with_underscores}.txt`

---

## Author

**Taruni** - Text Extraction and OCR Pipeline Development

