# Resume / Candidate Screening System

## Project Overview

This project is a Machine Learning and Natural Language Processing (NLP) based Resume Screening System 

The system automatically extracts resume text from CSV and PDF files, preprocesses the text using NLP techniques, compares resumes with a job description using TF-IDF Vectorization and Cosine Similarity, ranks candidates based on relevance, extracts skills, identifies missing skills, and generates candidate ranking reports with visualizations.

## Objectives

- Read resume data from CSV and PDF files
- Clean and preprocess resume text
- Extract technical skills using NLP
- Parse job descriptions
- Calculate resume-job similarity
- Rank candidates
- Identify missing skills
- Visualize candidate comparison

---

## Features

- Resume text cleaning and preprocessing
- Skill extraction using NLP
- Job description parsing
- Resume-to-job similarity scoring using TF-IDF and Cosine Similarity
- Candidate ranking based on job relevance
- Skill gap identification
- Match percentage calculation
- Top candidate visualization using a bar chart
- Export ranked candidates to CSV

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- spaCy
- Matplotlib
- Jupyter Notebook
- VS Code
- pdfplumber

---

## Dataset

This project used from the Resume Dataset from Kaggle.

Download the dataset from:

https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset

The dataset contains resumes from multiple job categories such as:
- Accountant
- Advocate
- Agriculture
- Banking
- BPO
- Engineering
- Finance
- Healthcare
- HR
- Information Technology
- Sales
- Teacher
- And more.

The project supports both:
- *CSV-based resumes* (resumes.csv)
- *PDF resumes* organized into category folders

---
## Project Structure

```
FUTURE_ML_03/
│
├── data/
│   ├── resumes.csv
│   ├── job_description.txt
│   └── data/
│       ├── ACCOUNTANT/
│       ├── ADVOCATE/
│       ├── AGRICULTURE/
│       ├── ...
│       └── TEACHER/
│
├── output/
│   ├── ranked_candidates.csv
│   └── Candidate_Ranking_PDF.csv
│
├── screenshots/
│
├── Resume_Screening.ipynb
├── resume_screening.py
├── resume_screening_pdf.py
├── requirements.txt
├── README.md

```

---

## Installation

Install the required libraries:

```bash
pip install -r requirements.txt
```

Download the spaCy English model:

```bash
python -m spacy download en_core_web_sm
```

---

## How to Run

Run the Jupyter Notebook:

- Open `notebook/Resume_Screening.ipynb`

OR

Run the Python script:

```bash
python resume_screening.py
```

---

## Workflow

1. Load Resume Dataset (CSV/PDF)

2. Clean Resume Text

3. Load Job Description

4. Extract Skills

5. TF-IDF Vectorization

6. Cosine Similarity

7. Candidate Ranking

8. Missing Skill Detection

9. Match Percentage Calculation

10. Generate Output CSV

11. Visualize Candidate Ranking

## Output

The project generates:

- Ranked candidates based on similarity score
- Match percentage for each resume
- Missing skills for each candidate
- Bar chart visualization of top candidates
- `ranked_candidates.csv` inside the `output` folder

CSV Version

- output/ranked_candidates_samples.csv

PDF Version

- output/Candidate_Ranking_pdf.csv

## Screenshots

Project screenshots are available inside the screenshots folder.

---

## Future Improvements

- Support DOCX resumes
- Named Entity Recognition (NER)
- Resume Upload Interface
- AI-based Resume Recommendation
- Advanced Skill Weighting


---

## Author

Lavanya M G

BE Student | Machine Learning Interns

Jawaharlal Nehru National College of Engineering,Shimoga

## Acknowledgement

This project was completed as part of the Future Interns Machine Learning Internship Program
