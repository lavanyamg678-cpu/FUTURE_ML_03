# Resume / Candidate Screening System

## Project Overview

This project is a Machine Learning and Natural Language Processing (NLP) based Resume Screening System developed as part of the Future Interns Machine Learning Internship – Task 3.

The system automatically compares resumes with a job description, ranks candidates based on similarity, and identifies missing skills required for the job role.

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

---

## Dataset

- Resume Dataset (Kaggle)
- Custom Job Description

---

## Project Structure

```
FUTURE_ML_03/
│
├── data/
│   ├── resumes.csv
│   └── job_description.txt
│
├── notebook/
│   └── Resume_Screening.ipynb
│
├── output/
│   └── ranked_candidates.csv
│
├── resume_screening.py
├── requirements.txt
├── README.md
└── .gitignore
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

## Output

The project generates:

- Ranked candidates based on similarity score
- Match percentage for each resume
- Missing skills for each candidate
- Bar chart visualization of top candidates
- `ranked_candidates.csv` inside the `output` folder

---

## Author

Lavanya M G

BE Student | Machine Learning Interns
Jawaharlal Nehru National College of Engineering,Shimoga

## Acknowledgement

This project was completed as part of the Future Interns Machine Learning Internship Program
