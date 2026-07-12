import os
import re
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import en_core_web_sm

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Download NLTK stopwords
# -----------------------------
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

# -----------------------------
# Load spaCy model
# -----------------------------
nlp = en_core_web_sm.load()

# -----------------------------
# Load Resume Dataset
# -----------------------------
df = pd.read_csv("data/resumes.csv")

print("Dataset Loaded Successfully!")
print(df.head())

# -----------------------------
# Resume Cleaning Function
# -----------------------------
def clean_resume(text):
    text = str(text).lower()

    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()

df["Clean_Resume"] = df["Resume_str"].apply(clean_resume)

# -----------------------------
# Read Job Description
# -----------------------------
with open("data/job_description.txt", "r", encoding="utf-8") as f:
    job_description = f.read()

print("\nJob Description Loaded Successfully!\n")

# -----------------------------
# TF-IDF Vectorization
# -----------------------------
documents = [job_description] + df["Clean_Resume"].tolist()

vectorizer = TfidfVectorizer(stop_words="english")

tfidf_matrix = vectorizer.fit_transform(documents)

job_vector = tfidf_matrix[0]
resume_vectors = tfidf_matrix[1:]

similarity_scores = cosine_similarity(job_vector, resume_vectors)

df["Similarity Score"] = similarity_scores.flatten()

# -----------------------------
# Rank Candidates
# -----------------------------
ranked_df = df.sort_values(
    by="Similarity Score",
    ascending=False
).reset_index(drop=True)

# -----------------------------
# Skill List
# -----------------------------
skills = [
    "python",
    "java",
    "c++",
    "sql",
    "machine learning",
    "deep learning",
    "data science",
    "artificial intelligence",
    "nlp",
    "tensorflow",
    "keras",
    "pytorch",
    "scikit-learn",
    "pandas",
    "numpy",
    "excel",
    "power bi",
    "tableau",
    "git",
    "github",
    "communication",
    "problem solving"
]

# -----------------------------
# Extract Skills
# -----------------------------
def extract_skills(text):

    text = text.lower()

    found = []

    for skill in skills:
        if skill in text:
            found.append(skill)

    return sorted(list(set(found)))

ranked_df["Extracted Skills"] = ranked_df["Clean_Resume"].apply(extract_skills)

# -----------------------------
# Job Skills
# -----------------------------
job_skills = extract_skills(job_description)

# -----------------------------
# Missing Skills
# -----------------------------
def missing_skills(resume_skills):

    missing = []

    for skill in job_skills:

        if skill not in resume_skills:
            missing.append(skill)

    return missing

ranked_df["Missing Skills"] = ranked_df["Extracted Skills"].apply(missing_skills)

# -----------------------------
# Match Percentage
# -----------------------------
def match_percentage(resume_skills):

    if len(job_skills) == 0:
        return 0

    matched = len(set(resume_skills) & set(job_skills))

    return round((matched / len(job_skills)) * 100, 2)

ranked_df["Match %"] = ranked_df["Extracted Skills"].apply(match_percentage)

# -----------------------------
# Save Output
# -----------------------------
os.makedirs("output", exist_ok=True)

ranked_df.to_csv(
    "output/ranked_candidates.csv",
    index=False
)

print("\nTop 10 Ranked Candidates\n")

print(
    ranked_df[
        [
            "Category",
            "Similarity Score",
            "Match %",
            "Missing Skills"
        ]
    ].head(10)
)

print("\nOutput saved to output/ranked_candidates.csv")

# -----------------------------
# Visualization
# -----------------------------
top10 = ranked_df.head(10)

plt.figure(figsize=(12,6))

bars = plt.bar(
    top10["Category"],
    top10["Match %"]
)

plt.title("Top 10 Candidate Match Percentage")

plt.xlabel("Candidate Category")

plt.ylabel("Match Percentage (%)")

plt.xticks(rotation=45)

plt.ylim(0,100)

for bar in bars:

    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width()/2,
        height + 1,
        f"{height:.0f}%",
        ha="center",
        fontsize=9
    )

plt.tight_layout()

plt.show()

print("\nProject Completed Successfully!")