import os
import re
import nltk
import pdfplumber
import pandas as pd
import matplotlib.pyplot as plt
import en_core_web_sm

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------
# Download NLTK Stopwords
# ------------------------------------
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

# ------------------------------------
# Load spaCy Model
# ------------------------------------
nlp = en_core_web_sm.load()

# ------------------------------------
# Resume Folder
# ------------------------------------
resume_folder = "data/data"

resume_names = []
resume_categories = []
resume_texts = []

print("Reading PDF Resumes...")

# ------------------------------------
# Read PDF Resumes
# ------------------------------------
for category in os.listdir(resume_folder):

    category_path = os.path.join(resume_folder, category)

    if os.path.isdir(category_path):

        for file in os.listdir(category_path):

            if file.lower().endswith(".pdf"):

                pdf_path = os.path.join(category_path, file)

                text = ""

                try:

                    with pdfplumber.open(pdf_path) as pdf:

                        for page in pdf.pages:

                            page_text = page.extract_text()

                            if page_text:

                                text += page_text + "\n"

                    resume_names.append(file)
                    resume_categories.append(category)
                    resume_texts.append(text)

                except Exception as e:

                    print(f"Could not read {file}")

print(f"\nTotal PDF Resumes Loaded : {len(resume_names)}")

# ------------------------------------
# Create DataFrame
# ------------------------------------
df = pd.DataFrame({
    "Resume": resume_names,
    "Category": resume_categories,
    "Resume_Text": resume_texts
})

print(df.head())
# ------------------------------------
# Clean Resume Text
# ------------------------------------
def clean_resume(text):

    text = str(text).lower()

    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = re.sub(r"\s+", " ", text)

    words = text.split()

    words = [word for word in words if word not in stop_words]

    return " ".join(words)


df["Clean_Resume"] = df["Resume_Text"].apply(clean_resume)

print("\nResume Cleaning Completed!")

# ------------------------------------
# Read Job Description
# ------------------------------------
with open("data/job_description.txt", "r", encoding="utf-8") as f:
    job_description = f.read()

print("\nJob Description Loaded Successfully!")

# ------------------------------------
# TF-IDF Vectorization
# ------------------------------------
documents = df["Clean_Resume"].tolist()

documents.append(job_description)

vectorizer = TfidfVectorizer()

tfidf_matrix = vectorizer.fit_transform(documents)

resume_vectors = tfidf_matrix[:-1]

job_vector = tfidf_matrix[-1]

print("\nTF-IDF Vectorization Completed!")

# ------------------------------------
# Similarity Calculation
# ------------------------------------
scores = cosine_similarity(
    job_vector,
    resume_vectors
).flatten()

df["Similarity Score"] = scores

print("\nSimilarity Scores Calculated!")

# ------------------------------------
# Rank Candidates
# ------------------------------------
ranked_df = df.sort_values(
    by="Similarity Score",
    ascending=False
).reset_index(drop=True)

print("\nTop 5 Candidates")

print(
    ranked_df[
        ["Resume", "Category", "Similarity Score"]
    ].head()
)
# ------------------------------------
# Skill List
# ------------------------------------
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

# ------------------------------------
# Extract Skills
# ------------------------------------
def extract_skills(text):

    text = text.lower()

    found = []

    for skill in skills:
        if skill in text:
            found.append(skill)

    return sorted(list(set(found)))

ranked_df["Extracted Skills"] = ranked_df["Clean_Resume"].apply(extract_skills)

# ------------------------------------
# Job Skills
# ------------------------------------
job_skills = extract_skills(job_description)

print("\nSkills Required for Job:")
print(job_skills)

# ------------------------------------
# Missing Skills
# ------------------------------------
def missing_skills(resume_skills):

    return [skill for skill in job_skills if skill not in resume_skills]

ranked_df["Missing Skills"] = ranked_df["Extracted Skills"].apply(missing_skills)

# ------------------------------------
# Match Percentage
# ------------------------------------
def match_percentage(resume_skills):

    if len(job_skills) == 0:
        return 0

    matched = len(set(resume_skills) & set(job_skills))

    return round((matched / len(job_skills)) * 100, 2)

ranked_df["Match %"] = ranked_df["Extracted Skills"].apply(match_percentage)

print("\nTop 10 Ranked Candidates\n")

print(
    ranked_df[
        [
            "Resume",
            "Category",
            "Similarity Score",
            "Match %",
            "Missing Skills"
        ]
    ].head(10)
)
# ------------------------------------
# Create Output Folder
# ------------------------------------
os.makedirs("output", exist_ok=True)

# ------------------------------------
# Save Ranked Candidates
# ------------------------------------
output_file = "output/ranked_candidates.csv"

ranked_df.to_csv(output_file, index=False)

print(f"\nResults saved successfully to: {output_file}")

# ------------------------------------
# Display Top 10 Candidates
# ------------------------------------
print("\nTop 10 Ranked Candidates\n")

print(
    ranked_df[
        [
            "Resume",
            "Category",
            "Similarity Score",
            "Match %",
            "Missing Skills"
        ]
    ].head(10)
)

# ------------------------------------
# Visualization
# ------------------------------------
top10 = ranked_df.head(10)

plt.figure(figsize=(12,6))

bars = plt.bar(
    top10["Resume"],
    top10["Match %"]
)

plt.title("Top 10 Candidate Match Percentage")

plt.xlabel("Resume")

plt.ylabel("Match Percentage (%)")

plt.xticks(rotation=90)

plt.ylim(0,100)

for bar in bars:

    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width()/2,
        height + 1,
        f"{height:.0f}%",
        ha="center",
        fontsize=8
    )

plt.tight_layout()

plt.show()

print("\nProject Completed Successfully!")