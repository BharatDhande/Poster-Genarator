from transformers import pipeline

def analyze_text(prompt):
    keywords = ["AI-based Medical Diagnosis", "95% accuracy", "diabetes", "heart disease"]
    tone = "trustworthy"
    mood = "professional"
    return keywords, tone, mood
