import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

# ==========================================
# 1. SETUP & MODEL LOADING
# ==========================================

# Define the paths to your local model folders
# These folders must be in the same directory as this script
PATH_MODEL_1 = "./eligibility_classifier"
PATH_MODEL_2 = "./job_recommender"

print("⏳ Loading AI Models... This may take a minute.")

try:
    # Load Model 1: The Classifier (Yes/No)
    # We use the 'text-classification' pipeline
    classifier = pipeline("text-classification", model=PATH_MODEL_1)
    print("✅ Model 1 (Eligibility Classifier) Loaded Successfully.")

    # Load Model 2: The Recommender (T5 Generator)
    # We use the 'text2text-generation' pipeline
    recommender = pipeline("text2text-generation", model=PATH_MODEL_2)
    print("✅ Model 2 (Job Recommender) Loaded Successfully.")

except Exception as e:
    print(f"❌ CRITICAL ERROR: Could not load models.\nError details: {e}")
    print("Please ensure 'eligibility_classifier' and 'job_recommender' folders are in this directory.")
    # We continue, but predictions will fail if models aren't loaded
    classifier = None
    recommender = None

# Initialize the App
app = FastAPI(title="AI Recruiter Agent API", version="1.0")


# ==========================================
# 2. DEFINE DATA FORMATS (INPUTS)
# ==========================================

class EligibilityInput(BaseModel):
    resume_text: str
    job_description: str

class RecommendationInput(BaseModel):
    resume_text: str


# ==========================================
# 3. DEFINE API ENDPOINTS (ROUTES)
# ==========================================

@app.get("/")
def home():
    """Simple health check to see if server is running."""
    return {"status": "Online", "message": "AI Recruiter API is ready."}


@app.post("/predict_eligibility")
def predict_eligibility(data: EligibilityInput):
    """
    Endpoint for Model 1.
    Receives: Resume + Job Description
    Returns: Label (1/0) and Confidence Score
    """
    if not classifier:
        raise HTTPException(status_code=500, detail="Model 1 is not loaded.")

    # Format input exactly how we trained it: [Resume] [SEP] [Job Desc]
    combined_input = f"{data.resume_text} [SEP] {data.job_description}"

    # Run the prediction (the model handles tokenization internally)
    # Truncation ensures we don't crash on long text
    result = classifier(combined_input, truncation=True, max_length=512)

    # Result looks like: [{'label': 'LABEL_1', 'score': 0.98}]
    return {
        "status": "success",
        "prediction": result[0]['label'], # LABEL_1 (Yes) or LABEL_0 (No)
        "confidence": result[0]['score']
    }


@app.post("/recommend_job")
def recommend_job(data: RecommendationInput):
    """
    Endpoint for Model 2.
    Receives: Resume
    Returns: Suggested Job Title
    """
    if not recommender:
        raise HTTPException(status_code=500, detail="Model 2 is not loaded.")

    # Add the prefix we used during training
    input_text = f"recommend job title: {data.resume_text}"

    # Generate the text
    result = recommender(
        input_text, 
        max_new_tokens=30, # Don't generate a whole essay, just a title
        num_return_sequences=1
    )

    # Result looks like: [{'generated_text': 'Data Scientist'}]
    return {
        "status": "success",
        "suggested_job": result[0]['generated_text']
    }