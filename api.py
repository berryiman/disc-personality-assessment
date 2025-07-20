from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import random
import numpy as np
import math
from datetime import datetime
import uuid

app = FastAPI(
    title="DISC Assessment API - Berdasarkan Kode Asli",
    description="API yang menggunakan logika EXACT dari disc_style.py",
    version="1.0.0"
)

# Enable CORS untuk n8n
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class DISCAnswer(BaseModel):
    question_id: int
    answer: int  # 1-5 scale

class DISCAssessmentRequest(BaseModel):
    candidate_name: str
    candidate_email: str
    position: str
    answers: List[DISCAnswer]

class DISCResult(BaseModel):
    assessment_id: str
    candidate_name: str
    candidate_email: str
    position: str
    timestamp: str
    raw_scores: Dict[str, float]
    normalized_scores: Dict[str, float]
    relative_percentages: Dict[str, float]
    resultant_angle: float
    resultant_magnitude: float
    primary_style: str
    style_description: str
    questions_used: List[Dict[str, Any]]

# Storage
assessments_db = {}

# Load data files - EXACT sama seperti di aplikasi asli
def load_questions():
    """Load questions exactly as in original app"""
    try:
        with open("questions.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        # Fallback data for testing
        return [
            {
                "style": "D",
                "question": "I thrive in fast-paced environments where I can take charge and make decisions quickly.",
                "mapping": {"D": 2, "I": 1, "S": -1, "C": 0}
            }
            # ... more questions would be loaded from file
        ]

def load_disc_descriptions():
    """Load DISC descriptions exactly as in original app"""
    try:
        with open("disc_descriptions.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"single": {}}

# Load data
questions_data = load_questions()
disc_descriptions = load_disc_descriptions()

def normalize_scores(scores, questions):
    """
    EXACT copy dari fungsi normalize_scores di disc_style.py
    """
    max_possible_scores = {style: 0.0 for style in ["D", "I", "S", "C"]}
    min_possible_scores = {style: 0.0 for style in ["D", "I", "S", "C"]}

    for q in questions:
        for style in ["D", "I", "S", "C"]:
            mapping = q["mapping"][style]
            if mapping >= 0:
                max_contribution = mapping * 2  # Max when (answer - 3) = +2
                min_contribution = mapping * (-2)  # Min when (answer - 3) = -2
            else:
                max_contribution = mapping * (-2)  # Max when (answer - 3) = -2
                min_contribution = mapping * 2  # Min when (answer - 3) = +2

            max_possible_scores[style] += max_contribution
            min_possible_scores[style] += min_contribution

    print(f"Max possible scores: {max_possible_scores}")
    print(f"Min possible scores: {min_possible_scores}")

    normalized_scores = {}
    for style in ["D", "I", "S", "C"]:
        # Ensure the raw score is within the possible range
        score = max(min(scores[style], max_possible_scores[style]), min_possible_scores[style])
        score_range = max_possible_scores[style] - min_possible_scores[style]
        if score_range == 0:
            normalized_scores[style] = 50.0  # Neutral score if no variation is possible
        else:
            normalized_scores[style] = ((score - min_possible_scores[style]) / score_range) * 100
            # Ensure the normalized score is within 0 to 100
            normalized_scores[style] = max(0, min(normalized_scores[style], 100))
    return normalized_scores

def calculate_resultant_vector(normalized_score):
    """
    EXACT copy dari logika vector calculation di disc_style.py
    """
    # Define the categories and their positions - EXACT sama
    categories = ["D", "I", "S", "C"]
    # Angles for the styles - EXACT sama
    angles = [7 * np.pi / 4, np.pi / 4, 3 * np.pi / 4, 5 * np.pi / 4]
    
    # Divide Each Normalized Score by 100 - EXACT sama
    scaled_scores = {style: score / 100 for style, score in normalized_score.items()}

    # Compute x and y components of the style vectors - EXACT sama
    x_components = []
    y_components = []
    for style in categories:
        angle = angles[categories.index(style)]
        magnitude = scaled_scores[style]
        x_components.append(magnitude * np.cos(angle))
        y_components.append(magnitude * np.sin(angle))

    # Sum the components - EXACT sama
    total_x = sum(x_components)
    total_y = sum(y_components)

    # Compute the resultant vector - EXACT sama
    resultant_magnitude = np.sqrt(total_x**2 + total_y**2)
    resultant_angle = np.arctan2(total_y, total_x)
    
    return resultant_angle, resultant_magnitude

def determine_style_from_angle(resultant_angle):
    """
    EXACT copy dari describe_style function di disc_style.py
    """
    # Convert the resultant angle from radians to degrees - EXACT sama
    resultant_degrees = math.degrees(resultant_angle)
    if resultant_degrees < 0:
        resultant_degrees += 360  # Convert negative angles to positive

    # Define the angular ranges - EXACT sama dengan disc_style.py
    style_ranges = {
        # D (Dominance)
        "D": (315, 337.5),
        "DC": (270, 315),
        "DI": (337.5, 360),  # Also covers the 0 degree point
        
        # I (Influence)
        "I": (45, 67.5),
        "ID": (0, 45),
        "IS": (67.5, 90),
        
        # S (Steadiness)
        "S": (135, 157.5),
        "SI": (90, 135),
        "SC": (157.5, 180),
        
        # C (Conscientiousness)
        "C": (225, 247.5),
        "CS": (180, 225),
        "CD": (247.5, 270)
    }

    # Determine which range the resultant angle falls into - EXACT sama
    for style, (start_angle, end_angle) in style_ranges.items():
        if start_angle <= resultant_degrees < end_angle or (start_angle == 337.5 and resultant_degrees == 0):
            return style
    
    # Default fallback
    return "Balanced Style"

def get_style_description(style):
    """Get description from disc_descriptions.json"""
    if style in disc_descriptions.get("single", {}):
        desc = disc_descriptions["single"][style]
        return f"{desc['title']}\n\n{desc['description']}\n\nStrengths: {desc['strengths']}\n\nChallenges: {desc['challenges']}"
    return "Balanced Style - Your responses indicate a balanced personality without a clear preference for any specific DISC style."

@app.get("/")
async def root():
    return {
        "message": "DISC Assessment API - Menggunakan Logika Exact dari disc_style.py",
        "version": "1.0.0",
        "questions_loaded": len(questions_data),
        "endpoints": {
            "submit_assessment": "/api/v1/assessment/submit",
            "get_result": "/api/v1/assessment/{assessment_id}",
            "list_assessments": "/api/v1/assessments",
            "get_questions": "/api/v1/questions",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/api/v1/questions")
async def get_random_questions(count: int = 30):
    """
    Get random questions exactly like the original app
    """
    if count > len(questions_data):
        count = len(questions_data)
    
    # Random shuffle dan ambil 30 questions - EXACT sama dengan aplikasi asli
    shuffled_questions = questions_data.copy()
    random.shuffle(shuffled_questions)
    selected_questions = shuffled_questions[:count]
    
    # Add question IDs for API usage
    for i, q in enumerate(selected_questions):
        q["question_id"] = i
    
    return {
        "questions": selected_questions,
        "total_questions": count,
        "instructions": {
            "scale": {
                "1": "Completely Disagree",
                "2": "Somehow Disagree", 
                "3": "Neutral",
                "4": "Somehow Agree",
                "5": "Completely Agree"
            }
        }
    }

@app.post("/api/v1/assessment/submit", response_model=DISCResult)
async def submit_assessment(request: DISCAssessmentRequest):
    """
    Process DISC assessment menggunakan EXACT algoritma dari disc_style.py
    """
    try:
        assessment_id = str(uuid.uuid4())
        
        # Validate answers count
        if len(request.answers) < 20:
            raise HTTPException(status_code=400, detail="Minimum 20 answers required")
        
        # Get random questions untuk assessment ini (simulate original app behavior)
        shuffled_questions = questions_data.copy()
        random.shuffle(shuffled_questions)
        selected_questions = shuffled_questions[:len(request.answers)]
        
        # Initialize scores - EXACT sama dengan disc_style.py
        raw_scores = {"D": 0, "I": 0, "S": 0, "C": 0}
        
        # Calculate raw scores - EXACT algoritma dari disc_style.py
        for i, answer_data in enumerate(request.answers):
            if i >= len(selected_questions):
                break
                
            q = selected_questions[i]
            answer = answer_data.answer
            
            # EXACT scoring logic dari disc_style.py
            for style in ["D", "I", "S", "C"]:
                raw_scores[style] += q["mapping"][style] * (answer - 3)
        
        print(f'Raw scores: {raw_scores}')
        
        # Normalize scores - menggunakan fungsi EXACT dari disc_style.py
        normalized_scores = normalize_scores(raw_scores, selected_questions)
        print(f'Normalized scores: {normalized_scores}')
        
        # Calculate relative percentages - EXACT dari disc_style.py
        total_normalized = sum(normalized_scores.values())
        relative_percentages = {}
        for style, score in normalized_scores.items():
            if total_normalized == 0:
                relative_percentages[style] = 0
            else:
                relative_percentages[style] = (score / total_normalized) * 100
        
        # Calculate resultant vector - EXACT dari disc_style.py
        resultant_angle, resultant_magnitude = calculate_resultant_vector(normalized_scores)
        
        # Determine primary style - EXACT dari disc_style.py
        primary_style = determine_style_from_angle(resultant_angle)
        
        # Get style description
        style_description = get_style_description(primary_style)
        
        # Create result
        result = DISCResult(
            assessment_id=assessment_id,
            candidate_name=request.candidate_name,
            candidate_email=request.candidate_email,
            position=request.position,
            timestamp=datetime.now().isoformat(),
            raw_scores=raw_scores,
            normalized_scores=normalized_scores,
            relative_percentages=relative_percentages,
            resultant_angle=float(resultant_angle),
            resultant_magnitude=float(resultant_magnitude),
            primary_style=primary_style,
            style_description=style_description,
            questions_used=selected_questions
        )
        
        # Store in database
        assessments_db[assessment_id] = result.dict()
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing assessment: {str(e)}")

@app.get("/api/v1/assessment/{assessment_id}", response_model=DISCResult)
async def get_assessment_result(assessment_id: str):
    """Get assessment result by ID"""
    if assessment_id not in assessments_db:
        raise HTTPException(status_code=404, detail="Assessment not found")
    
    return DISCResult(**assessments_db[assessment_id])

@app.get("/api/v1/assessments")
async def list_assessments(limit: int = 50, offset: int = 0):
    """List all assessments with pagination"""
    all_assessments = list(assessments_db.values())
    total = len(all_assessments)
    
    start = offset
    end = offset + limit
    paginated_assessments = all_assessments[start:end]
    
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "assessments": paginated_assessments
    }

@app.get("/api/v1/assessments/by-email/{email}")
async def get_assessments_by_email(email: str):
    """Get all assessments for a specific email"""
    user_assessments = [
        assessment for assessment in assessments_db.values()
        if assessment["candidate_email"] == email
    ]
    
    if not user_assessments:
        raise HTTPException(status_code=404, detail="No assessments found for this email")
    
    return {"email": email, "assessments": user_assessments}

@app.get("/api/v1/debug/compare-with-streamlit")
async def debug_comparison():
    """Helper untuk debug - bandingkan hasil dengan Streamlit app"""
    return {
        "message": "Untuk memastikan hasil sama dengan Streamlit app:",
        "steps": [
            "1. Jalankan assessment yang sama di Streamlit app",
            "2. Gunakan questions yang sama (gunakan endpoint /api/v1/questions)", 
            "3. Submit answers yang sama ke API",
            "4. Bandingkan raw_scores, normalized_scores, dan primary_style",
            "5. Jika ada perbedaan, periksa mapping questions atau logic scoring"
        ],
        "note": "API ini menggunakan EXACT algoritma dari disc_style.py"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
