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
import os
import psycopg2
from psycopg2.extras import RealDictCursor
import psycopg2.pool
from contextlib import contextmanager

app = FastAPI(
    title="DISC Assessment API - Dengan PostgreSQL",
    description="API yang menggunakan logika EXACT dari disc_style.py + PostgreSQL storage",
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

# PostgreSQL Connection
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    # Fallback untuk development
    DATABASE_URL = "postgresql://user:password@localhost:5432/dbname"

# Connection pool
connection_pool = None

def init_connection_pool():
    global connection_pool
    try:
        connection_pool = psycopg2.pool.SimpleConnectionPool(
            1, 20,  # min dan max connections
            DATABASE_URL
        )
        print("PostgreSQL connection pool created successfully")
    except Exception as e:
        print(f"Error creating connection pool: {e}")
        connection_pool = None

@contextmanager
def get_db_connection():
    if connection_pool is None:
        raise HTTPException(status_code=503, detail="Database connection not available")
    
    conn = connection_pool.getconn()
    try:
        yield conn
    finally:
        connection_pool.putconn(conn)

def create_tables():
    """Create tables if they don't exist"""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS disc_assessments (
        id SERIAL PRIMARY KEY,
        assessment_id VARCHAR(50) UNIQUE NOT NULL,
        candidate_name VARCHAR(255) NOT NULL,
        candidate_email VARCHAR(255) NOT NULL,
        position VARCHAR(255),
        raw_scores JSONB NOT NULL,
        normalized_scores JSONB NOT NULL,
        relative_percentages JSONB NOT NULL,
        resultant_angle FLOAT NOT NULL,
        resultant_magnitude FLOAT NOT NULL,
        primary_style VARCHAR(10) NOT NULL,
        style_description TEXT,
        questions_used JSONB,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE INDEX IF NOT EXISTS idx_assessment_id ON disc_assessments(assessment_id);
    CREATE INDEX IF NOT EXISTS idx_candidate_email ON disc_assessments(candidate_email);
    CREATE INDEX IF NOT EXISTS idx_created_at ON disc_assessments(created_at);
    """
    
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(create_table_sql)
                conn.commit()
        print("Database tables created successfully")
    except Exception as e:
        print(f"Error creating tables: {e}")

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
    questions_used: Optional[List[Dict[str, Any]]] = None

# Load data files - EXACT sama seperti di aplikasi asli
def load_questions():
    """Load questions exactly as in original app"""
    try:
        with open("questions.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

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

# EXACT functions dari disc_style.py (sama persis)
def normalize_scores(scores, questions):
    """EXACT copy dari fungsi normalize_scores di disc_style.py"""
    max_possible_scores = {style: 0.0 for style in ["D", "I", "S", "C"]}
    min_possible_scores = {style: 0.0 for style in ["D", "I", "S", "C"]}

    for q in questions:
        for style in ["D", "I", "S", "C"]:
            mapping = q["mapping"][style]
            if mapping >= 0:
                max_contribution = mapping * 2
                min_contribution = mapping * (-2)
            else:
                max_contribution = mapping * (-2)
                min_contribution = mapping * 2

            max_possible_scores[style] += max_contribution
            min_possible_scores[style] += min_contribution

    normalized_scores = {}
    for style in ["D", "I", "S", "C"]:
        score = max(min(scores[style], max_possible_scores[style]), min_possible_scores[style])
        score_range = max_possible_scores[style] - min_possible_scores[style]
        if score_range == 0:
            normalized_scores[style] = 50.0
        else:
            normalized_scores[style] = ((score - min_possible_scores[style]) / score_range) * 100
            normalized_scores[style] = max(0, min(normalized_scores[style], 100))
    return normalized_scores

def calculate_resultant_vector(normalized_score):
    """EXACT copy dari logika vector calculation di disc_style.py"""
    categories = ["D", "I", "S", "C"]
    angles = [7 * np.pi / 4, np.pi / 4, 3 * np.pi / 4, 5 * np.pi / 4]
    
    scaled_scores = {style: score / 100 for style, score in normalized_score.items()}

    x_components = []
    y_components = []
    for style in categories:
        angle = angles[categories.index(style)]
        magnitude = scaled_scores[style]
        x_components.append(magnitude * np.cos(angle))
        y_components.append(magnitude * np.sin(angle))

    total_x = sum(x_components)
    total_y = sum(y_components)

    resultant_magnitude = np.sqrt(total_x**2 + total_y**2)
    resultant_angle = np.arctan2(total_y, total_x)
    
    return resultant_angle, resultant_magnitude

def determine_style_from_angle(resultant_angle):
    """EXACT copy dari describe_style function di disc_style.py"""
    resultant_degrees = math.degrees(resultant_angle)
    if resultant_degrees < 0:
        resultant_degrees += 360

    style_ranges = {
        "D": (315, 337.5),
        "DC": (270, 315),
        "DI": (337.5, 360),
        "I": (45, 67.5),
        "ID": (0, 45),
        "IS": (67.5, 90),
        "S": (135, 157.5),
        "SI": (90, 135),
        "SC": (157.5, 180),
        "C": (225, 247.5),
        "CS": (180, 225),
        "CD": (247.5, 270)
    }

    for style, (start_angle, end_angle) in style_ranges.items():
        if start_angle <= resultant_degrees < end_angle or (start_angle == 337.5 and resultant_degrees == 0):
            return style
    
    return "Balanced Style"

def get_style_description(style):
    """Get description from disc_descriptions.json"""
    if style in disc_descriptions.get("single", {}):
        desc = disc_descriptions["single"][style]
        return f"{desc['title']}\n\n{desc['description']}\n\nStrengths: {desc['strengths']}\n\nChallenges: {desc['challenges']}"
    return "Balanced Style - Your responses indicate a balanced personality without a clear preference for any specific DISC style."

# Database functions
def save_assessment_to_db(result: DISCResult):
    """Save assessment result to PostgreSQL"""
    insert_sql = """
    INSERT INTO disc_assessments (
        assessment_id, candidate_name, candidate_email, position,
        raw_scores, normalized_scores, relative_percentages,
        resultant_angle, resultant_magnitude, primary_style,
        style_description, questions_used
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(insert_sql, (
                    result.assessment_id,
                    result.candidate_name,
                    result.candidate_email,
                    result.position,
                    json.dumps(result.raw_scores),
                    json.dumps(result.normalized_scores),
                    json.dumps(result.relative_percentages),
                    result.resultant_angle,
                    result.resultant_magnitude,
                    result.primary_style,
                    result.style_description,
                    json.dumps(result.questions_used) if result.questions_used else None
                ))
                conn.commit()
        return True
    except Exception as e:
        print(f"Error saving to database: {e}")
        return False

def get_assessment_from_db(assessment_id: str):
    """Get assessment by ID from PostgreSQL"""
    select_sql = """
    SELECT assessment_id, candidate_name, candidate_email, position,
           raw_scores, normalized_scores, relative_percentages,
           resultant_angle, resultant_magnitude, primary_style,
           style_description, questions_used, created_at
    FROM disc_assessments 
    WHERE assessment_id = %s
    """
    
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(select_sql, (assessment_id,))
                row = cur.fetchone()
                if row:
                    return dict(row)
        return None
    except Exception as e:
        print(f"Error getting from database: {e}")
        return None

def get_assessments_by_email_from_db(email: str):
    """Get assessments by email from PostgreSQL"""
    select_sql = """
    SELECT assessment_id, candidate_name, candidate_email, position,
           raw_scores, normalized_scores, relative_percentages,
           resultant_angle, resultant_magnitude, primary_style,
           style_description, created_at
    FROM disc_assessments 
    WHERE candidate_email = %s
    ORDER BY created_at DESC
    """
    
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(select_sql, (email,))
                rows = cur.fetchall()
                return [dict(row) for row in rows]
        return []
    except Exception as e:
        print(f"Error getting assessments by email: {e}")
        return []

def get_all_assessments_from_db(limit: int = 50, offset: int = 0):
    """Get all assessments with pagination from PostgreSQL"""
    select_sql = """
    SELECT assessment_id, candidate_name, candidate_email, position,
           raw_scores, normalized_scores, relative_percentages,
           resultant_angle, resultant_magnitude, primary_style,
           style_description, created_at
    FROM disc_assessments 
    ORDER BY created_at DESC
    LIMIT %s OFFSET %s
    """
    
    count_sql = "SELECT COUNT(*) FROM disc_assessments"
    
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(select_sql, (limit, offset))
                rows = cur.fetchall()
                
                cur.execute(count_sql)
                total = cur.fetchone()['count']
                
                return [dict(row) for row in rows], total
        return [], 0
    except Exception as e:
        print(f"Error getting all assessments: {e}")
        return [], 0

# API Endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    init_connection_pool()
    create_tables()

@app.get("/")
async def root():
    return {
        "message": "DISC Assessment API - Dengan PostgreSQL Storage",
        "version": "1.0.0",
        "questions_loaded": len(questions_data),
        "database": "PostgreSQL" if connection_pool else "Not Connected",
        "endpoints": {
            "submit_assessment": "/api/v1/assessment/submit",
            "get_result": "/api/v1/assessment/{assessment_id}",
            "list_assessments": "/api/v1/assessments",
            "get_by_email": "/api/v1/assessments/by-email/{email}",
            "get_questions": "/api/v1/questions",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    db_status = "connected" if connection_pool else "disconnected"
    return {
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "database": db_status
    }

@app.get("/api/v1/questions")
async def get_random_questions(count: int = 30):
    """Get random questions exactly like the original app"""
    if count > len(questions_data):
        count = len(questions_data)
    
    shuffled_questions = questions_data.copy()
    random.shuffle(shuffled_questions)
    selected_questions = shuffled_questions[:count]
    
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
    """Process DISC assessment dan simpan ke PostgreSQL"""
    try:
        assessment_id = str(uuid.uuid4())
        
        if len(request.answers) < 20:
            raise HTTPException(status_code=400, detail="Minimum 20 answers required")
        
        shuffled_questions = questions_data.copy()
        random.shuffle(shuffled_questions)
        selected_questions = shuffled_questions[:len(request.answers)]
        
        raw_scores = {"D": 0, "I": 0, "S": 0, "C": 0}
        
        for i, answer_data in enumerate(request.answers):
            if i >= len(selected_questions):
                break
                
            q = selected_questions[i]
            answer = answer_data.answer
            
            for style in ["D", "I", "S", "C"]:
                raw_scores[style] += q["mapping"][style] * (answer - 3)
        
        normalized_scores = normalize_scores(raw_scores, selected_questions)
        
        total_normalized = sum(normalized_scores.values())
        relative_percentages = {}
        for style, score in normalized_scores.items():
            if total_normalized == 0:
                relative_percentages[style] = 0
            else:
                relative_percentages[style] = (score / total_normalized) * 100
        
        resultant_angle, resultant_magnitude = calculate_resultant_vector(normalized_scores)
        primary_style = determine_style_from_angle(resultant_angle)
        style_description = get_style_description(primary_style)
        
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
        
        # Save to PostgreSQL
        if save_assessment_to_db(result):
            print(f"Assessment {assessment_id} saved to database")
        else:
            print(f"Failed to save assessment {assessment_id} to database")
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing assessment: {str(e)}")

@app.get("/api/v1/assessment/{assessment_id}", response_model=DISCResult)
async def get_assessment_result(assessment_id: str):
    """Get assessment result by ID dari PostgreSQL"""
    row = get_assessment_from_db(assessment_id)
    if not row:
        raise HTTPException(status_code=404, detail="Assessment not found")
    
    return DISCResult(
        assessment_id=row['assessment_id'],
        candidate_name=row['candidate_name'],
        candidate_email=row['candidate_email'],
        position=row['position'],
        timestamp=row['created_at'].isoformat(),
        raw_scores=row['raw_scores'],
        normalized_scores=row['normalized_scores'],
        relative_percentages=row['relative_percentages'],
        resultant_angle=row['resultant_angle'],
        resultant_magnitude=row['resultant_magnitude'],
        primary_style=row['primary_style'],
        style_description=row['style_description'],
        questions_used=row['questions_used']
    )

@app.get("/api/v1/assessments")
async def list_assessments(limit: int = 50, offset: int = 0):
    """List all assessments with pagination dari PostgreSQL"""
    assessments, total = get_all_assessments_from_db(limit, offset)
    
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "assessments": assessments
    }

@app.get("/api/v1/assessments/by-email/{email}")
async def get_assessments_by_email(email: str):
    """Get all assessments for a specific email dari PostgreSQL"""
    assessments = get_assessments_by_email_from_db(email)
    
    if not assessments:
        raise HTTPException(status_code=404, detail="No assessments found for this email")
    
    return {"email": email, "assessments": assessments}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
