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
import sys
import psycopg2
from psycopg2.extras import RealDictCursor
import psycopg2.pool
from contextlib import contextmanager
import logging
import requests
import asyncio
import aiohttp

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="DISC Assessment API - Railway PostgreSQL with n8n Webhook",
    description="API yang menggunakan logika EXACT dari disc_style.py + PostgreSQL storage untuk Railway + n8n webhook integration",
    version="2.1.0"
)

# Enable CORS untuk n8n dan external access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================== DATABASE CONFIGURATION ==================

def get_database_url():
    """
    Get database URL dengan multiple fallbacks untuk Railway
    Railway bisa provide DATABASE_URL dalam berbagai format
    """
    logger.info("🔍 Searching for database configuration...")
    
    # Method 1: Standard DATABASE_URL
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        logger.info("✅ Found DATABASE_URL")
        return database_url
    
    # Method 2: Railway PostgreSQL service variables
    pg_host = os.getenv("PGHOST")
    pg_port = os.getenv("PGPORT", "5432")
    pg_user = os.getenv("PGUSER")
    pg_password = os.getenv("PGPASSWORD")
    pg_database = os.getenv("PGDATABASE")
    
    if all([pg_host, pg_user, pg_password, pg_database]):
        database_url = f"postgresql://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_database}"
        logger.info("✅ Built DATABASE_URL from PG* variables")
        return database_url
    
    # Method 3: Railway private network URL
    private_url = os.getenv("DATABASE_PRIVATE_URL")
    if private_url:
        logger.info("✅ Found DATABASE_PRIVATE_URL")
        return private_url
    
    # Method 4: Railway public URL
    public_url = os.getenv("DATABASE_PUBLIC_URL")
    if public_url:
        logger.info("✅ Found DATABASE_PUBLIC_URL")
        return public_url
    
    # Method 5: Alternative naming patterns
    for env_name in ["POSTGRES_URL", "POSTGRESQL_URL", "DATABASE_CONNECTION_STRING"]:
        url = os.getenv(env_name)
        if url:
            logger.info(f"✅ Found {env_name}")
            return url
    
    # Log available environment variables for debugging
    logger.error("🚨 No database URL found!")
    logger.info("Available environment variables:")
    for key in sorted(os.environ.keys()):
        if any(term in key.upper() for term in ['DATABASE', 'POSTGRES', 'PG', 'RAILWAY']):
            value = os.environ[key]
            # Mask sensitive data
            if any(sensitive in key.lower() for sensitive in ['password', 'secret', 'key', 'token']):
                value = "***MASKED***"
            elif len(value) > 100:
                value = value[:50] + "..." + value[-20:]
            logger.info(f"  {key}={value}")
    
    return None

def is_production():
    """Check if running in production (Railway)"""
    return any([
        os.getenv("RAILWAY_ENVIRONMENT"),
        os.getenv("RAILWAY_PROJECT_ID"),
        os.getenv("RAILWAY_SERVICE_ID"),
        os.getenv("RAILWAY_DEPLOYMENT_ID")
    ])

def get_ssl_mode():
    """Get appropriate SSL mode for database connection"""
    if is_production():
        return "require"
    return "prefer"

# ================== WEBHOOK CONFIGURATION ==================

def get_webhook_config():
    """Get webhook configuration from environment variables"""
    return {
        "n8n_webhook_url": os.getenv("N8N_WEBHOOK_URL"),
        "webhook_enabled": os.getenv("WEBHOOK_ENABLED", "false").lower() == "true",
        "webhook_timeout": int(os.getenv("WEBHOOK_TIMEOUT", "30")),
        "webhook_retry_attempts": int(os.getenv("WEBHOOK_RETRY_ATTEMPTS", "3")),
        "webhook_secret": os.getenv("WEBHOOK_SECRET", "")
    }

webhook_config = get_webhook_config()

# Initialize database configuration
try:
    DATABASE_URL = get_database_url()
    if not DATABASE_URL:
        error_msg = "Database URL not found in environment variables"
        logger.error(f"🚨 {error_msg}")
        if is_production():
            raise Exception(error_msg)
        else:
            logger.warning("⚠️ Using fallback localhost database for development")
            DATABASE_URL = "postgresql://user:password@localhost:5432/dbname"
    else:
        # Mask password in logs
        masked_url = DATABASE_URL
        if "@" in masked_url:
            parts = masked_url.split("@")
            if ":" in parts[0]:
                auth_parts = parts[0].split(":")
                if len(auth_parts) >= 3:  # postgresql://user:password
                    auth_parts[-1] = "***"
                    parts[0] = ":".join(auth_parts)
                    masked_url = "@".join(parts)
        logger.info(f"✅ Database URL configured: {masked_url}")

except Exception as e:
    logger.error(f"🚨 Database configuration failed: {e}")
    if is_production():
        sys.exit(1)
    DATABASE_URL = None

# Connection pool
connection_pool = None

def init_connection_pool():
    """Initialize PostgreSQL connection pool dengan error handling"""
    global connection_pool
    
    if not DATABASE_URL:
        logger.error("🚨 Cannot initialize connection pool: DATABASE_URL not available")
        return False
    
    try:
        logger.info("🔄 Initializing PostgreSQL connection pool...")
        
        # Parse SSL requirements
        ssl_mode = get_ssl_mode()
        
        connection_pool = psycopg2.pool.SimpleConnectionPool(
            1, 20,  # min dan max connections
            DATABASE_URL,
            sslmode=ssl_mode,
            connect_timeout=30,
            application_name="disc-assessment-api"
        )
        
        logger.info("✅ PostgreSQL connection pool created successfully")
        
        # Test connection
        test_conn = None
        try:
            test_conn = connection_pool.getconn()
            with test_conn.cursor() as cur:
                cur.execute("SELECT version(), current_database(), current_user;")
                version, db_name, db_user = cur.fetchone()
                logger.info(f"✅ Database test successful:")
                logger.info(f"   Database: {db_name}")
                logger.info(f"   User: {db_user}")
                logger.info(f"   Version: {version[:100]}...")
            connection_pool.putconn(test_conn)
            return True
            
        except Exception as test_error:
            logger.error(f"🚨 Database connection test failed: {test_error}")
            if test_conn:
                connection_pool.putconn(test_conn)
            return False
            
    except Exception as e:
        logger.error(f"🚨 Error creating connection pool: {e}")
        logger.error(f"Database URL (masked): {DATABASE_URL[:50]}...")
        connection_pool = None
        
        # Additional debugging info
        if is_production():
            logger.error("Running in production mode - Railway environment detected")
        else:
            logger.error("Running in development mode")
            
        return False

@contextmanager
def get_db_connection():
    """Context manager untuk database connections dengan error handling"""
    if connection_pool is None:
        logger.error("🚨 Database connection pool not available")
        raise HTTPException(
            status_code=503, 
            detail={
                "error": "Database connection not available",
                "suggestion": "Check Railway PostgreSQL service configuration",
                "is_production": is_production()
            }
        )
    
    conn = None
    try:
        conn = connection_pool.getconn()
        if conn is None:
            raise Exception("Failed to get connection from pool")
        
        # Test if connection is still alive
        conn.poll()
        if conn.closed:
            raise Exception("Connection is closed")
            
        yield conn
        
    except psycopg2.Error as e:
        logger.error(f"🚨 PostgreSQL error: {e}")
        raise HTTPException(
            status_code=503,
            detail={
                "error": f"Database error: {str(e)}",
                "suggestion": "Database connectivity issue - check Railway PostgreSQL service"
            }
        )
    except Exception as e:
        logger.error(f"🚨 Database connection error: {e}")
        raise HTTPException(
            status_code=503,
            detail={
                "error": f"Database operation failed: {str(e)}",
                "suggestion": "Check database connectivity and try again"
            }
        )
    finally:
        if conn and connection_pool:
            try:
                connection_pool.putconn(conn)
            except Exception as e:
                logger.error(f"Error returning connection to pool: {e}")

def create_tables():
    """Create tables if they don't exist dengan improved error handling"""
    if not connection_pool:
        logger.error("🚨 Cannot create tables: connection pool not available")
        return False
        
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
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    -- Table untuk n8n results tracking
    CREATE TABLE IF NOT EXISTS disc_results_n8n (
        id SERIAL PRIMARY KEY,
        assessment_id VARCHAR(50) NOT NULL,
        candidate_name VARCHAR(255) NOT NULL,
        candidate_email VARCHAR(255) NOT NULL,
        position VARCHAR(255),
        primary_style VARCHAR(10) NOT NULL,
        d_score FLOAT NOT NULL,
        i_score FLOAT NOT NULL,
        s_score FLOAT NOT NULL,
        c_score FLOAT NOT NULL,
        completed_at TIMESTAMP NOT NULL,
        processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    -- Table untuk webhook logs
    CREATE TABLE IF NOT EXISTS webhook_logs (
        id SERIAL PRIMARY KEY,
        assessment_id VARCHAR(50),
        webhook_url TEXT,
        payload JSONB,
        response_status INTEGER,
        response_body TEXT,
        error_message TEXT,
        attempt_number INTEGER DEFAULT 1,
        sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        success BOOLEAN DEFAULT FALSE
    );
    
    -- Indexes untuk performance
    CREATE INDEX IF NOT EXISTS idx_assessment_id ON disc_assessments(assessment_id);
    CREATE INDEX IF NOT EXISTS idx_candidate_email ON disc_assessments(candidate_email);
    CREATE INDEX IF NOT EXISTS idx_created_at ON disc_assessments(created_at);
    CREATE INDEX IF NOT EXISTS idx_primary_style ON disc_assessments(primary_style);
    
    CREATE INDEX IF NOT EXISTS idx_n8n_assessment_id ON disc_results_n8n(assessment_id);
    CREATE INDEX IF NOT EXISTS idx_n8n_candidate_email ON disc_results_n8n(candidate_email);
    CREATE INDEX IF NOT EXISTS idx_n8n_completed_at ON disc_results_n8n(completed_at);
    CREATE INDEX IF NOT EXISTS idx_n8n_primary_style ON disc_results_n8n(primary_style);
    
    CREATE INDEX IF NOT EXISTS idx_webhook_logs_assessment_id ON webhook_logs(assessment_id);
    CREATE INDEX IF NOT EXISTS idx_webhook_logs_success ON webhook_logs(success);
    CREATE INDEX IF NOT EXISTS idx_webhook_logs_sent_at ON webhook_logs(sent_at);
    
    -- Trigger untuk updated_at
    CREATE OR REPLACE FUNCTION update_updated_at_column()
    RETURNS TRIGGER AS $$
    BEGIN
        NEW.updated_at = CURRENT_TIMESTAMP;
        RETURN NEW;
    END;
    $$ language 'plpgsql';
    
    DROP TRIGGER IF EXISTS update_disc_assessments_updated_at ON disc_assessments;
    CREATE TRIGGER update_disc_assessments_updated_at
        BEFORE UPDATE ON disc_assessments
        FOR EACH ROW
        EXECUTE FUNCTION update_updated_at_column();
    """
    
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(create_table_sql)
                conn.commit()
        logger.info("✅ Database tables created/verified successfully")
        return True
    except Exception as e:
        logger.error(f"🚨 Error creating tables: {e}")
        return False

# ================== WEBHOOK FUNCTIONS ==================

async def log_webhook_attempt(assessment_id: str, webhook_url: str, payload: dict, 
                             response_status: int = None, response_body: str = None, 
                             error_message: str = None, attempt_number: int = 1, 
                             success: bool = False):
    """Log webhook attempt to database for debugging"""
    if not connection_pool:
        return
    
    insert_sql = """
    INSERT INTO webhook_logs (
        assessment_id, webhook_url, payload, response_status, 
        response_body, error_message, attempt_number, success
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """
    
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(insert_sql, (
                    assessment_id,
                    webhook_url,
                    json.dumps(payload),
                    response_status,
                    response_body[:1000] if response_body else None,  # Limit response body
                    error_message[:500] if error_message else None,   # Limit error message
                    attempt_number,
                    success
                ))
                conn.commit()
    except Exception as e:
        logger.error(f"🚨 Error logging webhook attempt: {e}")

async def send_webhook_to_n8n(assessment_data: dict, webhook_url: str = None):
    """
    Send assessment data to n8n webhook asynchronously dengan comprehensive logging
    """
    assessment_id = assessment_data.get("assessment_id", "unknown")
    
    if not webhook_config["webhook_enabled"]:
        logger.info(f"📤 Webhook disabled for assessment {assessment_id}")
        return True
    
    url = webhook_url or webhook_config["n8n_webhook_url"]
    if not url:
        logger.warning(f"⚠️ No webhook URL configured for assessment {assessment_id}")
        return False
    
    # Prepare webhook payload
    payload = {
        "event_type": "disc_assessment_completed",
        "timestamp": datetime.now().isoformat(),
        "assessment_id": assessment_data.get("assessment_id"),
        "candidate": {
            "name": assessment_data.get("candidate_name"),
            "email": assessment_data.get("candidate_email"),
            "position": assessment_data.get("position", "")
        },
        "results": {
            "primary_style": assessment_data.get("primary_style"),
            "raw_scores": assessment_data.get("raw_scores"),
            "normalized_scores": assessment_data.get("normalized_scores"),
            "relative_percentages": assessment_data.get("relative_percentages"),
            "resultant_angle": assessment_data.get("resultant_angle"),
            "resultant_magnitude": assessment_data.get("resultant_magnitude"),
            "style_description": assessment_data.get("style_description")
        },
        "metadata": {
            "questions_count": len(assessment_data.get("questions_used", [])),
            "assessment_completed_at": assessment_data.get("timestamp"),
            "source": "disc-assessment-api",
            "version": "2.1.0"
        }
    }
    
    # Add webhook secret if configured
    if webhook_config["webhook_secret"]:
        payload["webhook_secret"] = webhook_config["webhook_secret"]
    
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "DISC-Assessment-API/2.1.0",
        "X-Webhook-Source": "disc-assessment"
    }
    
    if webhook_config["webhook_secret"]:
        headers["X-Webhook-Secret"] = webhook_config["webhook_secret"]
    
    attempt = 1
    max_attempts = webhook_config["webhook_retry_attempts"]
    
    while attempt <= max_attempts:
        try:
            logger.info(f"📤 Sending webhook to n8n (attempt {attempt}/{max_attempts}) for {assessment_id}")
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=webhook_config["webhook_timeout"])
                ) as response:
                    
                    response_text = await response.text()
                    
                    # Log the attempt
                    await log_webhook_attempt(
                        assessment_id=assessment_id,
                        webhook_url=url,
                        payload=payload,
                        response_status=response.status,
                        response_body=response_text,
                        attempt_number=attempt,
                        success=response.status in [200, 201, 202]
                    )
                    
                    if response.status in [200, 201, 202]:
                        logger.info(f"✅ Webhook sent successfully for {assessment_id} (status: {response.status})")
                        return True
                    else:
                        logger.warning(f"⚠️ Webhook failed for {assessment_id} with status {response.status}")
                        
                        if response.status >= 500 and attempt < max_attempts:
                            wait_time = 2 ** attempt
                            logger.info(f"🔄 Retrying in {wait_time} seconds...")
                            await asyncio.sleep(wait_time)
                            attempt += 1
                            continue
                        
                        return False
                        
        except asyncio.TimeoutError:
            error_msg = f"Webhook timeout (attempt {attempt}/{max_attempts})"
            logger.warning(f"⏰ {error_msg} for {assessment_id}")
            
            await log_webhook_attempt(
                assessment_id=assessment_id,
                webhook_url=url,
                payload=payload,
                error_message=error_msg,
                attempt_number=attempt,
                success=False
            )
            
            if attempt < max_attempts:
                await asyncio.sleep(2 ** attempt)
                attempt += 1
                continue
            return False
            
        except Exception as e:
            error_msg = f"Webhook error: {str(e)}"
            logger.error(f"🚨 {error_msg} for {assessment_id} (attempt {attempt}/{max_attempts})")
            
            await log_webhook_attempt(
                assessment_id=assessment_id,
                webhook_url=url,
                payload=payload,
                error_message=error_msg,
                attempt_number=attempt,
                success=False
            )
            
            if attempt < max_attempts:
                await asyncio.sleep(2 ** attempt)
                attempt += 1
                continue
            return False
    
    logger.error(f"🚨 Webhook failed after {max_attempts} attempts for {assessment_id}")
    return False

def send_webhook_sync(assessment_data: dict):
    """
    Synchronous wrapper for webhook sending (fallback)
    """
    try:
        url = webhook_config["n8n_webhook_url"]
        if not url or not webhook_config["webhook_enabled"]:
            return False
        
        payload = {
            "event_type": "disc_assessment_completed",
            "timestamp": datetime.now().isoformat(),
            "assessment_id": assessment_data.get("assessment_id"),
            "candidate": {
                "name": assessment_data.get("candidate_name"),
                "email": assessment_data.get("candidate_email"),
                "position": assessment_data.get("position", "")
            },
            "results": {
                "primary_style": assessment_data.get("primary_style"),
                "raw_scores": assessment_data.get("raw_scores"),
                "normalized_scores": assessment_data.get("normalized_scores"),
                "relative_percentages": assessment_data.get("relative_percentages"),
                "resultant_angle": assessment_data.get("resultant_angle"),
                "resultant_magnitude": assessment_data.get("resultant_magnitude"),
                "style_description": assessment_data.get("style_description")
            },
            "metadata": {
                "questions_count": len(assessment_data.get("questions_used", [])),
                "assessment_completed_at": assessment_data.get("timestamp"),
                "source": "disc-assessment-api",
                "version": "2.1.0"
            }
        }
        
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "DISC-Assessment-API/2.1.0",
            "X-Webhook-Source": "disc-assessment"
        }
        
        if webhook_config["webhook_secret"]:
            payload["webhook_secret"] = webhook_config["webhook_secret"]
            headers["X-Webhook-Secret"] = webhook_config["webhook_secret"]
        
        response = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=webhook_config["webhook_timeout"]
        )
        
        if response.status_code in [200, 201, 202]:
            logger.info(f"✅ Webhook sent successfully (sync) to n8n")
            return True
        else:
            logger.warning(f"⚠️ Webhook failed (sync) with status {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"🚨 Webhook sync error: {e}")
        return False

# ================== PYDANTIC MODELS ==================

class DISCAnswer(BaseModel):
    question_id: int
    answer: int  # 1-5 scale

class DISCAssessmentRequest(BaseModel):
    candidate_name: str
    candidate_email: str
    position: str = ""
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

# ================== DATA LOADING ==================

def load_questions():
    """Load questions exactly as in original app"""
    try:
        with open("questions.json", "r", encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"✅ Loaded {len(data)} questions from questions.json")
        return data
    except FileNotFoundError:
        logger.error("🚨 questions.json not found")
        return []
    except Exception as e:
        logger.error(f"🚨 Error loading questions: {e}")
        return []

def load_disc_descriptions():
    """Load DISC descriptions exactly as in original app"""
    try:
        with open("disc_descriptions.json", "r", encoding='utf-8') as f:
            data = json.load(f)
        logger.info("✅ Loaded DISC descriptions from disc_descriptions.json")
        return data
    except FileNotFoundError:
        logger.error("🚨 disc_descriptions.json not found")
        return {"single": {}}
    except Exception as e:
        logger.error(f"🚨 Error loading DISC descriptions: {e}")
        return {"single": {}}

# Load data
questions_data = load_questions()
disc_descriptions = load_disc_descriptions()

# ================== DISC LOGIC FUNCTIONS (EXACT COPY) ==================

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

# ================== DATABASE OPERATIONS ==================

def save_assessment_to_db(result: DISCResult):
    """Save assessment result to PostgreSQL dengan improved error handling"""
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
        logger.info(f"✅ Assessment {result.assessment_id} saved to database")
        return True
    except Exception as e:
        logger.error(f"🚨 Error saving assessment to database: {e}")
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
        logger.error(f"🚨 Error getting assessment from database: {e}")
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
        logger.error(f"🚨 Error getting assessments by email: {e}")
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
        logger.error(f"🚨 Error getting all assessments: {e}")
        return [], 0

# ================== API ENDPOINTS ==================

@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    logger.info("🚀 Starting DISC Assessment API with n8n webhook integration...")
    
    success = init_connection_pool()
    if success:
        create_tables()
        logger.info("✅ API startup completed successfully")
        
        # Log webhook configuration
        logger.info("🔗 Webhook configuration:")
        logger.info(f"   Enabled: {webhook_config['webhook_enabled']}")
        if webhook_config['webhook_enabled']:
            masked_url = webhook_config['n8n_webhook_url']
            if masked_url and len(masked_url) > 50:
                masked_url = masked_url[:50] + "..."
            logger.info(f"   URL: {masked_url}")
            logger.info(f"   Timeout: {webhook_config['webhook_timeout']}s")
            logger.info(f"   Retry attempts: {webhook_config['webhook_retry_attempts']}")
            logger.info(f"   Secret configured: {bool(webhook_config['webhook_secret'])}")
    else:
        logger.error("🚨 API startup completed with database issues")

@app.get("/")
async def root():
    """Root endpoint dengan informasi lengkap"""
    db_status = "connected" if connection_pool else "disconnected"
    
    return {
        "message": "DISC Assessment API - Railway PostgreSQL with n8n Webhook",
        "version": "2.1.0",
        "questions_loaded": len(questions_data),
        "descriptions_loaded": len(disc_descriptions.get("single", {})),
        "database_status": db_status,
        "is_production": is_production(),
        "webhook": {
            "enabled": webhook_config["webhook_enabled"],
            "configured": bool(webhook_config["n8n_webhook_url"])
        },
        "endpoints": {
            "submit_assessment": "POST /api/v1/assessment/submit",
            "get_result": "GET /api/v1/assessment/{assessment_id}",
            "list_assessments": "GET /api/v1/assessments",
            "get_by_email": "GET /api/v1/assessments/by-email/{email}",
            "get_questions": "GET /api/v1/questions",
            "webhook_test": "POST /api/v1/webhook/test",
            "webhook_config": "GET /api/v1/webhook/config",
            "webhook_logs": "GET /api/v1/webhook/logs",
            "webhook_stats": "GET /api/v1/webhook/stats",
            "health": "GET /health",
            "debug_env": "GET /debug/environment",
            "debug_db": "GET /debug/database"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint dengan webhook status"""
    db_status = "connected" if connection_pool else "disconnected"
    
    # Test database jika tersedia
    db_test = False
    if connection_pool:
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    result = cur.fetchone()
                    db_test = result[0] == 1
        except:
            db_test = False
    
    return {
        "status": "healthy" if db_test else "degraded",
        "timestamp": datetime.now().isoformat(),
        "database_pool": db_status,
        "database_test": "passed" if db_test else "failed",
        "questions_available": len(questions_data) > 0,
        "descriptions_available": len(disc_descriptions.get("single", {})) > 0,
        "is_production": is_production(),
        "webhook": {
            "enabled": webhook_config["webhook_enabled"],
            "configured": bool(webhook_config["n8n_webhook_url"]),
            "url": webhook_config["n8n_webhook_url"][:50] + "..." if webhook_config["n8n_webhook_url"] and len(webhook_config["n8n_webhook_url"]) > 50 else webhook_config["n8n_webhook_url"]
        }
    }

@app.get("/debug/environment")
async def debug_environment():
    """Debug endpoint untuk environment variables"""
    if is_production():
        return {"message": "Debug endpoint disabled in production"}
    
    env_vars = {}
    for key in sorted(os.environ.keys()):
        if any(term in key.upper() for term in ['DATABASE', 'POSTGRES', 'PG', 'RAILWAY', 'PORT', 'SERVICE', 'WEBHOOK', 'N8N']):
            value = os.environ[key]
            # Hide sensitive data
            if any(sensitive in key.lower() for sensitive in ['password', 'secret', 'key', 'token']):
                value = "***MASKED***"
            elif 'url' in key.lower() and len(value) > 50:
                value = value[:30] + "..." + value[-10:]
            env_vars[key] = value
    
    return {
        "environment_variables": env_vars,
        "database_pool_status": "connected" if connection_pool else "disconnected",
        "is_production": is_production(),
        "questions_loaded": len(questions_data),
        "descriptions_loaded": len(disc_descriptions.get("single", {})),
        "webhook_config": {
            "enabled": webhook_config["webhook_enabled"],
            "url_configured": bool(webhook_config["n8n_webhook_url"]),
            "secret_configured": bool(webhook_config["webhook_secret"])
        }
    }

@app.get("/debug/database")
async def debug_database():
    """Test database connection dan informasi"""
    if not connection_pool:
        return {
            "status": "error",
            "error": "Connection pool not initialized",
            "database_url_configured": DATABASE_URL is not None
        }
    
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Basic database info
                cur.execute("SELECT version(), current_database(), current_user;")
                version, db_name, db_user = cur.fetchone()
                
                # Table count
                cur.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';")
                table_count = cur.fetchone()[0]
                
                # Assessment count
                try:
                    cur.execute("SELECT COUNT(*) FROM disc_assessments;")
                    assessment_count = cur.fetchone()[0]
                except:
                    assessment_count = "Table not found"
                
                # Webhook logs count
                try:
                    cur.execute("SELECT COUNT(*) FROM webhook_logs;")
                    webhook_logs_count = cur.fetchone()[0]
                except:
                    webhook_logs_count = "Table not found"
                
                return {
                    "status": "connected",
                    "database_info": {
                        "version": version,
                        "database": db_name,
                        "user": db_user,
                        "tables_count": table_count,
                        "assessments_count": assessment_count,
                        "webhook_logs_count": webhook_logs_count
                    },
                    "connection_pool": "active",
                    "ssl_mode": get_ssl_mode()
                }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "database_url_configured": DATABASE_URL is not None,
            "connection_pool_status": "connected" if connection_pool else "disconnected"
        }

@app.get("/api/v1/questions")
async def get_random_questions(count: int = 30):
    """Get random questions exactly like the original app"""
    if not questions_data:
        raise HTTPException(status_code=503, detail="Questions data not available")
    
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
    """Process DISC assessment, simpan ke PostgreSQL, dan kirim webhook ke n8n"""
    try:
        # Validate input
        if not questions_data:
            raise HTTPException(status_code=503, detail="Questions data not available")
        
        if len(request.answers) < 20:
            raise HTTPException(status_code=400, detail="Minimum 20 answers required")
        
        if len(request.answers) > len(questions_data):
            raise HTTPException(status_code=400, detail=f"Too many answers. Maximum: {len(questions_data)}")
        
        # Generate unique assessment ID
        assessment_id = str(uuid.uuid4())
        logger.info(f"Processing assessment {assessment_id} for {request.candidate_email}")
        
        # Prepare questions (sama seperti original)
        shuffled_questions = questions_data.copy()
        random.shuffle(shuffled_questions)
        selected_questions = shuffled_questions[:len(request.answers)]
        
        # Calculate raw scores
        raw_scores = {"D": 0, "I": 0, "S": 0, "C": 0}
        
        for i, answer_data in enumerate(request.answers):
            if i >= len(selected_questions):
                break
                
            q = selected_questions[i]
            answer = answer_data.answer
            
            # Validate answer range
            if answer < 1 or answer > 5:
                raise HTTPException(status_code=400, detail=f"Answer {answer} out of range (1-5)")
            
            for style in ["D", "I", "S", "C"]:
                raw_scores[style] += q["mapping"][style] * (answer - 3)
        
        # Normalize scores
        normalized_scores = normalize_scores(raw_scores, selected_questions)
        
        # Calculate relative percentages
        total_normalized = sum(normalized_scores.values())
        relative_percentages = {}
        for style, score in normalized_scores.items():
            if total_normalized == 0:
                relative_percentages[style] = 0
            else:
                relative_percentages[style] = (score / total_normalized) * 100
        
        # Calculate resultant vector
        resultant_angle, resultant_magnitude = calculate_resultant_vector(normalized_scores)
        
        # Determine primary style
        primary_style = determine_style_from_angle(resultant_angle)
        style_description = get_style_description(primary_style)
        
        # Create result object
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
        save_success = save_assessment_to_db(result)
        if not save_success:
            logger.warning(f"Failed to save assessment {assessment_id} to database, but returning result")
        
        # 🆕 SEND WEBHOOK TO N8N
        try:
            webhook_data = result.dict()
            webhook_success = await send_webhook_to_n8n(webhook_data)
            
            if webhook_success:
                logger.info(f"✅ Webhook sent to n8n for assessment {assessment_id}")
            else:
                logger.warning(f"⚠️ Failed to send webhook to n8n for assessment {assessment_id}")
                
        except Exception as webhook_error:
            logger.error(f"🚨 Webhook error for assessment {assessment_id}: {webhook_error}")
            # Don't fail the assessment if webhook fails
        
        logger.info(f"✅ Assessment {assessment_id} processed successfully for {request.candidate_email}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"🚨 Error processing assessment: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing assessment: {str(e)}")

@app.get("/api/v1/assessment/{assessment_id}", response_model=DISCResult)
async def get_assessment_result(assessment_id: str):
    """Get assessment result by ID dari PostgreSQL"""
    if not assessment_id or len(assessment_id) < 10:
        raise HTTPException(status_code=400, detail="Invalid assessment ID")
    
    row = get_assessment_from_db(assessment_id)
    if not row:
        raise HTTPException(status_code=404, detail="Assessment not found")
    
    try:
        return DISCResult(
            assessment_id=row['assessment_id'],
            candidate_name=row['candidate_name'],
            candidate_email=row['candidate_email'],
            position=row['position'] or "",
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
    except Exception as e:
        logger.error(f"🚨 Error formatting assessment result: {e}")
        raise HTTPException(status_code=500, detail="Error formatting assessment result")

@app.get("/api/v1/assessments")
async def list_assessments(limit: int = 50, offset: int = 0):
    """List all assessments with pagination dari PostgreSQL"""
    # Validate parameters
    if limit < 1 or limit > 200:
        raise HTTPException(status_code=400, detail="Limit must be between 1 and 200")
    
    if offset < 0:
        raise HTTPException(status_code=400, detail="Offset must be non-negative")
    
    try:
        assessments, total = get_all_assessments_from_db(limit, offset)
        
        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "count": len(assessments),
            "has_more": (offset + len(assessments)) < total,
            "assessments": assessments
        }
    except Exception as e:
        logger.error(f"🚨 Error listing assessments: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving assessments")

@app.get("/api/v1/assessments/by-email/{email}")
async def get_assessments_by_email(email: str):
    """Get all assessments for a specific email dari PostgreSQL"""
    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="Invalid email address")
    
    try:
        assessments = get_assessments_by_email_from_db(email)
        
        if not assessments:
            raise HTTPException(status_code=404, detail="No assessments found for this email")
        
        return {
            "email": email,
            "count": len(assessments),
            "assessments": assessments
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"🚨 Error getting assessments by email: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving assessments")

@app.get("/api/v1/statistics")
async def get_statistics():
    """Get basic statistics about assessments"""
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Total assessments
                cur.execute("SELECT COUNT(*) as total FROM disc_assessments")
                total = cur.fetchone()['total']
                
                # Assessments by style
                cur.execute("""
                    SELECT primary_style, COUNT(*) as count 
                    FROM disc_assessments 
                    GROUP BY primary_style 
                    ORDER BY count DESC
                """)
                by_style = cur.fetchall()
                
                # Recent assessments (last 30 days)
                cur.execute("""
                    SELECT COUNT(*) as recent 
                    FROM disc_assessments 
                    WHERE created_at >= NOW() - INTERVAL '30 days'
                """)
                recent = cur.fetchone()['recent']
                
                # Average scores
                cur.execute("""
                    SELECT 
                        AVG((normalized_scores->>'D')::float) as avg_d,
                        AVG((normalized_scores->>'I')::float) as avg_i,
                        AVG((normalized_scores->>'S')::float) as avg_s,
                        AVG((normalized_scores->>'C')::float) as avg_c
                    FROM disc_assessments
                """)
                avg_scores = cur.fetchone()
                
                return {
                    "total_assessments": total,
                    "recent_assessments_30d": recent,
                    "by_primary_style": [dict(row) for row in by_style],
                    "average_scores": {
                        "D": round(float(avg_scores['avg_d'] or 0), 2),
                        "I": round(float(avg_scores['avg_i'] or 0), 2),
                        "S": round(float(avg_scores['avg_s'] or 0), 2),
                        "C": round(float(avg_scores['avg_c'] or 0), 2)
                    },
                    "last_updated": datetime.now().isoformat()
                }
    except Exception as e:
        logger.error(f"🚨 Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving statistics")

@app.delete("/api/v1/assessment/{assessment_id}")
async def delete_assessment(assessment_id: str):
    """Delete an assessment (admin function)"""
    if not assessment_id or len(assessment_id) < 10:
        raise HTTPException(status_code=400, detail="Invalid assessment ID")
    
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM disc_assessments WHERE assessment_id = %s", (assessment_id,))
                deleted_count = cur.rowcount
                conn.commit()
                
                if deleted_count == 0:
                    raise HTTPException(status_code=404, detail="Assessment not found")
                
                logger.info(f"✅ Assessment {assessment_id} deleted")
                return {"message": "Assessment deleted successfully", "assessment_id": assessment_id}
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"🚨 Error deleting assessment: {e}")
        raise HTTPException(status_code=500, detail="Error deleting assessment")

# ================== WEBHOOK ENDPOINTS ==================

@app.post("/api/v1/webhook/test")
async def test_webhook(test_data: Optional[dict] = None):
    """Test webhook connectivity to n8n"""
    if not webhook_config["webhook_enabled"]:
        return {
            "status": "disabled",
            "message": "Webhook is disabled",
            "config": {
                "webhook_enabled": False,
                "webhook_url": "not configured" if not webhook_config["n8n_webhook_url"] else "configured"
            }
        }
    
    if not webhook_config["n8n_webhook_url"]:
        return {
            "status": "error",
            "message": "Webhook URL not configured",
            "config": {
                "webhook_enabled": webhook_config["webhook_enabled"],
                "webhook_url": "not configured"
            }
        }
    
    # Create test data
    test_payload = test_data or {
        "event_type": "webhook_test",
        "timestamp": datetime.now().isoformat(),
        "assessment_id": "test-" + str(uuid.uuid4())[:8],
        "candidate": {
            "name": "Test User",
            "email": "test@example.com",
            "position": "Test Position"
        },
        "results": {
            "primary_style": "D",
            "raw_scores": {"D": 10, "I": 5, "S": 3, "C": 7},
            "normalized_scores": {"D": 75, "I": 45, "S": 35, "C": 60},
            "relative_percentages": {"D": 35.0, "I": 21.0, "S": 16.0, "C": 28.0}
        },
        "metadata": {
            "source": "webhook-test",
            "version": "2.1.0"
        }
    }
    
    try:
        success = await send_webhook_to_n8n(test_payload)
        
        return {
            "status": "success" if success else "failed",
            "message": "Test webhook sent successfully" if success else "Test webhook failed",
            "webhook_url": webhook_config["n8n_webhook_url"],
            "test_payload": test_payload
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Test webhook error: {str(e)}",
            "webhook_url": webhook_config["n8n_webhook_url"]
        }

@app.get("/api/v1/webhook/config")
async def get_webhook_config_endpoint():
    """Get current webhook configuration"""
    return {
        "webhook_enabled": webhook_config["webhook_enabled"],
        "webhook_url_configured": bool(webhook_config["n8n_webhook_url"]),
        "webhook_url": webhook_config["n8n_webhook_url"][:50] + "..." if webhook_config["n8n_webhook_url"] and len(webhook_config["n8n_webhook_url"]) > 50 else webhook_config["n8n_webhook_url"],
        "webhook_timeout": webhook_config["webhook_timeout"],
        "webhook_retry_attempts": webhook_config["webhook_retry_attempts"],
        "webhook_secret_configured": bool(webhook_config["webhook_secret"])
    }

@app.get("/api/v1/webhook/logs")
async def get_webhook_logs(limit: int = 50, assessment_id: str = None, success_only: bool = False):
    """Get webhook logs for debugging"""
    try:
        base_sql = """
        SELECT assessment_id, webhook_url, response_status, error_message, 
               attempt_number, success, sent_at
        FROM webhook_logs
        WHERE 1=1
        """
        params = []
        
        if assessment_id:
            base_sql += " AND assessment_id = %s"
            params.append(assessment_id)
        
        if success_only:
            base_sql += " AND success = true"
        
        base_sql += " ORDER BY sent_at DESC LIMIT %s"
        params.append(limit)
        
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(base_sql, params)
                logs = cur.fetchall()
                
                return {
                    "total": len(logs),
                    "limit": limit,
                    "filters": {
                        "assessment_id": assessment_id,
                        "success_only": success_only
                    },
                    "logs": [dict(log) for log in logs]
                }
                
    except Exception as e:
        logger.error(f"🚨 Error getting webhook logs: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving webhook logs")

@app.get("/api/v1/webhook/stats")
async def get_webhook_stats():
    """Get webhook statistics"""
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Total webhooks
                cur.execute("SELECT COUNT(*) as total FROM webhook_logs")
                total = cur.fetchone()['total']
                
                if total == 0:
                    return {
                        "total_webhooks": 0,
                        "successful_webhooks": 0,
                        "failed_webhooks": 0,
                        "success_rate": 0,
                        "recent_24h": 0,
                        "top_errors": [],
                        "last_updated": datetime.now().isoformat()
                    }
                
                # Success rate
                cur.execute("SELECT COUNT(*) as successful FROM webhook_logs WHERE success = true")
                successful = cur.fetchone()['successful']
                
                # Recent webhooks (last 24h)
                cur.execute("""
                    SELECT COUNT(*) as recent 
                    FROM webhook_logs 
                    WHERE sent_at >= NOW() - INTERVAL '24 hours'
                """)
                recent = cur.fetchone()['recent']
                
                # Failed webhooks by error
                cur.execute("""
                    SELECT error_message, COUNT(*) as count
                    FROM webhook_logs 
                    WHERE success = false AND error_message IS NOT NULL
                    GROUP BY error_message
                    ORDER BY count DESC
                    LIMIT 5
                """)
                errors = cur.fetchall()
                
                success_rate = (successful / total * 100) if total > 0 else 0
                
                return {
                    "total_webhooks": total,
                    "successful_webhooks": successful,
                    "failed_webhooks": total - successful,
                    "success_rate": round(success_rate, 2),
                    "recent_24h": recent,
                    "top_errors": [dict(error) for error in errors],
                    "last_updated": datetime.now().isoformat()
                }
                
    except Exception as e:
        logger.error(f"🚨 Error getting webhook stats: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving webhook statistics")

# Error handlers
@app.exception_handler(psycopg2.Error)
async def database_exception_handler(request, exc):
    logger.error(f"🚨 Database error: {exc}")
    return HTTPException(
        status_code=503,
        detail={
            "error": "Database service temporarily unavailable",
            "suggestion": "Please try again in a few moments"
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"🚨 Unexpected error: {exc}")
    return HTTPException(
        status_code=500,
        detail={
            "error": "Internal server error",
            "message": str(exc) if not is_production() else "An unexpected error occurred"
        }
    )

# Startup validation
if __name__ == "__main__":
    import uvicorn
    
    # Validate required files
    required_files = ["questions.json", "disc_descriptions.json"]
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        logger.error(f"🚨 Missing required files: {missing_files}")
        sys.exit(1)
    
    logger.info("🚀 Starting DISC Assessment API server with n8n webhook integration...")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=int(os.getenv("PORT", 8000)),
        log_level="info"
    )
