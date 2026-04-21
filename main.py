from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
import logging
from datetime import datetime
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ZeonWise Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Database Config (supports env vars for cloud deployment)
DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "database": os.environ.get("DB_NAME", "zeonwisedb"),
    "user": os.environ.get("DB_USER", "postgres"),
    "password": os.environ.get("DB_PASSWORD", "1234"),
    "port": int(os.environ.get("DB_PORT", "5432"))
}

# --- Request/Response Models ---
class LoginRequest(BaseModel):
    staff_id: str
    password: str

class StaffData(BaseModel):
    staff_id: str
    name: str
    email: str
    department: str
    role: str = "staff"

class LoginResponse(BaseModel):
    success: bool
    message: str
    staff: Optional[StaffData] = None

class CheckInRequest(BaseModel):
    staff_id: str
    latitude: float
    longitude: float
    emotion: str
    timestamp: Optional[str] = None

class CheckInResponse(BaseModel):
    success: bool
    message: str
    attendance_id: Optional[int] = None
    emotion_feedback: Optional[str] = None

class FaceVerifyRequest(BaseModel):
    staff_id: str
    face_encoding: str  # Comma-separated string of floats

class FaceVerifyResponse(BaseModel):
    success: bool
    message: str
    is_match: bool
    similarity: Optional[float] = None

# --- Database Helper ---
def get_db_connection():
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True  # ✅ Critical: Ensure INSERT/UPDATE commits immediately
    return conn

# --- Endpoints ---
@app.get("/")
def read_root():
    return {
        "message": "ZeonWise Backend is running! 🚀",
        "endpoints": {
            "login": "POST /api/login",
            "checkin": "POST /api/checkin", 
            "face_verify": "POST /api/face/verify"
        }
    }

@app.post("/api/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    try:
        logger.info(f"🔍 Login attempt: {request.staff_id}")
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(
            "SELECT staff_id, name, email, department, role FROM staff WHERE staff_id = %s AND password = %s AND is_active = TRUE",
            (request.staff_id, request.password)
        )
        staff = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if staff:
            logger.info(f"✅ Login successful: {request.staff_id}")
            return LoginResponse(
                success=True, 
                message="Login successful", 
                staff=StaffData(**staff)
            )
        else:
            logger.warning(f"❌ Login failed: {request.staff_id}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, 
                detail="Invalid Staff ID or Password"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"💥 Login error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="Server error"
        )

@app.post("/api/checkin", response_model=CheckInResponse)
async def check_in(request: CheckInRequest):
    try:
        logger.info(f"📍 Check-in: {request.staff_id}")
        conn = get_db_connection()
        cursor = conn.cursor()
        
        timestamp = request.timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        cursor.execute(
            """INSERT INTO attendance 
               (staff_id, check_in_time, location_lat, location_lng, emotion, status) 
               VALUES (%s, %s, %s, %s, %s, 'present') 
               RETURNING id, check_in_time""",
            (request.staff_id, timestamp, request.latitude, request.longitude, request.emotion)
        )
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        
        # Motivational messages
        messages = {
            "HAPPY": "🌟 You're glowing today! Keep up the great energy! 💪",
            "SAD": "💙 Tough days don't last, tough people do. You've got this! 🤗",
            "NEUTRAL": "⚡ Focus mode activated! Let's make today productive! 🎯",
            "SURPRISED": "🎉 Ready for surprises? Let's crush today's goals! 🚀",
            "ANGRY": "🔥 Channel that energy into success! You're unstoppable! 💪"
        }
        motivation = messages.get(request.emotion, "✨ Every day is a new opportunity. Let's make it count! 🌈")
        
        return CheckInResponse(
            success=True, 
            message="Check-in successful", 
            attendance_id=result[0], 
            emotion_feedback=motivation
        )
    except Exception as e:
        logger.error(f"💥 Check-in error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="Check-in failed"
        )

@app.post("/api/face/verify", response_model=FaceVerifyResponse)
async def verify_face(request: FaceVerifyRequest):
    try:
        logger.info(f"🔍 Face verify: {request.staff_id}")
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT face_encoding FROM staff WHERE staff_id = %s", 
            (request.staff_id,)
        )
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        
        # ✅ Handle first-time enrollment (no face stored yet)
        if not result or not result[0]:
            return FaceVerifyResponse(
                success=True, 
                is_match=True, 
                similarity=None, 
                message="No face enrolled. First-time check-in allowed."
            )
        
        # ✅ Parse live encoding (comma-separated string from Android)
        live_encoding = np.array([float(x.strip()) for x in request.face_encoding.split(',')])
        
        # ✅ Parse stored encoding (handle both string and binary formats)
        stored_data = result[0]
        if isinstance(stored_data, str) and ',' in stored_data:
            # Comma-separated string format
            stored_encoding = np.array([float(x.strip()) for x in stored_data.split(',')])
        else:
            # Fallback: try binary format
            stored_encoding = np.frombuffer(stored_data, dtype=np.float64)
        
        # ✅ Cosine similarity calculation
        norm_live = np.linalg.norm(live_encoding)
        norm_stored = np.linalg.norm(stored_encoding)
        
        if norm_live == 0 or norm_stored == 0:
            similarity = 0.0
        else:
            similarity = np.dot(live_encoding, stored_encoding) / (norm_live * norm_stored)
        
        is_match = similarity >= 0.6  # 60% threshold
        logger.info(f"Similarity: {similarity:.3f} - Match: {is_match}")
        
        return FaceVerifyResponse(
            success=True, 
            is_match=bool(is_match), 
            similarity=float(similarity), 
            message="Face verified" if is_match else "Face does not match"
        )
    except Exception as e:
        logger.error(f"💥 Face verify error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="Face verification failed"
        )