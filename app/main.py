import os
import numpy as np
import cv2
import math
import tempfile
import re
import hashlib
import smtplib
import io
import csv
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from insightface.app import FaceAnalysis
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from psycopg2 import OperationalError
from .face_processor import FaceProcessor
from .face_vector import FaceEmbeddingDB

load_dotenv()

# Environment variables validation
MAX_DISTANCE = os.getenv('MAX_DISTANCE')
if not MAX_DISTANCE:
    raise ValueError("Missing required environment variable: MAX_DISTANCE")
MAX_DISTANCE = int(MAX_DISTANCE)

ADMIN_PASSCODE = os.getenv('ADMIN_PASSCODE')
if not ADMIN_PASSCODE:
    raise ValueError("Missing required environment variable: ADMIN_PASSCODE")

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("APP_URL")],
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],
)

# Database configuration
db_params = {
    'dbname': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'host': os.getenv('DB_HOST'),
    'port': os.getenv('DB_PORT')
}

# Validate required environment variables
required_env_vars = ['DB_NAME', 'DB_USER', 'DB_PASSWORD', 'DB_HOST', 'DB_PORT']
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Initialize database handler
db_handler = FaceEmbeddingDB(db_params)

# Initialize face processor
face_processor = FaceProcessor(db_handler)

# Initialize InsightFace model globally
face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great circle distance between two points on earth in meters."""
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371000  # Radius of earth in meters
    return c * r

def extract_face_embedding_from_bytes(image_bytes: bytes) -> Optional[np.ndarray]:
    """Extract face embedding from image bytes using InsightFace."""
    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return None
            
        faces = face_app.get(img)
        if not faces:
            return None
            
        return faces[0].embedding
        
    except Exception as e:
        print(f"Error extracting face embedding from bytes: {e}")
        return None

def extract_face_embedding_from_file(image_path: str) -> Optional[np.ndarray]:
    """Extract face embedding from an image file using InsightFace."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
            
        faces = face_app.get(img)
        if not faces:
            return None
            
        return faces[0].embedding
        
    except Exception as e:
        print(f"Error extracting face embedding: {e}")
        return None

def get_session_type() -> Optional[str]:
    """Determine session type based on current time."""
    current_time = datetime.now().time()
    morning_start = datetime.strptime("06:00", "%H:%M").time()
    morning_end = datetime.strptime("12:00", "%H:%M").time()
    afternoon_start = datetime.strptime("13:00", "%H:%M").time()
    afternoon_end = datetime.strptime("18:00", "%H:%M").time()
    
    if morning_start <= current_time <= morning_end:
        return "morning"
    elif afternoon_start <= current_time <= afternoon_end:
        return "afternoon"
    else:
        return None

def process_face_verification(image_bytes: bytes, latitude: float, longitude: float, action: str):
    """Common function to process face verification for check-in/check-out."""
    # Extract face embedding
    face_encoding = extract_face_embedding_from_bytes(image_bytes)
    if face_encoding is None:
        return {"status": "error", "message": "No face detected in the image"}
    
    results = db_handler.vector_search(face_encoding)
    if not results:
        return {"status": "error", "message": "Face not recognized"}
    
    best_match = results[0]
    entity_id = best_match["entity_id"]
    name = best_match["name"]
    
    employee_info = db_handler.get_employee_branch_location(entity_id)
    if not employee_info:
        return {"status": "error", "message": "Employee branch information not found"}
    
    branch_latitude = employee_info["latitude"]
    branch_longitude = employee_info["longitude"]
    
    distance = haversine(latitude, longitude, branch_latitude, branch_longitude)
    if distance > MAX_DISTANCE:
        return {
            "status": "error",
            "message": f"Location verification failed. You are {int(distance)}m away from your branch."
        }
    
    # Determine session type
    session_type = get_session_type()
    if session_type is None:
        return {
            "status": "error",
            "message": f"{action.capitalize()} is only allowed during morning (6:00-12:00) or afternoon (13:00-18:00) sessions."
        }
    
    return {
        "entity_id": entity_id,
        "name": name,
        "employee_info": employee_info,
        "distance": int(distance),
        "session_type": session_type
    }

@app.post("/branch/add")
async def add_branch(
    branch_name: str = Form(...),
    latitude: float = Form(...),
    longitude: float = Form(...)
):
    """Add a new branch with its geolocation."""
    try:
        branch_id = db_handler.add_branch(branch_name, latitude, longitude)
        if branch_id:
            return {
                "status": "success",
                "message": f"Branch '{branch_name}' added successfully",
                "branch_id": branch_id
            }
        else:
            return {"status": "error", "message": "Failed to add branch"}
    except Exception as e:
        return {"status": "error", "message": f"Error adding branch: {str(e)}"}

@app.get("/branches")
async def get_branches():
    """Get all branches with their locations."""
    try:
        branches = db_handler.get_branches()
        return {"status": "success", "branches": branches}
    except Exception as e:
        return {"status": "error", "message": f"Error retrieving branches: {str(e)}"}

@app.post("/enroll-employee")
async def enroll_employee(
    entity_id: str = Form(...),
    name: str = Form(...),
    branch_id: int = Form(...),
    photo: UploadFile = File(...)
):
    """Enroll an employee with photo, entity ID, and branch assignment."""
    try:
        # Input validation
        if not entity_id or entity_id.strip() == "":
            return {"status": "error", "message": "Entity ID cannot be empty"}

        if not name or name.strip() == "":
            return {"status": "error", "message": "Name cannot be empty"}

        # Clean inputs
        entity_id = entity_id.strip()
        name = name.strip()

        # Create employee images directory
        employee_images_dir = "employee_images"
        os.makedirs(employee_images_dir, exist_ok=True)
        employee_dir = os.path.join(employee_images_dir, entity_id)
        os.makedirs(employee_dir, exist_ok=True)

        # Save the uploaded photo
        photo_path = os.path.join(employee_dir, f"{entity_id}.jpg")
        with open(photo_path, "wb") as f:
            content = await photo.read()
            f.write(content)

        # Generate embedding using InsightFace
        encoding = extract_face_embedding_from_file(photo_path)
        if encoding is None:
            return {"status": "error", "message": "No face detected in the image"}

        # Store the embedding in the database with branch assignment
        success = db_handler.store_embedding(entity_id, name, encoding, branch_id)
        if success:
            return {
                "status": "success",
                "message": f"Employee {name} with ID {entity_id} enrolled successfully"
            }
        else:
            return {"status": "error", "message": "Failed to store employee data"}

    except Exception as e:
        return {"status": "error", "message": f"Error: {str(e)}"}

@app.post("/verify-face")
async def verify_face(
    photo: UploadFile = File(...),
    latitude: float = Form(...),
    longitude: float = Form(...)
):
    """Verify a face against stored embeddings and check location against employee's branch."""
    try:
        content = await photo.read()
        result = process_face_verification(content, latitude, longitude, "verification")
        
        if "status" in result:
            return result
        
        # All checks passed, return success
        return {
            "status": "success",
            "message": "Face and location verified successfully",
            "employee": {
                "entity_id": result["entity_id"],
                "name": result["name"],
                "branch": result["employee_info"]["branch_name"],
                "distance": result["distance"]
            }
        }

    except Exception as e:
        return {"status": "error", "message": f"Error: {str(e)}"}

@app.post("/process-checkin")
async def process_checkin(
    photo: UploadFile = File(...),
    latitude: float = Form(...),
    longitude: float = Form(...)
):
    """Process check-in photo and validate against employee's branch location."""
    try:
        content = await photo.read()
        result = process_face_verification(content, latitude, longitude, "check-in")
        
        if "status" in result:
            return result
        
        entity_id = result["entity_id"]
        name = result["name"]
        session_type = result["session_type"]
        
        # Check if the employee has already checked in today for this session
        if db_handler.has_checked_in_today(entity_id, session_type):
            return {
                "status": "error", 
                "message": f"{name} has already checked in for the {session_type} session today"
            }

        # Log the attendance
        success = db_handler.log_attendance(entity_id, 'checkin', latitude, longitude, session_type)
        if not success:
            return {"status": "error", "message": "Failed to log check-in"}
            
        # All checks passed, return success
        return {
            "status": "success",
            "message": f"{name} checked in successfully for the {session_type} session!",
            "employee": {
                "entity_id": entity_id,
                "name": name,
                "branch": result["employee_info"]["branch_name"],
                "distance": result["distance"],
                "session_type": session_type
            }
        }

    except Exception as e:
        return {"status": "error", "message": f"Error during check-in: {str(e)}"}

@app.post("/process-checkout")
async def process_checkout(
    photo: UploadFile = File(...),
    latitude: float = Form(...),
    longitude: float = Form(...)
):
    """Process check-out photo and validate against employee's branch location."""
    try:
        content = await photo.read()
        result = process_face_verification(content, latitude, longitude, "check-out")
        
        if "status" in result:
            return result
        
        entity_id = result["entity_id"]
        name = result["name"]
        session_type = result["session_type"]
        
        # Check if the employee has already checked out today for this session
        if db_handler.has_checked_out_today(entity_id, session_type):
            return {
                "status": "error", 
                "message": f"{name} has already checked out for the {session_type} session today"
            }

        # Log the attendance
        success = db_handler.log_attendance(entity_id, 'checkout', latitude, longitude, session_type)
        if not success:
            return {"status": "error", "message": "Failed to log check-out"}
            
        # All checks passed, return success
        return {
            "status": "success",
            "message": f"{name} checked out successfully for the {session_type} session!",
            "employee": {
                "entity_id": entity_id,
                "name": name,
                "branch": result["employee_info"]["branch_name"],
                "distance": result["distance"],
                "session_type": session_type
            }
        }

    except Exception as e:
        return {"status": "error", "message": f"Error during check-out: {str(e)}"}
    
@app.get("/employees")
async def get_employees():
    """
    Get a list of all employees with their branch assignments.
    
    Returns:
        List of employees with their details.
    """
    try:
        employees = db_handler.get_all_employees()
        if employees:
            return {
                "status": "success",
                "employees": employees,
                "count": len(employees)
            }
        else:
            return {
                "status": "success",
                "message": "No employees found",
                "employees": [],
                "count": 0
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error retrieving employees: {str(e)}"
        }
    
@app.get("/attendance/{entity_id}")
async def get_attendance(entity_id: str):
    """Get attendance records for a specific employee."""
    try:
        # Modification needed in db_handler to retrieve attendance by entity_id
        attendance_records = db_handler.retrieve_attendance(entity_id)
        return {"status": "success", "entity_id": entity_id, "attendance_records": attendance_records}
    except Exception as e:
        return {"status": "error", "message": f"Error retrieving attendance: {str(e)}"}

@app.get("/")
async def root():
    return {"message": "Face Recognition API is running"}

@app.get("/getall")
async def find():
    results = db_handler.retrieve_all_data()
    return results

@app.get("/delete")
async def delete():
    db_handler.delete_tables()
    return {"message": "Tables deleted"}

@app.delete("/delete-user/{name}")
async def delete_user(name: str):
    try:
        success = db_handler.delete_user(name)
        if success:
            return {"status": "success", "message": f"User '{name}' deleted successfully"}
        else:
            return {"status": "error", "message": f"User '{name}' not found"}
    except Exception as e:
        return {"status": "error", "message": f"Error deleting user: {str(e)}"}

@app.get("/attendance/{user_name}")
async def get_attendance(user_name: str):
    """Fetch attendance records for a specific user."""
    attendance_records = db_handler.retrieve_attendance(user_name)
    return attendance_records

@app.get("/user-report/{user_name}")
async def get_user_report(user_name: str):
    """
    Get detailed attendance report for a specific user.

    Args:
        user_name: The name of the user to get the report for

    Returns:
        List of daily attendance records with check-in and check-out times
    """
    try:
        report = db_handler.get_user_attendance_report(user_name)
        if report:
            return {
                "status": "success",
                "user_name": user_name,
                "attendance_records": report
            }
        else:
            return {
                "status": "error",
                "message": f"No attendance records found for user '{user_name}'"
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error retrieving attendance report: {str(e)}"
        }

@app.get("/notfound")
async def notfound():
    return {"message": "Table not found"}

@app.get("/user-report/{entity_id}")
async def get_user_report(entity_id: str):
    """
    Get detailed attendance report for a specific employee.

    Args:
        entity_id: The entity ID of the employee

    Returns:
        List of daily attendance records with check-in and check-out times
    """
    try:
        report = db_handler.get_user_attendance_report(entity_id)
        if report:
            return {
                "status": "success",
                "entity_id": entity_id,
                "attendance_records": report
            }
        else:
            return {
                "status": "error",
                "message": f"No attendance records found for employee with ID '{entity_id}'"
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error retrieving attendance report: {str(e)}"
        }
        
@app.post("/create-admin")
async def create_admin(request: Request):
    try:
        data = await request.json()
        email = data.get("email")
        passcode = data.get("passcode")

        if not email or not passcode:
            raise HTTPException(status_code=400, detail="Email and passcode are required")

        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            raise HTTPException(status_code=400, detail="Invalid email format")

        if passcode != os.getenv("ADMIN_PASSCODE"):
            raise HTTPException(status_code=401, detail="Invalid passcode")

        try:
            token = db_handler.create_admin_verification_token(email)
            if not token:
                raise HTTPException(status_code=500, detail="Failed to create verification token")

            # Send verification email
            if not face_processor.send_admin_verification_email(email, token):
                raise HTTPException(status_code=500, detail="Failed to send verification email")

            return {"message": "Verification email sent successfully"}
        except OperationalError as e:
            print(f"Database operational error: {e}")
            # Try to reconnect
            db_handler.connect()
            raise HTTPException(status_code=503, detail="Database connection error. Please try again.")
        except Exception as e:
            print(f"Error in create-admin: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error in create-admin: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/verify-admin-token")
async def verify_admin(
    token: str = Form(...),
    username: str = Form(...),
    password: str = Form(...)
):
    """
    Verify admin token and create admin account.
    
    Args:
        token: Verification token from email
        username: Chosen username for admin
        password: Chosen password for admin
        
    Returns:
        Status of admin account creation
    """
    try:
        # Input validation
        if not token or not username or not password:
            return {"status": "error", "message": "All fields are required"}
        
        
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        success = db_handler.verify_admin_token(token, username, password_hash)
        if success:
            return {
                "status": "success",
                "message": "Admin account created successfully. You can now log in."
            }
        else:
            return {
                "status": "error",
                "message": "Invalid or expired token. Please request a new admin invitation."
            }
    except Exception as e:
        return {"status": "error", "message": f"Error verifying admin token: {str(e)}"}

@app.post("/admin-login")
async def admin_login(username: str = Form(...), password: str = Form(...)):
    """
    Verify admin login credentials.
    
    Args:
        username: Admin username
        password: Admin password
        
    Returns:
        Login status
    """
    try:
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        is_valid = db_handler.verify_admin_credentials(username, password_hash)
        if is_valid:
            return {
                "status": "success",
                "message": "Login successful"
            }
        else:
            return {
                "status": "error",
                "message": "Invalid username or password"
            }
    except Exception as e:
        return {"status": "error", "message": f"Error during login: {str(e)}"}

@app.post("/send-admin-credentials")
async def send_admin_credentials(email: str = Form(...)):
    """
    Send admin access credentials to admin@support.com.
    
    Args:
        email: Email address to send credentials to
        
    Returns:
        Status of email sending operation
    """
    try:
        # Validate email format
        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            return {"status": "error", "message": "Invalid email format"}
        
        EMAIL_HOST = os.getenv("EMAIL_HOST")
        EMAIL_PORT = os.getenv("EMAIL_PORT")
        EMAIL_USERNAME = os.getenv("EMAIL_USERNAME")
        EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
        EMAIL_SUPPORT = os.getenv("EMAIL_SUPPORT")
        
        # Create a message with admin credentials
        msg = MIMEMultipart()
        msg['From'] = EMAIL_USERNAME
        msg['To'] = EMAIL_SUPPORT
        msg['Subject'] = "Admin Access Credentials Request"
        
        body = f"""
        <html>
        <body>
            <h2>Admin Access Credentials Request</h2>
            <p>A request for admin access credentials has been made from the following email:</p>
            <p><strong>{email}</strong></p>
            <p>Please review this request and take appropriate action.</p>
            <p>This is an automated message from the Face Recognition Attendance System.</p>
        </body>
        </html>
        """
        
        msg.attach(MIMEText(body, 'html'))
        
        # Send the email
        server = smtplib.SMTP(EMAIL_HOST, EMAIL_PORT)
        server.starttls()
        server.login(EMAIL_USERNAME, EMAIL_PASSWORD)
        
        server.send_message(msg)
        server.quit()
        
        return {
            "status": "success",
            "message": "Sent successfully..! Wait for approval"
        }
    except Exception as e:
        return {"status": "error", "message": f"Error sending admin credentials request: {str(e)}"}

@app.get("/attendance/today/summary")
async def today_attendance_summary():
    """
    Get summary of today's attendance and weekly average.
    
    Returns:
        Count of employees checked in today and weekly average percentage
    """
    try:
        # Get today's date
        today = datetime.now().date()
        
        # Query for today's attendance
        with db_handler.conn.cursor() as cur:
            # Get total employees count
            cur.execute("SELECT COUNT(*) FROM face_embeddings")
            total_employees = cur.fetchone()[0]
            
            if total_employees == 0:
                return {
                    "status": "success", 
                    "count": 0, 
                    "percentage": 0,
                    "weeklyAverage": 0
                }
            
            # Count unique employees who checked in today
            cur.execute("""
                SELECT COUNT(DISTINCT entity_id) 
                FROM attendance 
                WHERE event_type = 'checkin' 
                  AND session_type = 'morning'
                  AND DATE(event_time) = %s
            """, (today,))
            morning_count = cur.fetchone()[0]
            
            # Count unique employees who checked in today for afternoon session
            cur.execute("""
                SELECT COUNT(DISTINCT entity_id) 
                FROM attendance 
                WHERE event_type = 'checkin' 
                  AND session_type = 'afternoon'
                  AND DATE(event_time) = %s
            """, (today,))
            afternoon_count = cur.fetchone()[0]
            
            # Calculate percentages
            morning_percentage = (morning_count / total_employees) * 100 if total_employees > 0 else 0
            afternoon_percentage = (afternoon_count / total_employees) * 100 if total_employees > 0 else 0
            
            # Get weekly average (last 7 days)
            one_week_ago = today - timedelta(days=7)
            cur.execute("""
                WITH daily_counts AS (
                    SELECT 
                        DATE(event_time) as attendance_date,
                        session_type,
                        COUNT(DISTINCT entity_id) as daily_count
                    FROM attendance
                    WHERE 
                        event_type = 'checkin' AND 
                        DATE(event_time) BETWEEN %s AND %s
                    GROUP BY DATE(event_time), session_type
                )
                SELECT 
                    session_type,
                    AVG(daily_count) as weekly_avg
                FROM daily_counts
                GROUP BY session_type
            """, (one_week_ago, today))
            
            results = cur.fetchall()
            weekly_avg_counts = {}
            for session_type, avg_count in results:
                weekly_avg_counts[session_type] = avg_count if avg_count is not None else 0
            
            weekly_avg_percentages = {
                session_type: (avg_count / total_employees) * 100 if total_employees > 0 else 0
                for session_type, avg_count in weekly_avg_counts.items()
            }
            
        return {
            "status": "success",
            "morning": {
                "count": morning_count,
                "percentage": round(morning_percentage, 1)
            },
            "afternoon": {
                "count": afternoon_count,
                "percentage": round(afternoon_percentage, 1)
            },
            "weeklyAverage": {
                "morning": round(weekly_avg_percentages.get("morning", 0), 1),
                "afternoon": round(weekly_avg_percentages.get("afternoon", 0), 1)
            }
        }
    except Exception as e:
        return {"status": "error", "message": f"Error retrieving attendance summary: {str(e)}"}

@app.get("/attendance/recent-activity")
async def get_recent_activities():
    """
    Get recent attendance and system activities.
    
    Returns:
        List of recent activities with timestamps and details
    """
    try:
        with db_handler.conn.cursor() as cur:
            # Get recent attendance events
            cur.execute("""
                SELECT 
                    a.entity_id,
                    e.name,
                    a.event_type,
                    a.session_type,
                    a.event_time,
                    b.branch_name
                FROM 
                    attendance a
                JOIN 
                    face_embeddings e ON a.entity_id = e.entity_id
                JOIN 
                    branches b ON e.branch_id = b.branch_id
                ORDER BY 
                    a.event_time DESC
                LIMIT 20
            """)
            
            attendance_events = cur.fetchall()
            
            # Format the activities
            activities = []
            for event in attendance_events:
                entity_id, name, event_type, session_type, event_time, branch_name = event
                
                if event_type == 'checkin':
                    activity_type = 'check-in'
                    color = 'green'
                    description = f"{name} checked in for {session_type} session at {branch_name}"
                else:
                    activity_type = 'check-out'
                    color = 'red'
                    description = f"{name} checked out from {session_type} session at {branch_name}"
                
                activities.append({
                    "id": f"{entity_id}-{event_time.isoformat()}",
                    "type": activity_type,
                    "color": color,
                    "description": description,
                    "timestamp": event_time.isoformat(),
                    "timeFormatted": event_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "session_type": session_type
                })
            
            # Also get new employee enrollments (based on creation time in face_embeddings)
            cur.execute("""
                SELECT 
                    entity_id,
                    name,
                    created_at
                FROM 
                    face_embeddings
                ORDER BY 
                    created_at DESC
                LIMIT 5
            """)
            
            enrollments = cur.fetchall()
            
            for enrollment in enrollments:
                entity_id, name, created_at = enrollment
                
                activities.append({
                    "id": f"enroll-{entity_id}",
                    "type": "enrollment",
                    "color": "yellow",
                    "description": f"New employee enrolled: {name}",
                    "timestamp": created_at.isoformat(),
                    "timeFormatted": created_at.strftime("%Y-%m-%d %H:%M:%S")
                })
            
            # Get new branch additions
            cur.execute("""
                SELECT 
                    branch_id,
                    branch_name,
                    created_at
                FROM 
                    branches
                ORDER BY 
                    created_at DESC
                LIMIT 3
            """)
            
            new_branches = cur.fetchall()
            
            for branch in new_branches:
                branch_id, branch_name, created_at = branch
                
                activities.append({
                    "id": f"branch-{branch_id}",
                    "type": "branch",
                    "color": "blue",
                    "description": f"New branch added: {branch_name}",
                    "timestamp": created_at.isoformat(),
                    "timeFormatted": created_at.strftime("%Y-%m-%d %H:%M:%S")
                })
            
            # Sort all activities by timestamp
            activities.sort(key=lambda x: x["timestamp"], reverse=True)
            
            return {
                "status": "success",
                "activities": activities[:10]  # Return only the 10 most recent
            }
            
    except Exception as e:
        return {"status": "error", "message": f"Error retrieving recent activities: {str(e)}"}

@app.get("/attendance/report")
async def get_attendance_report(startDate: str = None, endDate: str = None):
    """
    Generate comprehensive attendance report for a date range.
    
    Args:
        startDate: Start date for the report (YYYY-MM-DD)
        endDate: End date for the report (YYYY-MM-DD)
        
    Returns:
        Attendance statistics and daily data for the specified period
    """
    try:
        # Set default dates if not provided
        if not startDate:
            start_date = datetime.now() - timedelta(days=30)
        else:
            start_date = datetime.strptime(startDate, "%Y-%m-%d")
            
        if not endDate:
            end_date = datetime.now()
        else:
            end_date = datetime.strptime(endDate, "%Y-%m-%d")
        
        with db_handler.conn.cursor() as cur:
            # Get total employees count
            cur.execute("SELECT COUNT(*) FROM face_embeddings")
            total_employees = cur.fetchone()[0]
            
            # Get daily attendance counts
            cur.execute("""
                WITH date_range AS (
                    SELECT generate_series(
                        %s::date, 
                        %s::date, 
                        '1 day'::interval
                    )::date AS day
                ),
                daily_counts AS (
                    SELECT 
                        DATE(event_time) as attendance_date,
                        session_type,
                        COUNT(DISTINCT entity_id) as check_in_count
                    FROM attendance
                    WHERE 
                        event_type = 'checkin' AND 
                        DATE(event_time) BETWEEN %s AND %s
                    GROUP BY DATE(event_time), session_type
                )
                SELECT 
                    dr.day,
                    'morning' as session_type,
                    COALESCE(dc.check_in_count, 0) as check_in_count
                FROM 
                    date_range dr
                LEFT JOIN 
                    daily_counts dc ON dr.day = dc.attendance_date AND dc.session_type = 'morning'
                
                UNION ALL
                
                SELECT 
                    dr.day,
                    'afternoon' as session_type,
                    COALESCE(dc.check_in_count, 0) as check_in_count
                FROM 
                    date_range dr
                LEFT JOIN 
                    daily_counts dc ON dr.day = dc.attendance_date AND dc.session_type = 'afternoon'
                
                ORDER BY 
                    dr.day, session_type
            """, (start_date, end_date, start_date, end_date))
            
            daily_data = cur.fetchall()
            
            # Format daily data
            formatted_daily_data = []
            total_attendance = 0
            
            current_date = None
            current_day_data = {}
            
            for day_data in daily_data:
                day, session_type, count = day_data
                
                if current_date != day:
                    if current_date is not None:
                        formatted_daily_data.append(current_day_data)
                    current_date = day
                    current_day_data = {
                        "date": day.strftime("%Y-%m-%d"),
                        "morning": {
                            "count": 0,
                            "percentage": 0
                        },
                        "afternoon": {
                            "count": 0,
                            "percentage": 0
                        }
                    }
                
                percentage = (count / total_employees) * 100 if total_employees > 0 else 0
                total_attendance += count
                
                current_day_data[session_type]["count"] = count
                current_day_data[session_type]["percentage"] = round(percentage, 1)
            
            # Add the last day
            if current_date is not None:
                formatted_daily_data.append(current_day_data)
            
            # Calculate summary statistics
            total_days = (end_date - start_date).days + 1
            avg_attendance = total_attendance / (total_days * 2) if total_days > 0 else 0  # Multiply by 2 for morning and afternoon
            avg_percentage = (avg_attendance / total_employees) * 100 if total_employees > 0 else 0
            
            # Get top branches by attendance
            cur.execute("""
                WITH branch_attendance AS (
                    SELECT 
                        b.branch_id,
                        b.branch_name,
                        COUNT(DISTINCT a.entity_id) as attendance_count
                    FROM 
                        attendance a
                    JOIN 
                        face_embeddings e ON a.entity_id = e.entity_id
                    JOIN 
                        branches b ON e.branch_id = b.branch_id
                    WHERE 
                        a.event_type = 'checkin' AND 
                        DATE(a.event_time) BETWEEN %s AND %s
                    GROUP BY 
                        b.branch_id, b.branch_name
                )
                SELECT 
                    branch_name,
                    attendance_count
                FROM 
                    branch_attendance
                ORDER BY 
                    attendance_count DESC
                LIMIT 5
            """, (start_date, end_date))
            
            top_branches = [{"name": name, "count": count} for name, count in cur.fetchall()]
            
            return {
                "status": "success",
                "summary": {
                    "totalEmployees": total_employees,
                    "avgAttendance": round(avg_attendance, 1),
                    "avgPercentage": round(avg_percentage, 1),
                    "period": {
                        "start": start_date.strftime("%Y-%m-%d"),
                        "end": end_date.strftime("%Y-%m-%d"),
                        "days": total_days
                    },
                    "topBranches": top_branches
                },
                "dailyData": formatted_daily_data
            }
            
    except Exception as e:
        return {"status": "error", "message": f"Error generating attendance report: {str(e)}"}

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources when the application shuts down."""
    try:
        if db_handler:
            db_handler.close()
    except Exception as e:
        print(f"Error during shutdown: {e}")

@app.get("/attendance/daily-report")
async def get_daily_attendance_report(
    date: str,
    branch: str = None,
    search: str = None
):
    """
    Get daily attendance report with optional filtering by branch and search term.
    
    Args:
        date: Date in YYYY-MM-DD format
        branch: Optional branch name to filter by
        search: Optional search term to filter by name or employee ID
        
    Returns:
        List of attendance records for the specified date
    """
    try:
        with db_handler.conn.cursor() as cur:
            # Base query with joins to get all necessary information
            query = """
                SELECT 
                    a.entity_id,
                    e.name,
                    b.branch_name,
                    a.event_time,
                    a.event_type,
                    a.session_type
                FROM 
                    attendance a
                JOIN 
                    face_embeddings e ON a.entity_id = e.entity_id
                JOIN 
                    branches b ON e.branch_id = b.branch_id
                WHERE 
                    DATE(a.event_time) = %s
            """
            params = [date]
            
            # Add branch filter if specified
            if branch and branch != "All Branches":
                query += " AND b.branch_name = %s"
                params.append(branch)
            
            # Add search filter if specified
            if search:
                query += " AND (e.name ILIKE %s OR a.entity_id ILIKE %s)"
                search_term = f"%{search}%"
                params.extend([search_term, search_term])
            
            query += " ORDER BY a.event_time"
            
            cur.execute(query, params)
            records = cur.fetchall()
            
            # Process records to group check-ins and check-outs
            attendance_data = {}
            for record in records:
                entity_id, name, branch_name, event_time, event_type, session_type = record
                
                if entity_id not in attendance_data:
                    attendance_data[entity_id] = {
                        "entity_id": entity_id,
                        "name": name,
                        "branch_name": branch_name,
                        "checkIn": None,
                        "checkOut": None,
                        "session_type": session_type
                    }
                
                if event_type == "checkin":
                    attendance_data[entity_id]["checkIn"] = event_time.isoformat()
                elif event_type == "checkout":
                    attendance_data[entity_id]["checkOut"] = event_time.isoformat()
            
            return {
                "status": "success",
                "records": list(attendance_data.values())
            }
            
    except Exception as e:
        return {"status": "error", "message": f"Error retrieving daily attendance report: {str(e)}"}

@app.get("/attendance/weekly-report")
async def get_weekly_attendance_report(
    start_date: str,
    end_date: str,
    branch: str = None,
    search: str = None
):
    """
    Get weekly attendance report with summary statistics and optional filtering.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        branch: Optional branch name to filter by
        search: Optional search term to filter by name or employee ID
        
    Returns:
        Weekly attendance summary and detailed records
    """
    try:
        with db_handler.conn.cursor() as cur:
            # Get total number of employees
            cur.execute("SELECT COUNT(*) FROM face_embeddings")
            total_employees = cur.fetchone()[0]
            
            # Get total number of working days in the period
            cur.execute("""
                SELECT COUNT(DISTINCT DATE(event_time))
                FROM attendance
                WHERE DATE(event_time) BETWEEN %s AND %s
            """, (start_date, end_date))
            total_days = cur.fetchone()[0] or 1  # Avoid division by zero
            
            # Base query for attendance records
            query = """
                WITH employee_attendance AS (
                    SELECT 
                        a.entity_id,
                        e.name,
                        b.branch_name,
                        COUNT(DISTINCT DATE(a.event_time)) as present_days,
                        COUNT(DISTINCT CASE WHEN a.event_type = 'checkin' THEN DATE(a.event_time) END) as checkin_days
                    FROM 
                        attendance a
                    JOIN 
                        face_embeddings e ON a.entity_id = e.entity_id
                    JOIN 
                        branches b ON e.branch_id = b.branch_id
                    WHERE 
                        DATE(a.event_time) BETWEEN %s AND %s
            """
            params = [start_date, end_date]
            
            # Add branch filter if specified
            if branch and branch != "All Branches":
                query += " AND b.branch_name = %s"
                params.append(branch)
            
            # Add search filter if specified
            if search:
                query += " AND (e.name ILIKE %s OR a.entity_id ILIKE %s)"
                search_term = f"%{search}%"
                params.extend([search_term, search_term])
            
            query += """
                    GROUP BY a.entity_id, e.name, b.branch_name
                )
                SELECT 
                    entity_id,
                    name,
                    branch_name,
                    present_days,
                    ROUND((present_days::float / %s) * 100, 1) as attendance_rate
                FROM 
                    employee_attendance
                ORDER BY 
                    attendance_rate DESC, name
            """
            params.append(total_days)
            
            cur.execute(query, params)
            records = cur.fetchall()
            
            # Calculate summary statistics
            total_present = sum(record[3] for record in records)  # present_days
            avg_attendance_rate = sum(record[4] for record in records) / len(records) if records else 0
            
            # Get weekly average (last 7 days)
            cur.execute("""
                SELECT 
                    COUNT(DISTINCT entity_id)::float / NULLIF((SELECT COUNT(*) FROM face_embeddings), 0) * 100
                FROM attendance
                WHERE DATE(event_time) >= CURRENT_DATE - INTERVAL '7 days'
                AND event_type = 'checkin'
            """)
            weekly_average = cur.fetchone()[0] or 0
            
            return {
                "status": "success",
                "summary": {
                    "count": total_present,
                    "percentage": round(avg_attendance_rate, 1),
                    "weeklyAverage": round(weekly_average, 1)
                },
                "records": [
                    {
                        "entity_id": record[0],
                        "name": record[1],
                        "branch_name": record[2],
                        "presentDays": record[3],
                        "attendanceRate": record[4]
                    }
                    for record in records
                ]
            }
            
    except Exception as e:
        return {"status": "error", "message": f"Error retrieving weekly attendance report: {str(e)}"}

@app.get("/attendance/export")
async def export_attendance(
    start_date: str,
    end_date: str,
    report_type: str = "daily",
    branch: str = None
):
    """
    Export attendance data to CSV format.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        report_type: Type of report ('daily' or 'weekly')
        branch: Optional branch name to filter by
        
    Returns:
        CSV file containing attendance data
    """
    try:
        with db_handler.conn.cursor() as cur:
            if report_type == "daily":
                # Query for daily report
                query = """
                    SELECT 
                        a.entity_id,
                        e.name,
                        b.branch_name,
                        DATE(a.event_time) as date,
                        a.session_type,
                        MIN(CASE WHEN a.event_type = 'checkin' THEN a.event_time END) as checkin_time,
                        MAX(CASE WHEN a.event_type = 'checkout' THEN a.event_time END) as checkout_time
                    FROM 
                        attendance a
                    JOIN 
                        face_embeddings e ON a.entity_id = e.entity_id
                    JOIN 
                        branches b ON e.branch_id = b.branch_id
                    WHERE 
                        DATE(a.event_time) BETWEEN %s AND %s
                """
            else:
                # Query for weekly report
                query = """
                    WITH employee_attendance AS (
                        SELECT 
                            a.entity_id,
                            e.name,
                            b.branch_name,
                            COUNT(DISTINCT DATE(a.event_time)) as present_days,
                            COUNT(DISTINCT CASE WHEN a.event_type = 'checkin' THEN DATE(a.event_time) END) as checkin_days
                        FROM 
                            attendance a
                        JOIN 
                            face_embeddings e ON a.entity_id = e.entity_id
                        JOIN 
                            branches b ON e.branch_id = b.branch_id
                        WHERE 
                            DATE(a.event_time) BETWEEN %s AND %s
                """
            
            params = [start_date, end_date]
            
            # Add branch filter if specified
            if branch and branch != "All Branches":
                query += " AND b.branch_name = %s"
                params.append(branch)
            
            if report_type == "daily":
                query += """
                    GROUP BY a.entity_id, e.name, b.branch_name, DATE(a.event_time), a.session_type
                    ORDER BY date, name
                """
            else:
                query += """
                        GROUP BY a.entity_id, e.name, b.branch_name
                    )
                    SELECT 
                        entity_id,
                        name,
                        branch_name,
                        present_days,
                        ROUND((present_days::float / (SELECT COUNT(DISTINCT DATE(event_time)) FROM attendance WHERE DATE(event_time) BETWEEN %s AND %s)) * 100, 1) as attendance_rate
                    FROM 
                        employee_attendance
                    ORDER BY 
                        attendance_rate DESC, name
                """
                params.extend([start_date, end_date])
            
            cur.execute(query, params)
            records = cur.fetchall()
            
            # Create CSV file in memory
            output = io.StringIO()
            writer = csv.writer(output)
            
            if report_type == "daily":
                # Write daily report headers
                writer.writerow([
                    "Employee ID",
                    "Name",
                    "Branch",
                    "Date",
                    "Session",
                    "Check-in Time",
                    "Check-out Time"
                ])
                
                # Write daily report data
                for record in records:
                    writer.writerow([
                        record[0],  # entity_id
                        record[1],  # name
                        record[2],  # branch_name
                        record[3],  # date
                        record[4],  # session_type
                        record[5].strftime("%H:%M:%S") if record[5] else "N/A",  # checkin_time
                        record[6].strftime("%H:%M:%S") if record[6] else "N/A"   # checkout_time
                    ])
            else:
                # Write weekly report headers
                writer.writerow([
                    "Employee ID",
                    "Name",
                    "Branch",
                    "Present Days",
                    "Attendance Rate (%)"
                ])
                
                # Write weekly report data
                for record in records:
                    writer.writerow([
                        record[0],  # entity_id
                        record[1],  # name
                        record[2],  # branch_name
                        record[3],  # present_days
                        record[4]   # attendance_rate
                    ])
            
            # Prepare the response
            output.seek(0)
            return StreamingResponse(
                iter([output.getvalue()]),
                media_type="text/csv",
                headers={
                    "Content-Disposition": f"attachment; filename=attendance_report_{start_date}_to_{end_date}.csv"
                }
            )
            
    except Exception as e:
        return {"status": "error", "message": f"Error exporting attendance data: {str(e)}"}
    
@app.post("/forgot-password")
async def update_password(
    username: str = Form(...),
    new_password: str = Form(...)
):
    """
    Update password for a given username.
    """
    try:
        password_hash = hashlib.sha256(new_password.encode()).hexdigest()
        success = db_handler.forgot_password(username, password_hash)
        if success:
            return {"status": "success", "message": "Password updated successfully"}
        else:
            return {"status": "error", "message": "Username not found or update failed"}
    except Exception as e:
        return {"status": "error", "message": f"Error updating password: {str(e)}"}