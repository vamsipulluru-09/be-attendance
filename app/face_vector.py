from typing import List, Dict, Optional
import psycopg2
import numpy as np
from psycopg2.extras import execute_values
from datetime import datetime, timedelta, timezone
import psycopg2.extensions

import secrets
from dotenv import load_dotenv
import os

load_dotenv()




class FaceEmbeddingDB:
    def __init__(self, db_params: Dict[str, str]):
        self.db_params = db_params
        self.conn = None
        self.connect()
        self.create_tables()

    def connect(self):
        """Establish a database connection and close any existing connection."""
        try:
            # Close existing connection if it exists
            if self.conn is not None:
                try:
                    self.conn.close()
                except Exception:
                    pass
                self.conn = None
                
            # Create a new connection
            self.conn = psycopg2.connect(**self.db_params)
            print("Successfully connected to the database")
        except Exception as e:
            print(f"Error connecting to database: {e}")
            raise

    def ensure_connection(self):
        """Ensure the database connection is valid, reconnect if necessary."""
        try:
            # Check if connection is valid
            if self.conn is None or self.conn.closed:
                print("Connection is closed or None, reconnecting...")
                self.connect()
                return
                
            # Try a simple query to test the connection
            with self.conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
            print(f"Connection error detected: {e}, reconnecting...")
            self.connect()
        except Exception as e:
            print(f"Unexpected error checking connection: {e}")
            self.connect()

    def create_tables(self):
        """Create necessary tables if they don't exist."""
        try:
            self.ensure_connection()
            create_tables_query = """
            CREATE EXTENSION IF NOT EXISTS vector;
            
            CREATE TABLE IF NOT EXISTS branches (
                branch_id SERIAL PRIMARY KEY,
                branch_name VARCHAR(255) NOT NULL UNIQUE,
                latitude FLOAT NOT NULL,
                longitude FLOAT NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS face_embeddings (
                id SERIAL PRIMARY KEY,
                entity_id VARCHAR(255) NOT NULL UNIQUE,
                name VARCHAR(255) NOT NULL,
                embedding vector(128) NOT NULL,
                branch_id INTEGER REFERENCES branches(branch_id),
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS attendance (
                id SERIAL PRIMARY KEY,
                entity_id VARCHAR(255) NOT NULL,
                event_type VARCHAR(10) NOT NULL,
                session_type VARCHAR(10) NOT NULL,
                event_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                latitude FLOAT NOT NULL,
                longitude FLOAT NOT NULL,
                FOREIGN KEY (entity_id) REFERENCES face_embeddings (entity_id) ON DELETE CASCADE
            );
            
            CREATE TABLE IF NOT EXISTS admins (
            id SERIAL PRIMARY KEY,
            username VARCHAR(255) NOT NULL UNIQUE,
            email VARCHAR(255) NOT NULL UNIQUE,
            password_hash VARCHAR(255) NOT NULL,
            is_active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS admin_verification_tokens (
                id SERIAL PRIMARY KEY,
                email VARCHAR(255) NOT NULL UNIQUE,
                token VARCHAR(255) NOT NULL UNIQUE,
                expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
            """
            with self.conn.cursor() as cur:
                cur.execute(create_tables_query)
                self.conn.commit()
            print("Tables created or already exist")
        except Exception as e:
            print(f"Error creating tables: {e}")
            self.conn.rollback()
            raise

    def store_embedding(self, entity_id: str, name: str, embedding: np.ndarray, branch_id: int) -> bool:
        """Store facial embedding for an employee with entity_id and branch assignment."""
        try:
            # First validate that the branch exists
            with self.conn.cursor() as cur:
                cur.execute("SELECT 1 FROM branches WHERE branch_id = %s", (branch_id,))
                if not cur.fetchone():
                    print(f"Branch with ID {branch_id} does not exist")
                    return False
                
                # Insert the embedding
                cur.execute("""
                    INSERT INTO face_embeddings (entity_id, name, embedding, branch_id)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (entity_id) DO UPDATE
                    SET name = EXCLUDED.name, embedding = EXCLUDED.embedding, branch_id = EXCLUDED.branch_id
                """, (entity_id, name, embedding.tolist(), branch_id))
                self.conn.commit()
                return True
        except Exception as e:
            print(f"Error storing embedding: {e}")
            self.conn.rollback()
            return False

    def get_employee_branch_location(self, entity_id: str) -> Dict[str, any]:
        """Get an employee's branch location."""
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT e.entity_id, e.name, b.branch_id, b.branch_name, b.latitude, b.longitude
                    FROM face_embeddings e
                    JOIN branches b ON e.branch_id = b.branch_id
                    WHERE e.entity_id = %s
                """, (entity_id,))
                result = cur.fetchone()
                if result:
                    return {
                        "entity_id": result[0],
                        "name": result[1],
                        "branch_id": result[2],
                        "branch_name": result[3],
                        "latitude": result[4],
                        "longitude": result[5]
                    }
                return None
        except Exception as e:
            print(f"Error retrieving employee branch location: {e}")
            return None

    def vector_search(self, encoding: np.ndarray) -> List[Dict[str, any]]:
        """Search for similar face embeddings."""
        try:
            with self.conn.cursor() as cur:
                threshold = 0.9
                
                query = """
                    SELECT id, entity_id, name, embedding, branch_id, created_at,
                    1 - (embedding <=> %s::vector) as similarity
                    FROM face_embeddings
                    WHERE 1 - (embedding <=> %s::vector) > %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT 5;
                """
                cur.execute(query, (encoding.tolist(), encoding.tolist(), threshold, encoding.tolist()))
                results = cur.fetchall()
                
                return [
                    {
                        "id": result[0],
                        "entity_id": result[1],
                        "name": result[2],
                        "embedding": result[3],
                        "branch_id": result[4],
                        "created_at": result[5],
                        "similarity": result[6]
                    }
                    for result in results
                ]
        except Exception as e:
            print(f"Error performing vector search: {e}")
            return []

    def add_branch(self, branch_name: str, latitude: float, longitude: float) -> int:
        """Add a new branch with geolocation data."""
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO branches (branch_name, latitude, longitude)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (branch_name) DO UPDATE
                    SET latitude = EXCLUDED.latitude, longitude = EXCLUDED.longitude
                    RETURNING branch_id
                """, (branch_name, latitude, longitude))
                branch_id = cur.fetchone()[0]
                self.conn.commit()
                return branch_id
        except Exception as e:
            print(f"Error adding branch: {e}")
            self.conn.rollback()
            return None

    def get_branches(self) -> List[Dict[str, any]]:
        """Get all branches."""
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT branch_id, branch_name, latitude, longitude
                    FROM branches
                    ORDER BY branch_name
                """)
                branches = cur.fetchall()
                return [
                    {
                        "branch_id": row[0],
                        "branch_name": row[1],
                        "latitude": row[2],
                        "longitude": row[3]
                    }
                    for row in branches
                ]
        except Exception as e:
            print(f"Error retrieving branches: {e}")
            return []

    def has_checked_in_today(self, entity_id: str, session_type: str) -> bool:
        """Check if an employee has already checked in today for a specific session."""
        today = datetime.now().date()
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT 1
                    FROM attendance
                    WHERE entity_id = %s
                      AND event_type = 'checkin'
                      AND session_type = %s
                      AND DATE(event_time) = %s
                """, (entity_id, session_type, today))
                return cur.fetchone() is not None
        except Exception as e:
            print(f"Error checking if user has checked in today: {e}")
            return False

    def has_checked_out_today(self, entity_id: str, session_type: str) -> bool:
        """Check if an employee has already checked out today for a specific session."""
        today = datetime.now().date()
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT 1
                    FROM attendance
                    WHERE entity_id = %s
                      AND event_type = 'checkout'
                      AND session_type = %s
                      AND DATE(event_time) = %s
                """, (entity_id, session_type, today))
                return cur.fetchone() is not None
        except Exception as e:
            print(f"Error checking if user has checked out today: {e}")
            return False

    def log_attendance(self, entity_id: str, event_type: str, latitude: float, longitude: float, session_type: str) -> bool:
        """Log attendance event for an employee."""
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO attendance (entity_id, event_type, session_type, latitude, longitude)
                    VALUES (%s, %s, %s, %s, %s)
                """, (entity_id, event_type, session_type, latitude, longitude))
                self.conn.commit()
                return True
        except Exception as e:
            print(f"Error logging attendance: {e}")
            self.conn.rollback()
            return False

    def retrieve_attendance(self, entity_id: str) -> List[Dict[str, any]]:
        """Get attendance records for a specific employee."""
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT a.event_type, a.event_time, a.latitude, a.longitude, e.name
                    FROM attendance a
                    JOIN face_embeddings e ON a.entity_id = e.entity_id
                    WHERE a.entity_id = %s
                    ORDER BY event_time DESC
                """, (entity_id,))
                attendance_records = cur.fetchall()
                return [
                    {
                        "event_type": row[0], 
                        "event_time": row[1],
                        "latitude": row[2],
                        "longitude": row[3],
                        "name": row[4]
                    }
                    for row in attendance_records
                ]
        except Exception as e:
            print(f"Error retrieving attendance: {e}")
            return []

    def get_user_attendance_report(self, entity_id: str) -> List[Dict[str, any]]:
        """Generate attendance report for a specific employee."""
        try:
            with self.conn.cursor() as cur:
                query = """
                    WITH daily_attendance AS (
                        SELECT
                            DATE(event_time) as attendance_date,
                            session_type,
                            MAX(CASE WHEN event_type = 'checkin' THEN event_time END) as checkin_time,
                            MAX(CASE WHEN event_type = 'checkout' THEN event_time END) as checkout_time
                        FROM attendance
                        WHERE entity_id = %s
                        GROUP BY DATE(event_time), session_type
                    )
                    SELECT
                        attendance_date,
                        session_type,
                        checkin_time,
                        checkout_time
                    FROM daily_attendance
                    ORDER BY attendance_date DESC, session_type;
                """
                cur.execute(query, (entity_id,))
                results = cur.fetchall()

                return [
                    {
                        "date": row[0],
                        "session_type": row[1],
                        "checkin_time": row[2],
                        "checkout_time": row[3]
                    }
                    for row in results
                ]
        except Exception as e:
            print(f"Error retrieving user attendance report: {e}")
            return []

    def retrieve_all_data(self) -> List[Dict[str, any]]:
        """Get all employee data."""
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT e.entity_id, e.name, b.branch_name, b.latitude, b.longitude
                    FROM face_embeddings e
                    JOIN branches b ON e.branch_id = b.branch_id
                """)
                results = cur.fetchall()
                return [
                    {
                        "entity_id": row[0],
                        "name": row[1],
                        "branch": row[2],
                        "latitude": row[3],
                        "longitude": row[4]
                    }
                    for row in results
                ]
        except Exception as e:
            print(f"Error retrieving all data: {e}")
            return []

    def delete_tables(self):
        """Drop all tables for development/testing purposes."""
        delete_tables_query = """
            DROP TABLE IF EXISTS attendance CASCADE;
            DROP TABLE IF EXISTS face_embeddings CASCADE;
            DROP TABLE IF EXISTS branches CASCADE;
        """
        try:
            with self.conn.cursor() as cur:
                cur.execute(delete_tables_query)
                self.conn.commit()
            print("All tables deleted")
            return True
        except Exception as e:
            print(f"Error deleting tables: {e}")
            self.conn.rollback()
            return False

    def delete_user(self, entity_id: str) -> bool:
        """Delete a specific user by entity_id."""
        try:
            with self.conn.cursor() as cur:
                cur.execute("DELETE FROM face_embeddings WHERE entity_id = %s", (entity_id,))
                rows_deleted = cur.rowcount
                self.conn.commit()
                return rows_deleted > 0
        except Exception as e:
            print(f"Error deleting user: {e}")
            self.conn.rollback()
            return False

    def close(self):
        """Close the database connection."""
        if self.conn:
            try:
                self.conn.close()
                print("Database connection closed")
            except Exception as e:
                print(f"Error closing database connection: {e}")
            finally:
                self.conn = None
            
    def get_all_employees(self) -> List[Dict[str, any]]:
            """
            Get all employees with their branch information.
            
            Returns:
                List of dictionaries containing employee details.
            """
            try:
                with self.conn.cursor() as cur:
                    cur.execute("""
                        SELECT 
                            e.entity_id, 
                            e.name, 
                            e.created_at,
                            b.branch_id, 
                            b.branch_name, 
                            b.latitude, 
                            b.longitude
                        FROM 
                            face_embeddings e
                        LEFT JOIN 
                            branches b ON e.branch_id = b.branch_id
                        ORDER BY 
                            e.name
                    """)
                    employees = cur.fetchall()
                    return [
                        {
                            "entity_id": row[0],
                            "name": row[1],
                            "branch_name": row[4],
                            }
                            for row in employees
                        ]
            except Exception as e:
                    print(f"Error retrieving employees: {e}")
                    return []
                
    def create_admin_verification_token(self, email: str) -> str:
        """Create a verification token for admin registration."""
        try:
            self.ensure_connection()
            # Generate a secure token
            token = secrets.token_urlsafe(32)
            
            # Set expiration time (24 hours from now)
            expires_at = datetime.now() + timedelta(hours=24)
            
            with self.conn.cursor() as cur:
                cur.execute("DELETE FROM admin_verification_tokens WHERE email = %s", (email,))
                
                # Insert new token
                cur.execute("""
                    INSERT INTO admin_verification_tokens (email, token, expires_at)
                    VALUES (%s, %s, %s)
                    RETURNING token
                """, (email, token, expires_at))
                
                result = cur.fetchone()
                self.conn.commit()
                return result[0]
        except Exception as e:
            print(f"Error creating admin verification token: {e}")
            if self.conn:
                self.conn.rollback()
            return None

    def verify_admin_token(self, token: str, username: str, password_hash: str) -> bool:
        """Verify admin token and create admin account."""
        try:
            self.ensure_connection()
            with self.conn.cursor() as cur:
                # Get token information
                cur.execute("""
                    SELECT email, expires_at FROM admin_verification_tokens
                    WHERE token = %s
                """, (token,))
                
                result = cur.fetchone()
                if not result:
                    return False
                    
                email, expires_at = result
                
                # Check if token is expired
                if datetime.now(timezone.utc) > expires_at:

                    # Delete expired token
                    cur.execute("DELETE FROM admin_verification_tokens WHERE token = %s", (token,))
                    self.conn.commit()
                    return False
                    
                # Create admin account
                cur.execute("""
                    INSERT INTO admins (username, email, password_hash)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (email) DO UPDATE
                    SET username = EXCLUDED.username, password_hash = EXCLUDED.password_hash
                """, (username, email, password_hash))
                
                # Delete used token
                cur.execute("DELETE FROM admin_verification_tokens WHERE token = %s", (token,))
                
                self.conn.commit()
                return True
        except Exception as e:
            print(f"Error verifying admin token: {e}")
            if self.conn:
                self.conn.rollback()
            return False
        

    def verify_admin_credentials(self, username: str, password_hash: str) -> bool:
        """Verify admin login credentials."""
        try:
            self.ensure_connection()
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT id FROM admins
                    WHERE username = %s AND password_hash = %s AND is_active = TRUE
                """, (username, password_hash))
                
                return cur.fetchone() is not None
        except Exception as e:
            print(f"Error verifying admin credentials: {e}")
            return False

    def forgot_password(self, username, new_password_hash):
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    "UPDATE admins SET password_hash = %s WHERE username = %s",
                    (new_password_hash, username)
                )
                self.conn.commit()
                return cur.rowcount > 0
        except Exception:
            return False

    