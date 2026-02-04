import os
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from .face_vector import FaceEmbeddingDB
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
from insightface.app import FaceAnalysis

load_dotenv()

APP_URL = os.getenv('APP_URL')
EMAIL_USERNAME = os.getenv('EMAIL_USERNAME')
EMAIL_HOST = os.getenv('EMAIL_HOST')
EMAIL_PORT = os.getenv('EMAIL_PORT')
EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD')


class FaceProcessor:
    def __init__(self, db_handler: FaceEmbeddingDB):
        self.db_handler = db_handler
        # Initialize InsightFace model
        self.face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))

    def calculate_cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        try:
            # Normalize embeddings
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Calculate cosine similarity
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            return float(similarity)
        except Exception as e:
            print(f"Error calculating cosine similarity: {e}")
            return 0.0

    def calculate_euclidean_distance(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate Euclidean distance between two embeddings."""
        try:
            distance = np.linalg.norm(embedding1 - embedding2)
            return float(distance)
        except Exception as e:
            print(f"Error calculating Euclidean distance: {e}")
            return float('inf')
            
    def process_image(self, image_numpy: np.ndarray) -> List[Dict]:
        """Process a single image and return detected faces with matching logic."""
        try:
            # Ensure image is in BGR format (OpenCV default)
            if len(image_numpy.shape) == 3 and image_numpy.shape[2] == 3:
                bgr_image = image_numpy
            else:
                bgr_image = cv2.cvtColor(image_numpy, cv2.COLOR_RGB2BGR)
            
            # Get faces using InsightFace
            faces = self.face_app.get(bgr_image)
            
            detected_faces = []
            
            for face in faces:
                face_embedding = face.embedding
                face_bbox = face.bbox.astype(int)
                
                # Get top matches from vector search
                results = self.db_handler.vector_search(face_embedding)
                
                name = "Unknown"
                confidence = 0.0
                
                if results:
                    try:
                        best_similarity = 0.0
                        best_name = "Unknown"
                        
                        for result in results:
                            if isinstance(result["embedding"], str):
                                embedding_str = result["embedding"].strip('[]')
                                embedding_values = [float(x.strip()) for x in embedding_str.split(',')]
                                candidate_embedding = np.array(embedding_values)
                            else:
                                candidate_embedding = np.array(result["embedding"])
                            
                            similarity = self.calculate_cosine_similarity(face_embedding, candidate_embedding)
                            
                            if similarity > best_similarity:
                                best_similarity = similarity
                                best_name = result["name"]
                        
                        recognition_threshold = 0.5  # Cosine similarity threshold
                        
                        if best_similarity > recognition_threshold:
                            name = best_name
                            confidence = best_similarity
                        
                    except Exception as e:
                        print(f"Error processing embeddings: {e}")
                        continue
                
                # Only include if confidence is high enough
                if name != "Unknown" or confidence > 0.3:
                    x1, y1, x2, y2 = face_bbox
                    detected_faces.append({
                        "name": name,
                        "confidence": float(confidence),
                        "location": {
                            "top": int(y1),
                            "right": int(x2),
                            "bottom": int(y2),
                            "left": int(x1)
                        }
                    })
            
            return detected_faces
            
        except Exception as e:
            print(f"Error processing image: {e}")
            return []

    def extract_face_embedding_from_image(self, image_numpy: np.ndarray) -> Optional[np.ndarray]:
        """Extract face embedding from image numpy array."""
        try:
            # Ensure image is in BGR format
            if len(image_numpy.shape) == 3 and image_numpy.shape[2] == 3:
                bgr_image = image_numpy
            else:
                bgr_image = cv2.cvtColor(image_numpy, cv2.COLOR_RGB2BGR)
            
            faces = self.face_app.get(bgr_image)
            
            if faces:
                return faces[0].embedding
            return None
            
        except Exception as e:
            print(f"Error extracting face embedding: {e}")
            return None
        