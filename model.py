import os
import cv2
import numpy as np
import pickle
import time
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings('ignore')
MODEL_PATH = "model.pkl"
class FastFaceRecognizer:
    def __init__(self):
        self.model = None
        self.face_cascade = None
        self._init_face_detector()
        self.load_model()

    def _init_face_detector(self):
        print("Initializing face detector...")
        self.use_haar = False
        self.use_mediapipe = False


        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if os.path.exists(cascade_path):
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
                self.use_haar = True
                print("✓ Haar Cascade initialized")
            else:
                # Try current directory
                cascade_path = 'haarcascade_frontalface_default.xml'
                if os.path.exists(cascade_path):
                    self.face_cascade = cv2.CascadeClassifier(cascade_path)
                    self.use_haar = True
                    print("✓ Haar Cascade initialized from current directory")
        except Exception as e:
            print(f"✗ Haar Cascade initialization failed: {e}")

        # Try MediaPipe as optional
        try:
            import mediapipe as mp
            self.face_detection_mp = mp.solutions.face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=0.3
            )
            self.use_mediapipe = True
            print("✓ MediaPipe initialized")
        except ImportError:
            print("✗ MediaPipe not available")
            self.face_detection_mp = None
        except Exception as e:
            print(f"✗ MediaPipe initialization failed: {e}")
            self.face_detection_mp = None

        if not self.use_haar and not self.use_mediapipe:
            print("⚠ No face detector available!")

    def detect_face(self, image):
        """Detect face using available method"""
        # Try MediaPipe first if available
        if self.use_mediapipe and self.face_detection_mp:
            try:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = self.face_detection_mp.process(rgb_image)
                if results.detections:
                    detection = results.detections[0]
                    bbox = detection.location_data.relative_bounding_box
                    h, w = image.shape[:2]
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)
                    return x, y, width, height
            except Exception as e:
                print(f"MediaPipe detection failed: {e}")

        # Fallback to Haar Cascade
        if self.use_haar and self.face_cascade:
            try:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )
                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    return x, y, w, h
            except Exception as e:
                print(f"Haar detection failed: {e}")

        return None

    def extract_face_embedding(self, image_stream):
        """Extract face embedding from image stream"""
        try:
            # Read image
            data = image_stream.read()
            if not data:
                return None

            nparr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                print("Failed to decode image")
                return None

            # Resize for faster processing
            max_size = 320
            height, width = img.shape[:2]
            if max(height, width) > max_size:
                scale = max_size / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                img = cv2.resize(img, (new_width, new_height))

            # Detect face
            face_rect = self.detect_face(img)
            if face_rect is None:
                print("No face detected in image")
                return None

            x, y, w, h = face_rect

            # Extract face region
            face = img[y:y + h, x:x + w]
            if face.size == 0:
                print("Empty face region")
                return None

            # Convert to grayscale and resize
            face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face_resized = cv2.resize(face_gray, (64, 64))

            # Apply histogram equalization for better contrast
            face_eq = cv2.equalizeHist(face_resized)

            # Normalize
            face_normalized = face_eq.astype(np.float32) / 255.0

            # Flatten and add simple features
            embedding = face_normalized.flatten()

            # Add basic statistics
            mean_val = np.mean(face_eq)
            std_val = np.std(face_eq)
            embedding = np.append(embedding, [mean_val, std_val])

            return embedding

        except Exception as e:
            print(f"Error in embedding extraction: {e}")
            return None

    def load_model(self):
        """Load trained model"""
        try:
            if os.path.exists(MODEL_PATH):
                print(f"Loading model from {MODEL_PATH}...")
                with open(MODEL_PATH, 'rb') as f:
                    self.model = pickle.load(f)

                # Verify model is valid
                if self.model is not None and hasattr(self.model, 'classes_'):
                    print(f"✓ Model loaded successfully with {len(self.model.classes_)} classes")
                    print(f"  Classes: {self.model.classes_}")
                    return True
                else:
                    print("✗ Model file is invalid")
                    self.model = None
                    return False
            else:
                print(f"Model file not found: {MODEL_PATH}")
                self.model = None
                return False
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
            return False

    def save_model(self):
        if self.model is None:
            print("No model to save")
            return False

        try:
            print(f"Saving model to {MODEL_PATH}...")
            with open(MODEL_PATH, 'wb') as f:
                pickle.dump(self.model, f)
            print(f"✓ Model saved successfully ({os.path.getsize(MODEL_PATH)} bytes)")
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False

    def train(self, dataset_dir, progress_callback=None):
        """Train model on dataset"""
        print(f"\n{'=' * 60}")
        print("STARTING MODEL TRAINING")
        print(f"{'=' * 60}")
        print(f"Dataset directory: {dataset_dir}")

        # Initialize as lists
        X_list = []
        y_list = []

        # Get list of student directories
        try:
            student_dirs = [d for d in os.listdir(dataset_dir)
                            if os.path.isdir(os.path.join(dataset_dir, d))]
            student_dirs.sort(key=lambda x: int(x) if x.isdigit() else x)
        except FileNotFoundError:
            print(f"✗ Dataset directory not found: {dataset_dir}")
            if progress_callback:
                progress_callback(0, "Dataset directory not found")
            return False

        if not student_dirs:
            print("✗ No student directories found")
            if progress_callback:
                progress_callback(0, "No training data found")
            return False

        print(f"Found {len(student_dirs)} student directories")
        total_students = len(student_dirs)

        for idx, sid in enumerate(student_dirs):
            folder = os.path.join(dataset_dir, sid)

            if not os.path.exists(folder):
                print(f"✗ Directory not found: {folder}")
                continue

            image_files = [f for f in os.listdir(folder)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            if not image_files:
                print(f"⚠ Student {sid}: No images found")
                continue

            # Limit number of images per student
            max_images_per_student = 20
            image_files = image_files[:max_images_per_student]

            student_embeddings = []
            images_processed = 0

            for img_file in image_files:
                img_path = os.path.join(folder, img_file)
                img = cv2.imread(img_path)

                if img is not None:
                    # Resize for faster processing
                    max_size = 320
                    h, w = img.shape[:2]
                    if max(h, w) > max_size:
                        scale = max_size / max(h, w)
                        new_w = int(w * scale)
                        new_h = int(h * scale)
                        img = cv2.resize(img, (new_w, new_h))

                    # Detect face
                    face_rect = self.detect_face(img)

                    if face_rect:
                        x, y, w, h = face_rect
                        face = img[y:y + h, x:x + w]

                        if face.size > 0:
                            # Convert to grayscale and resize
                            face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                            face_resized = cv2.resize(face_gray, (64, 64))
                            face_eq = cv2.equalizeHist(face_resized)

                            # Normalize and flatten
                            face_normalized = face_eq.astype(np.float32) / 255.0
                            embedding = face_normalized.flatten()

                            # Add basic statistics
                            mean_val = np.mean(face_eq)
                            std_val = np.std(face_eq)
                            embedding = np.append(embedding, [mean_val, std_val])

                            student_embeddings.append(embedding)
                            images_processed += 1

            if student_embeddings:
                # Use average of embeddings for each student
                if len(student_embeddings) > 1:
                    avg_embedding = np.mean(student_embeddings, axis=0)
                else:
                    avg_embedding = student_embeddings[0]

                X_list.append(avg_embedding)
                y_list.append(int(sid))
                print(f"✓ Student {sid}: Processed {images_processed}/{len(image_files)} images")
            else:
                print(f"✗ Student {sid}: No valid faces found in any images")

            # Update progress
            if progress_callback:
                progress = int((idx + 1) / total_students * 80)
                progress_callback(progress, f"Processed {idx + 1}/{total_students} students")

        if len(X_list) < 2:
            print(f"\n✗ Need at least 2 students with valid faces (found {len(X_list)})")
            if progress_callback:
                progress_callback(0, f"Need at least 2 students with valid faces (found {len(X_list)})")
            return False

        print(f"\n✓ Training with {len(X_list)} samples, {len(set(y_list))} unique students")
        if len(X_list) > 0 and len(X_list[0]) > 0:
            print(f"  Feature dimension: {len(X_list[0])}")

        # Convert to numpy arrays
        X_train = np.array(X_list)
        y_train = np.array(y_list)

        # Train classifier
        print("\nTraining classifier...")
        if progress_callback:
            progress_callback(90, "Training classifier...")

        try:
            # Use KNN with dynamic neighbors
            n_neighbors = min(3, len(X_train) - 1)
            print(f"  Using KNN with n_neighbors={n_neighbors}")

            self.model = KNeighborsClassifier(
                n_neighbors=n_neighbors,
                weights='distance',
                metric='euclidean'
            )
            self.model.fit(X_train, y_train)

            print(f"✓ Classifier trained successfully")
            print(f"  Classes: {self.model.classes_}")

            # Save model
            if self.save_model():
                if progress_callback:
                    progress_callback(100, f"Training completed! Model saved with {len(set(y_list))} students")
                print(f"\n{'=' * 60}")
                print("TRAINING COMPLETED SUCCESSFULLY!")
                print(f"{'=' * 60}")
                print(f"✓ Model trained with {len(set(y_list))} students")
                print(f"✓ Model saved to {MODEL_PATH}")
                print(f"✓ Ready for face recognition")
                return True
            else:
                print("\n✗ Failed to save model")
                if progress_callback:
                    progress_callback(0, "Training failed: Could not save model")
                return False

        except Exception as e:
            print(f"\n✗ Training error: {e}")
            import traceback
            traceback.print_exc()
            if progress_callback:
                progress_callback(0, f"Training failed: {str(e)}")
            return False

    def predict(self, embedding):

        if self.model is None:
            print("Model not loaded for prediction")
            return None, 0.0

        if embedding is None:
            print("No embedding provided for prediction")
            return None, 0.0

        try:
            # Ensure embedding is 2D
            if len(embedding.shape) == 1:
                embedding = embedding.reshape(1, -1)

            # Get prediction
            pred = self.model.predict(embedding)[0]

            # Get distances to all neighbors
            distances, indices = self.model.kneighbors(embedding, n_neighbors=len(self.model.classes_))

            # Calculate confidence based on distance ratio
            if distances[0][0] == 0:
                confidence = 1.0
            else:
                if len(distances[0]) > 1:
                    # Confidence based on ratio to second best match
                    confidence = min(1.0, distances[0][1] / (distances[0][0] + 1e-10))
                else:
                    confidence = 1.0 / (1.0 + distances[0][0])

            # Apply threshold
            if confidence < 0.3:  # Lower threshold
                print(f"Low confidence prediction: {confidence}")
                return None, confidence

            return int(pred), float(confidence)

        except Exception as e:
            print(f"Prediction error: {e}")
            return None, 0.0


# Global instance
recognizer = FastFaceRecognizer()


# Wrapper functions for Flask app
def extract_embedding_for_image(stream_or_bytes):
    return recognizer.extract_face_embedding(stream_or_bytes)


def load_model_if_exists():
    return recognizer.load_model()


def predict_with_model(emb):
    return recognizer.predict(emb)


def train_model_background(dataset_dir, progress_callback=None):
    return recognizer.train(dataset_dir, progress_callback)


def is_model_trained():

    return recognizer.model is not None and os.path.exists(MODEL_PATH)


# Main function for testing
def main():
    print(f"\n{'=' * 60}")
    print("FACE RECOGNITION MODEL TEST")
    print(f"{'=' * 60}")
    print("\n1. Testing face detector...")
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    test_image[25:75, 25:75] = 255

    face_rect = recognizer.detect_face(test_image)
    if face_rect:
        print(f"✓ Face detector working")
    else:
        print(f"⚠ Face detector not working on test image")

    # Check model status
    print("\n2. Checking model status...")
    if is_model_trained():
        print(f"✓ Model is trained and loaded")
    else:
        print(f"✗ Model not trained")

    # Check dataset
    print("\n3. Checking dataset...")
    dataset_dir = "dataset"
    if os.path.exists(dataset_dir):
        student_dirs = [d for d in os.listdir(dataset_dir)
                        if os.path.isdir(os.path.join(dataset_dir, d))]
        print(f"✓ Dataset exists with {len(student_dirs)} students")

        for sid in student_dirs[:3]:  # Show first 3
            folder = os.path.join(dataset_dir, sid)
            images = [f for f in os.listdir(folder)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"  Student {sid}: {len(images)} images")
    else:
        print(f"✗ Dataset directory not found: {dataset_dir}")

    print(f"\n{'=' * 60}")
    print("TEST COMPLETE")
    print(f"{'=' * 60}")
    print("\nTo train the model, run the Flask app and use the web interface.")
    print("Or run: python -c \"from model import recognizer; recognizer.train('dataset')\"")

    return True

if __name__ == "__main__":
    main()