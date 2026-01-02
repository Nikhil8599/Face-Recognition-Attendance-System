import os
import io
import threading
import sqlite3
import datetime
import json
import time
from flask import Flask, render_template, request, jsonify, send_file
from model import train_model_background, extract_embedding_for_image, load_model_if_exists, predict_with_model, \
    is_model_trained

app = Flask(__name__,
            static_folder="static",
            template_folder="templates",
            static_url_path="/static")

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "attendance.db")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
TRAIN_STATUS_FILE = os.path.join(BASE_DIR, "train_status.json")

# Create necessary directories
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "static", "images"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "static", "css"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "static", "js"), exist_ok=True)


# ---------- Database Setup ----------
def init_db():
    """Initialize database tables"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        # Students table
        c.execute('''CREATE TABLE IF NOT EXISTS students (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    roll TEXT,
                    class TEXT,
                    section TEXT,
                    reg_no TEXT,
                    created_at TEXT
                )''')

        # Attendance table
        c.execute('''CREATE TABLE IF NOT EXISTS attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id INTEGER,
                    name TEXT,
                    timestamp TEXT
                )''')

        conn.commit()
        conn.close()
        print(f"Database initialized at: {DB_PATH}")
    except Exception as e:
        print(f"Database initialization error: {e}")


# Initialize database on startup
init_db()


# ---------- Training Status Functions ----------
def write_train_status(status_dict):
    """Write training status to file"""
    try:
        with open(TRAIN_STATUS_FILE, "w") as f:
            json.dump(status_dict, f)
    except Exception as e:
        print(f"Error writing train status: {e}")


def read_train_status():
    """Read training status from file"""
    default_status = {
        "running": False,
        "progress": 0,
        "message": "Not trained yet",
        "trained": False
    }

    if not os.path.exists(TRAIN_STATUS_FILE):
        write_train_status(default_status)
        return default_status

    try:
        with open(TRAIN_STATUS_FILE, "r") as f:
            status = json.load(f)
            # Ensure all fields exist
            for key in default_status:
                if key not in status:
                    status[key] = default_status[key]
            return status
    except Exception as e:
        print(f"Error reading train status: {e}")
        return default_status


# ---------- Routes ----------
@app.route("/")
def index():
    """Home page - Dashboard"""
    return render_template("index.html")


@app.route("/stats")
def get_stats():
    """Get system statistics"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        # Total students
        c.execute("SELECT COUNT(*) FROM students")
        total_students = c.fetchone()[0] or 0

        # Today's attendance
        today = datetime.date.today().isoformat()
        c.execute("SELECT COUNT(DISTINCT student_id) FROM attendance WHERE date(timestamp) = ?", (today,))
        today_attendance = c.fetchone()[0] or 0

        # Model status
        model_trained = is_model_trained()

        conn.close()

        return jsonify({
            "success": True,
            "total_students": total_students,
            "today_attendance": today_attendance,
            "avg_attendance": "0%",
            "model_trained": model_trained
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/recent_activity")
def get_recent_activity():
    """Get recent attendance activity"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        c.execute('''
            SELECT a.name, a.timestamp, s.class 
            FROM attendance a
            LEFT JOIN students s ON a.student_id = s.id
            ORDER BY a.timestamp DESC 
            LIMIT 10
        ''')

        rows = c.fetchall()
        conn.close()

        activity = []
        for row in rows:
            activity.append({
                "name": row[0],
                "timestamp": row[1],
                "class": row[2] if row[2] else "N/A"
            })

        return jsonify(activity)
    except Exception as e:
        print(f"Error getting recent activity: {e}")
        return jsonify([])


@app.route("/attendance_stats")
def attendance_stats():
    """Get attendance statistics for chart"""
    try:
        import pandas as pd
        conn = sqlite3.connect(DB_PATH)

        # Get all attendance records
        df = pd.read_sql_query("SELECT timestamp FROM attendance", conn)
        conn.close()

        if df.empty:
            # Return empty data for chart
            dates = []
            counts = []
            for i in range(29, -1, -1):
                date = datetime.date.today() - datetime.timedelta(days=i)
                dates.append(date.strftime("%d-%b"))
                counts.append(0)
        else:
            # Process real data
            df['date'] = pd.to_datetime(df['timestamp']).dt.date
            last_30_days = []
            for i in range(29, -1, -1):
                day = datetime.date.today() - datetime.timedelta(days=i)
                last_30_days.append(day)

            counts = []
            dates = []
            for day in last_30_days:
                count = len(df[df['date'] == day])
                counts.append(int(count))
                dates.append(day.strftime("%d-%b"))

        return jsonify({
            "dates": dates,
            "counts": counts
        })
    except Exception as e:
        print(f"Error getting attendance stats: {e}")
        # Return default data on error
        dates = []
        counts = []
        for i in range(29, -1, -1):
            date = datetime.date.today() - datetime.timedelta(days=i)
            dates.append(date.strftime("%d-%b"))
            counts.append(0)
        return jsonify({"dates": dates, "counts": counts})


@app.route("/add_student", methods=["GET", "POST"])
def add_student():
    """Add student page"""
    if request.method == "GET":
        return render_template("add_student.html")

    # POST request - save student info
    try:
        data = request.form
        name = data.get("name", "").strip()
        roll = data.get("roll", "").strip()
        class_name = data.get("class", "").strip()
        section = data.get("sec", "").strip()
        reg_no = data.get("reg_no", "").strip()

        if not name:
            return jsonify({"success": False, "error": "Name is required"}), 400

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        # Insert student
        now = datetime.datetime.now().isoformat()
        c.execute('''INSERT INTO students (name, roll, class, section, reg_no, created_at) 
                     VALUES (?, ?, ?, ?, ?, ?)''',
                  (name, roll, class_name, section, reg_no, now))

        student_id = c.lastrowid
        conn.commit()
        conn.close()

        # Create student folder for images
        student_folder = os.path.join(DATASET_DIR, str(student_id))
        os.makedirs(student_folder, exist_ok=True)

        return jsonify({
            "success": True,
            "student_id": student_id,
            "message": f"Student '{name}' added successfully"
        })

    except Exception as e:
        print(f"Error adding student: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/upload_face", methods=["POST"])
def upload_face():
    """Upload face images for a student"""
    try:
        student_id = request.form.get("student_id")
        if not student_id:
            return jsonify({"success": False, "error": "Student ID is required"}), 400

        files = request.files.getlist("images[]")
        if not files:
            return jsonify({"success": False, "error": "No images provided"}), 400

        student_folder = os.path.join(DATASET_DIR, student_id)
        os.makedirs(student_folder, exist_ok=True)

        saved_count = 0
        for i, file in enumerate(files[:20]):  # Limit to 20 images
            if file.filename:
                # Generate unique filename
                timestamp = int(time.time() * 1000)
                filename = f"{timestamp}_{i}.jpg"
                filepath = os.path.join(student_folder, filename)
                file.save(filepath)
                saved_count += 1

        return jsonify({
            "success": True,
            "saved": saved_count,
            "message": f"Saved {saved_count} images for student {student_id}"
        })

    except Exception as e:
        print(f"Error uploading face images: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/train_model", methods=["GET"])
def train_model_route():
    """Start model training"""
    try:
        # Check if training is already running
        status = read_train_status()
        if status["running"]:
            return jsonify({
                "success": False,
                "error": "Training is already in progress"
            }), 400

        # Check if we have students to train on
        if not os.path.exists(DATASET_DIR):
            return jsonify({
                "success": False,
                "error": "No students found. Add students first."
            }), 400

        student_folders = [d for d in os.listdir(DATASET_DIR)
                           if os.path.isdir(os.path.join(DATASET_DIR, d))]

        if len(student_folders) < 2:
            return jsonify({
                "success": False,
                "error": f"Need at least 2 students for training. Found {len(student_folders)}."
            }), 400

        # Count students with images
        students_with_images = 0
        for folder in student_folders:
            folder_path = os.path.join(DATASET_DIR, folder)
            images = [f for f in os.listdir(folder_path)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if images:
                students_with_images += 1

        if students_with_images < 2:
            return jsonify({
                "success": False,
                "error": f"Need at least 2 students with face images. Found {students_with_images}."
            }), 400

        # Clear previous status and start training
        write_train_status({
            "running": True,
            "progress": 0,
            "message": "Starting training...",
            "trained": False
        })

        def training_thread():
            """Background training thread"""
            try:
                # Train model
                success = train_model_background(
                    DATASET_DIR,
                    lambda p, m: write_train_status({
                        "running": True,
                        "progress": p,
                        "message": m,
                        "trained": p == 100
                    })
                )

                if success:
                    write_train_status({
                        "running": False,
                        "progress": 100,
                        "message": "Training completed successfully!",
                        "trained": True
                    })
                    print("Model training completed successfully")
                    print(f"Model saved to: {MODEL_PATH}")
                else:
                    write_train_status({
                        "running": False,
                        "progress": 0,
                        "message": "Training failed. Please check your data and try again.",
                        "trained": False
                    })

            except Exception as e:
                print(f"Training error in thread: {e}")
                import traceback
                traceback.print_exc()
                write_train_status({
                    "running": False,
                    "progress": 0,
                    "message": f"Training error: {str(e)}",
                    "trained": False
                })

        # Start background thread
        thread = threading.Thread(target=training_thread, daemon=True)
        thread.start()

        return jsonify({
            "success": True,
            "message": f"Training started with {students_with_images} students",
            "students": students_with_images
        })

    except Exception as e:
        print(f"Error starting training: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/train_status", methods=["GET"])
def train_status():
    """Get training status"""
    return jsonify(read_train_status())


@app.route("/check_model", methods=["GET"])
def check_model():
    """Check model status"""
    try:
        model_exists = os.path.exists(MODEL_PATH)
        model_loaded = is_model_trained()

        # Count training samples
        train_samples = 0
        if os.path.exists(DATASET_DIR):
            student_folders = [d for d in os.listdir(DATASET_DIR)
                               if os.path.isdir(os.path.join(DATASET_DIR, d))]
            train_samples = len(student_folders)

        return jsonify({
            "success": True,
            "model_exists": model_exists,
            "model_loaded": model_loaded,
            "train_samples": train_samples,
            "status": read_train_status()
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/mark_attendance", methods=["GET"])
def mark_attendance_page():
    """Mark attendance page"""
    return render_template("mark_attendance.html")


@app.route("/recognize_face", methods=["POST"])
def recognize_face():
    """Recognize face from image"""
    try:
        if "image" not in request.files:
            return jsonify({
                "success": False,
                "recognized": False,
                "error": "No image provided"
            }), 400

        # Check if model exists
        if not os.path.exists(MODEL_PATH):
            return jsonify({
                "success": False,
                "recognized": False,
                "error": "Model not trained. Please train the model first."
            }), 200

        # Load model if not already loaded
        if not load_model_if_exists():
            return jsonify({
                "success": False,
                "recognized": False,
                "error": "Failed to load model. Please retrain."
            }), 200

        image_file = request.files["image"]

        # Extract embedding
        embedding = extract_embedding_for_image(image_file.stream)
        if embedding is None:
            return jsonify({
                "success": False,
                "recognized": False,
                "error": "No face detected in the image"
            }), 200

        # Predict
        student_id, confidence = predict_with_model(embedding)

        if student_id is None or confidence < 0.3:
            return jsonify({
                "success": False,
                "recognized": False,
                "error": "Face not recognized",
                "confidence": float(confidence) if confidence else 0
            }), 200

        # Get student info from database
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT name, class FROM students WHERE id = ?", (int(student_id),))
        student_info = c.fetchone()

        if not student_info:
            conn.close()
            return jsonify({
                "success": False,
                "recognized": False,
                "error": "Student not found in database"
            }), 200

        name, student_class = student_info
        student_class = student_class if student_class else "N/A"

        # Check if already marked today
        today = datetime.date.today().isoformat()
        c.execute('''SELECT id FROM attendance 
                     WHERE student_id = ? AND date(timestamp) = ? 
                     LIMIT 1''', (int(student_id), today))

        existing = c.fetchone()

        if not existing:
            # Save attendance record
            timestamp = datetime.datetime.now().isoformat()
            c.execute('''INSERT INTO attendance (student_id, name, timestamp) 
                         VALUES (?, ?, ?)''', (int(student_id), name, timestamp))
            conn.commit()
            already_marked = False
        else:
            already_marked = True

        conn.close()

        return jsonify({
            "success": True,
            "recognized": True,
            "student_id": int(student_id),
            "name": name,
            "class": student_class,
            "confidence": float(confidence),
            "already_marked": already_marked,
            "timestamp": datetime.datetime.now().isoformat()
        })

    except Exception as e:
        print(f"Error recognizing face: {e}")
        return jsonify({
            "success": False,
            "recognized": False,
            "error": f"Recognition error: {str(e)}"
        }), 500


@app.route("/attendance_record", methods=["GET"])
def attendance_record():
    """Attendance records page"""
    try:
        period = request.args.get("period", "all")

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        query = "SELECT id, student_id, name, timestamp FROM attendance"
        params = []

        if period == "daily":
            today = datetime.date.today().isoformat()
            query += " WHERE date(timestamp) = ?"
            params.append(today)
        elif period == "weekly":
            week_ago = (datetime.date.today() - datetime.timedelta(days=7)).isoformat()
            query += " WHERE date(timestamp) >= ?"
            params.append(week_ago)
        elif period == "monthly":
            month_ago = (datetime.date.today() - datetime.timedelta(days=30)).isoformat()
            query += " WHERE date(timestamp) >= ?"
            params.append(month_ago)

        query += " ORDER BY timestamp DESC LIMIT 1000"
        c.execute(query, params)
        records = c.fetchall()
        conn.close()

        return render_template("attendance_record.html", records=records, period=period)
    except Exception as e:
        print(f"Error getting attendance records: {e}")
        return render_template("attendance_record.html", records=[], period="all")


@app.route("/download_csv", methods=["GET"])
def download_csv():
    """Download attendance as CSV"""
    try:
        period = request.args.get("period", "all")

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        query = "SELECT id, student_id, name, timestamp FROM attendance"
        params = []

        if period == "daily":
            today = datetime.date.today().isoformat()
            query += " WHERE date(timestamp) = ?"
            params.append(today)
        elif period == "weekly":
            week_ago = (datetime.date.today() - datetime.timedelta(days=7)).isoformat()
            query += " WHERE date(timestamp) >= ?"
            params.append(week_ago)
        elif period == "monthly":
            month_ago = (datetime.date.today() - datetime.timedelta(days=30)).isoformat()
            query += " WHERE date(timestamp) >= ?"
            params.append(month_ago)

        query += " ORDER BY timestamp DESC"
        c.execute(query, params)
        records = c.fetchall()
        conn.close()

        # Create CSV
        output = io.StringIO()
        output.write("ID,Student ID,Name,Timestamp\n")

        for record in records:
            output.write(f"{record[0]},{record[1]},{record[2]},{record[3]}\n")

        # Create response
        response = io.BytesIO()
        response.write(output.getvalue().encode('utf-8'))
        response.seek(0)

        filename = f"attendance_{period}_{datetime.date.today()}.csv"

        return send_file(
            response,
            mimetype="text/csv",
            as_attachment=True,
            download_name=filename
        )

    except Exception as e:
        print(f"Error generating CSV: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/debug", methods=["GET"])
def debug_info():
    """Debug endpoint for system info"""
    info = {
        "app_running": True,
        "database": DB_PATH,
        "database_exists": os.path.exists(DB_PATH),
        "model_file": MODEL_PATH,
        "model_exists": os.path.exists(MODEL_PATH),
        "dataset_dir": DATASET_DIR,
        "dataset_exists": os.path.exists(DATASET_DIR),
        "students_count": 0,
        "train_status": read_train_status(),
        "current_time": datetime.datetime.now().isoformat()
    }

    # Count students
    if os.path.exists(DATASET_DIR):
        student_folders = [d for d in os.listdir(DATASET_DIR)
                           if os.path.isdir(os.path.join(DATASET_DIR, d))]
        info["students_count"] = len(student_folders)
        info["student_folders"] = student_folders[:10]  # First 10

        # Count images per student
        images_info = []
        for folder in student_folders[:5]:  # First 5 students
            folder_path = os.path.join(DATASET_DIR, folder)
            images = [f for f in os.listdir(folder_path)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            images_info.append({
                "student_id": folder,
                "image_count": len(images),
                "images": images[:3] if images else []
            })
        info["images_info"] = images_info

    return jsonify(info)


# ---------- Main Application ----------
if __name__ == "__main__":
    print("=" * 60)
    print("Digital Attendance System")
    print("=" * 60)
    print(f"Database: {DB_PATH}")
    print(f"Dataset: {DATASET_DIR}")
    print(f"Model: {MODEL_PATH}")
    print(f"Static folder: {app.static_folder}")
    print(f"Template folder: {app.template_folder}")
    print("-" * 60)

    # Check initial status
    if os.path.exists(MODEL_PATH):
        print(f"✓ Model file exists ({os.path.getsize(MODEL_PATH)} bytes)")
        if load_model_if_exists():
            print("✓ Model loaded successfully")
        else:
            print("✗ Model file exists but could not be loaded")
    else:
        print("✗ Model file not found - Train model after adding students")

    print(f"\nAccess the application at: http://localhost:5000")
    print("=" * 60)

    # Run the application
    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000,
        use_reloader=False
    )