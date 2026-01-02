import os
import pickle
import sqlite3

print("=" * 60)
print("Digital Attendance System - Debug")
print("=" * 60)

# Check files
print("\n1. Checking Files:")
print(f"   app.py exists: {os.path.exists('app.py')}")
print(f"   model.py exists: {os.path.exists('model.py')}")
print(f"   model.pkl exists: {os.path.exists('model.pkl')}")
print(f"   attendance.db exists: {os.path.exists('attendance.db')}")
print(f"   dataset folder exists: {os.path.exists('dataset')}")

# Check model
if os.path.exists('model.pkl'):
    print(f"\n2. Model File:")
    print(f"   Size: {os.path.getsize('model.pkl')} bytes")
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
            print(f"   Model type: {type(model).__name__}")
            if hasattr(model, 'classes_'):
                print(f"   Number of classes: {len(model.classes_)}")
                print(f"   Classes: {model.classes_}")
            else:
                print(f"   Model has no classes_ attribute")
    except Exception as e:
        print(f"   Error loading model: {e}")

# Check database
if os.path.exists('attendance.db'):
    print(f"\n3. Database:")
    try:
        conn = sqlite3.connect('attendance.db')
        c = conn.cursor()

        # Check students table
        c.execute("SELECT COUNT(*) FROM students")
        student_count = c.fetchone()[0]
        print(f"   Students in DB: {student_count}")

        if student_count > 0:
            c.execute("SELECT id, name FROM students LIMIT 5")
            students = c.fetchall()
            print(f"   First 5 students:")
            for sid, name in students:
                print(f"     ID {sid}: {name}")

        # Check attendance table
        c.execute("SELECT COUNT(*) FROM attendance")
        attendance_count = c.fetchone()[0]
        print(f"   Attendance records: {attendance_count}")

        conn.close()
    except Exception as e:
        print(f"   Error reading database: {e}")

# Check dataset
if os.path.exists('dataset'):
    print(f"\n4. Dataset:")
    student_dirs = [d for d in os.listdir('dataset')
                    if os.path.isdir(os.path.join('dataset', d))]
    print(f"   Student directories: {len(student_dirs)}")

    for sid in student_dirs[:5]:  # Show first 5
        folder = os.path.join('dataset', sid)
        images = [f for f in os.listdir(folder)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"   Student {sid}: {len(images)} images")
        if images:
            print(f"     Sample images: {', '.join(images[:3])}")

print("\n" + "=" * 60)
print("Debug Complete")
print("=" * 60)