import os
import shutil
import sqlite3

print("=" * 60)
print("CLEANUP SCRIPT - Digital Attendance System")
print("=" * 60)

# Files to delete
files_to_delete = [
    "model.pkl",
    "train_status.json",
    "attendance.db"
]

# Directories to clear
dirs_to_clear = [
    "dataset"
]

print("\n1. Deleting model files...")
for file in files_to_delete:
    if os.path.exists(file):
        os.remove(file)
        print(f"✓ Deleted: {file}")
    else:
        print(f"✗ Not found: {file}")

print("\n2. Clearing dataset directory...")
for dir_path in dirs_to_clear:
    if os.path.exists(dir_path):
        # Remove all contents but keep directory
        for item in os.listdir(dir_path):
            item_path = os.path.join(dir_path, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
                print(f"✓ Deleted folder: {item}")
            else:
                os.remove(item_path)
                print(f"✓ Deleted file: {item}")
    else:
        print(f"✗ Directory not found: {dir_path}")

print("\n3. Reinitializing database...")
try:
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()

    # Drop and recreate tables
    c.execute("DROP TABLE IF EXISTS attendance")
    c.execute("DROP TABLE IF EXISTS students")

    # Recreate tables
    c.execute('''CREATE TABLE students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                roll TEXT,
                class TEXT,
                section TEXT,
                reg_no TEXT,
                created_at TEXT
            )''')

    c.execute('''CREATE TABLE attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER,
                name TEXT,
                timestamp TEXT
            )''')

    conn.commit()
    conn.close()
    print("✓ Database reinitialized successfully")
except Exception as e:
    print(f"✗ Error reinitializing database: {e}")

print("\n" + "=" * 60)
print("CLEANUP COMPLETE")
print("=" * 60)
print("\nSystem is now ready for fresh start.")
print("Please:")
print("1. Add students using the web interface")
print("2. Capture face images for each student")
print("3. Train the model")
print("4. Start marking attendance")