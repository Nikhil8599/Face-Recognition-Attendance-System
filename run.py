import os
import sys
import subprocess


def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'flask',
        'opencv-python',
        'numpy',
        'scikit-learn',
        'pandas'
    ]

    print("Checking dependencies...")
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package}")

    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        install = input("\nInstall missing packages? (y/n): ")
        if install.lower() == 'y':
            for package in missing_packages:
                print(f"Installing {package}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        else:
            print("Some features may not work without these packages.")

    # Check for optional packages
    try:
        import mediapipe
        print(f"✓ mediapipe (optional)")
    except ImportError:
        print(f"⚠ mediapipe not installed (optional, for better face detection)")


def setup_directories():
    """Create necessary directories"""
    directories = [
        'templates',
        'static/css',
        'static/js',
        'static/images',
        'dataset'
    ]

    print("\nSetting up directories...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ {directory}/")


def check_haar_cascade():
    """Check if Haar cascade file exists"""
    print("\nChecking Haar cascade file...")

    # Check OpenCV's default location
    import cv2
    default_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

    if os.path.exists(default_path):
        print(f"✓ Haar cascade found: {default_path}")
        return True
    elif os.path.exists('haarcascade_frontalface_default.xml'):
        print("✓ Haar cascade found in current directory")
        return True
    else:
        print("✗ Haar cascade file not found")
        print("\nDownloading Haar cascade file...")
        try:
            import urllib.request
            url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
            urllib.request.urlretrieve(url, 'haarcascade_frontalface_default.xml')
            print("✓ Haar cascade downloaded successfully")
            return True
        except Exception as e:
            print(f"✗ Failed to download: {e}")
            print("\nYou can download it manually from:")
            print("https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml")
            print("And place it in the current directory.")
            return False


def start_application():
    """Start the Flask application"""
    print("\n" + "=" * 60)
    print("Starting Digital Attendance System")
    print("=" * 60)

    # Import and run the app
    from app import app

    print(f"\nAccess the application at:")
    print(f"  http://localhost:5000")
    print(f"\nPress Ctrl+C to stop the server")
    print("=" * 60)

    # Run the app
    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000,
        use_reloader=False
    )


def main():
    """Main entry point"""
    print("\n" + "=" * 60)
    print("DIGITAL ATTENDANCE SYSTEM")
    print("Face Recognition Attendance Management")
    print("=" * 60)

    try:
        # Setup
        check_dependencies()
        setup_directories()
        check_haar_cascade()

        # Start application
        start_application()

    except KeyboardInterrupt:
        print("\n\nApplication stopped by user")
    except Exception as e:
        print(f"\nError: {e}")
        print("\nTroubleshooting steps:")
        print("1. Make sure all dependencies are installed")
        print("2. Check if port 5000 is available")
        print("3. Check the error message above")

        import traceback
        traceback.print_exc()

        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())