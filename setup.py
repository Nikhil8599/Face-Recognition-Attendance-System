from setuptools import setup, find_packages

setup(
    name="digital-attendance-system",
    version="1.0.0",
    description="Face Recognition Attendance System",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "flask>=2.3.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
    ],
    extras_require={
        "better_detection": ["mediapipe>=0.10.0"],
    },
    python_requires=">=3.8",
)