from face_detector import FaceDetector

def main() -> None:
    """Initialize FaceDetector and start detection using the default camera."""
    fd = FaceDetector()
    fd.detect(cam=0, thres=0.9, face_recognition=True, face_points=False)

if __name__ == "__main__":
    main()