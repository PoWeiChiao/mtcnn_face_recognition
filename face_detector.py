import cv2
import os
from sympy import Point
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN

class FaceDetector:
    def __init__(self, image_size: int = 160, thresholds: list = None, 
                 min_face_size: int = 60, saved_path: str = "faces/"):
        if thresholds is None:
            thresholds = [0.4, 0.5, 0.5]
        self.image_size = image_size
        self.thresholds = thresholds
        self.min_face_size = min_face_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.mtcnn = MTCNN(image_size=self.image_size, keep_all=True, 
                           thresholds=self.thresholds, min_face_size=self.min_face_size, 
                           device=self.device)
        self.all_people_faces = {}
        self.saved_path = saved_path
        self._load_faces()

    def _load_faces(self) -> None:
        """Load and encode faces from the specified directory."""
        for file in os.listdir(self.saved_path):
            img_path = os.path.join(self.saved_path, file)
            img = cv2.imread(img_path)
            cropped = self.mtcnn(img)
            if cropped is not None:
                self.all_people_faces[file.split('.')[0]] = self._encode(cropped)[0, :]

    def _encode(self, img: torch.Tensor) -> torch.Tensor:
        """Encode a given face image using the pre-trained model."""
        return self.resnet(img.to(self.device))

    def _detect_box(self, img, save_path: str = None) -> tuple:
        """Detect faces in an image and optionally save them."""
        batch_boxes, batch_probs, batch_points = self.mtcnn.detect(img, landmarks=True)
        if not self.mtcnn.keep_all:
            batch_boxes, batch_probs, batch_points = self.mtcnn.select_boxes(
                batch_boxes, batch_probs, batch_points, img, method=self.selection_method
            )
        faces = self.mtcnn.extract(img, batch_boxes, save_path)
        return batch_boxes, faces, batch_points

    def detect(self, cam: int = 0, thres: float = 0.9, face_recognition: bool = True, face_points: bool = False) -> None:
        """Capture video from a camera, detect faces in real-time, and label them."""
        vdo = cv2.VideoCapture(cam)
        while vdo.isOpened():
            ret, img0 = vdo.read()
            if not ret:
                break

            batch_boxes, cropped_images, batch_points = self._detect_box(img0)
            if batch_boxes is not None:
                for box, cropped, points in zip(batch_boxes, cropped_images, batch_points):
                    x, y, x2, y2 = map(int, box)
                    min_key = 'Undetected'
                    if face_recognition:
                        img_embedding = self._encode(cropped.unsqueeze(0))
                        if self.all_people_faces:
                            detect_dict = {k: (v - img_embedding).norm().item() for k, v in self.all_people_faces.items()}
                            min_key, min_value = min(detect_dict.items(), key=lambda item: item[1])
                            if min_value >= thres:
                                min_key = 'Undetected'

                    color = (0, 255, 0) if min_key != 'Undetected' else (0, 0, 255)
                    cv2.rectangle(img0, (x, y), (x2, y2), color, 2)
                    if face_recognition:
                        cv2.putText(img0, min_key, (x + 5, y + 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
                    if face_points:
                        for point in points:
                            cv2.circle(img0, tuple(map(int, point)), 10, (255, 255, 255), -1)

            cv2.imshow("output", img0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        vdo.release()
        cv2.destroyAllWindows()