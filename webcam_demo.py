import cv2
import os
import time
import torch
from types import MethodType
from facenet_pytorch import InceptionResnetV1, MTCNN

def encode(img: torch.Tensor) -> torch.Tensor:
    """
    Encodes an image using a pre-trained ResNet model.

    Args:
        img (torch.Tensor): A numerical representation of an image.

    Returns:
        torch.Tensor: The feature representation of the input image.
    """
    return resnet(img)

def detect_box(self, img, save_path: str = None) -> tuple:
    """
    Detect faces in an image and optionally save them.

    Args:
        img: The image in which faces are to be detected.
        save_path: An optional path where the extracted face images will be saved.

    Returns:
        A tuple containing:
        - batch_boxes: The bounding boxes of the detected faces.
        - faces: The extracted face images.
    """
    # Detect faces
    batch_boxes, batch_probs, batch_points = self.detect(img, landmarks=True)
    
    # Select faces if not keeping all
    if not self.keep_all:
        batch_boxes, batch_probs, batch_points = self.select_boxes(
            batch_boxes, batch_probs, batch_points, img, method=self.selection_method
        )
    
    # Extract faces
    faces = self.extract(img, batch_boxes, save_path)
    
    return batch_boxes, faces

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = InceptionResnetV1(pretrained='vggface2').eval()
mtcnn = MTCNN(image_size=160, keep_all=True, thresholds=[0.4, 0.5, 0.5], min_face_size=60, device=device)
mtcnn.detect_box = MethodType(detect_box, mtcnn)

saved_pictures = "faces/"
all_people_faces = {}   
for file in os.listdir(saved_pictures):
    img = cv2.imread(f'{saved_pictures}/{file}')
    cropped = mtcnn(img)
    if cropped is not None:
        all_people_faces[file.split('.')[0]] = encode(cropped)[0, :]

def detect(cam: int = 0, thres: float = 0.9) -> None:
    """
    Captures video from a specified camera, detects faces in each frame using MTCNN,
    and performs face recognition by comparing detected faces against a known set of embeddings.
    Draws rectangles around detected faces and labels them based on recognition results.

    :param cam: Camera index to capture video from.
    :param thres: Threshold for face recognition.
    """
    vdo = cv2.VideoCapture(cam)
    
    while vdo.isOpened():
        ret, img0 = vdo.read()
        if not ret:
            break
        
        ts0 = time.time()
        batch_boxes, cropped_images = mtcnn.detect_box(img0)
        print(f'face detection: {time.time() - ts0:.1f}s')

        if cropped_images is not None:
            for box, cropped in zip(batch_boxes, cropped_images):
                ts2 = time.time()
                x, y, x2, y2 = map(int, box)
                img_embedding = encode(cropped.unsqueeze(0))
                
                detect_dict = {k: (v - img_embedding).norm().item() for k, v in all_people_faces.items()}
                min_key = min(detect_dict, key=detect_dict.get)
                if detect_dict[min_key] >= thres:
                    min_key = 'Undetected'
                
                print(f'face recognition: {time.time() - ts2:.1f}s')
                
                color = (0, 255, 0) if min_key != 'Undetected' else (0, 0, 255)
                cv2.rectangle(img0, (x, y), (x2, y2), color, 2)
                cv2.putText(img0, min_key, (x + 5, y + 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("output", img0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vdo.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect(0)

