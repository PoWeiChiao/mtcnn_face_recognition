import os
import cv2
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from types import MethodType
import time

### helper function
def encode(img):
    res = resnet(torch.Tensor(img))
    return res

def detect_box(self, img, save_path=None):
    # Detect faces
    batch_boxes, batch_probs, batch_points = self.detect(img, landmarks=True)
    # Select faces
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

def detect(cam=0, thres=0.9):
    vdo = cv2.VideoCapture(cam)
    while vdo.grab():
        _, img0 = vdo.retrieve()
        ts0 = time.time()
        batch_boxes, cropped_images = mtcnn.detect_box(img0)
        ts1 = time.time()
        print('face detection: {:.1f}'.format(ts1 - ts0))

        if cropped_images is not None:
            for box, cropped in zip(batch_boxes, cropped_images):
                ts2 = time.time()
                x, y, x2, y2 = [int(x) for x in box]
                img_embedding = encode(cropped.unsqueeze(0))
                detect_dict = {}
                for k, v in all_people_faces.items():
                    detect_dict[k] = (v - img_embedding).norm().item()
                min_key = min(detect_dict, key=detect_dict.get)
                if detect_dict[min_key] >= thres:
                    min_key = 'Undetected'
                ts3 = time.time()
                print('face recognition: {:.1f}'.format(ts3 - ts2))
                if min_key == 'Undetected':
                    cv2.rectangle(img0, (x, y), (x2, y2), (0, 0, 255), 2)
                else:
                    cv2.rectangle(img0, (x, y), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                  img0, min_key, (x + 5, y + 10), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
                
        ### display
        cv2.imshow("output", img0)
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    detect(0)
