from PIL import Image
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN

# Initialize the MTCNN model
mtcnn = MTCNN(keep_all=True)

# Load an image
image_path = 'path'  # Replace with your image path
image = Image.open(image_path)

# Detect faces in the image
boxes, probs, points = mtcnn.detect(image, landmarks=True)

# Plotting the results
plt.figure(figsize=(10, 10))
plt.imshow(image)

if boxes is not None:
    for box, point in boxes, points:
        # Draw bounding box
        plt.gca().add_patch(plt.Rectangle(
            (box[0], box[1]),  # x, y
            box[2] - box[0],   # width
            box[3] - box[1],   # height
            fill=False,
            color='red',
            linewidth=3
        ))

plt.axis('off')
plt.show()