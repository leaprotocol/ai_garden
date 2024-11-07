import cv2
import torch
from transformers import AutoProcessor, AutoModelForObjectDetection

# Load the processor and model
model_checkpoint = "microsoft/OmniParser"  # Replace with the actual model checkpoint
processor = AutoProcessor.from_pretrained(model_checkpoint)
model = AutoModelForObjectDetection.from_pretrained(model_checkpoint)

# Load your image
image_path = "path/to/your/image.jpg"  # Replace with your image path
image = cv2.imread(image_path)

# Prepare the image for the model
inputs = processor(images=image, return_tensors="pt")

# Perform inference to get bounding boxes
with torch.no_grad():
    outputs = model(**inputs)

# Extract bounding boxes and scores from outputs
boxes = outputs.logits  # Assuming logits contain bounding box predictions
scores = outputs.scores  # Assuming scores contain confidence scores

# Filter boxes based on a confidence threshold (e.g., 0.5)
threshold = 0.5
filtered_boxes = []
for box, score in zip(boxes, scores):
    if score > threshold:
        filtered_boxes.append(box)

# Visualize bounding boxes on the image
for box in filtered_boxes:
    x1, y1, x2, y2 = box.int().tolist()  # Convert to integer coordinates
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw rectangle

# Show the image with bounding boxes
cv2.imshow('Bounding Boxes', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
