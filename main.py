import cv2
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile
from torchvision import models, transforms
from ultralytics import YOLO

app = FastAPI()

# --- Setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class_names = ['cataract', 'conjunctivitis', 'healthy', 'scleral_icterus']

URGENCY_MAP = {
    'healthy': 'Routine',
    'conjunctivitis': 'Semi-Urgent',
    'cataract': 'Urgent',
    'scleral_icterus': 'Urgent',
}

# --- EfficientNet (classification) ---
classifier_model = models.efficientnet_b0(weights=None)
classifier_model.classifier[1] = nn.Linear(
    classifier_model.classifier[1].in_features, len(class_names)
)

checkpoint_path = 'checkpoint_epoch25.pth'  # confirm exact filename/extension on your server
try:
  classifier_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
  print(f"SUCCESS: Loaded EfficientNet weights from '{checkpoint_path}'")
except Exception as e:
  print(f"ERROR: Failed to load '{checkpoint_path}'. Exception: {e}")

classifier_model.to(device)
classifier_model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    ),
])

# --- YOLO (eye detector) ---
yolo_path = 'eye_detector_v2.pt'
try:
  yolo_model = YOLO(yolo_path)
  print(f"SUCCESS: Loaded YOLO eye detector from '{yolo_path}'")
except Exception as e:
  print(f"ERROR: Failed to load YOLO model '{yolo_path}'. Exception: {e}")

YOLO_CONF_THRESHOLD = 0.4
CROP_PADDING = 0.15

# --- New: confidence-tier thresholds (Concern 1) ---
CONFIDENCE_HIGH = 0.80
CONFIDENCE_LOW = 0.50

# --- New: glare-detection thresholds (Concern 2) ---
GLARE_BRIGHTNESS_THRESHOLD = 235
GLARE_AREA_FRACTION_THRESHOLD = 0.25


def pad_box(x1, y1, x2, y2, img_w, img_h, padding=CROP_PADDING):
  box_w, box_h = x2 - x1, y2 - y1
  pad_x, pad_y = box_w * padding, box_h * padding
  return (
      max(0, int(x1 - pad_x)),
      max(0, int(y1 - pad_y)),
      min(img_w, int(x2 + pad_x)),
      min(img_h, int(y2 + pad_y)),
  )


# --- New function (Concern 2) ---
def detect_large_glare(cropped_eye_bgr):
  gray = cv2.cvtColor(cropped_eye_bgr, cv2.COLOR_BGR2GRAY)
  h, w = gray.shape
  y1, y2 = int(h * 0.25), int(h * 0.75)
  x1, x2 = int(w * 0.25), int(w * 0.75)
  central = gray[y1:y2, x1:x2]
  bright_fraction = float(np.sum(central > GLARE_BRIGHTNESS_THRESHOLD)) / central.size
  return bright_fraction > GLARE_AREA_FRACTION_THRESHOLD, bright_fraction

@app.post('/predict')
async def predict_eye(file: UploadFile = File(...)):
  contents = await file.read()
  nparr = np.frombuffer(contents, np.uint8)
  img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
  img_h, img_w = img.shape[:2]

  results = yolo_model(img, conf=YOLO_CONF_THRESHOLD)
  boxes = results[0].boxes

  if len(boxes) == 0:
    cv2.imwrite('last_cropped_eye.jpg', img)
    return {
        'condition': None,
        'confidence': 0.0,
        'urgencyLevel': None,
        'eyeDetected': False,
        'message': 'No eye detected. Please retake the photo, centering your eye in the frame.',
    }

  if len(boxes) >= 2:
    cv2.imwrite('last_cropped_eye.jpg', img)
    return {
        'condition': None,
        'confidence': 0.0,
        'urgencyLevel': None,
        'eyeDetected': False,
        'message': 'Two eyes detected. Please retake showing only one eye.',
    }

  # Exactly one eye — crop with padding, then classify
  box = boxes[0].xyxy[0].cpu().numpy()
  x1, y1, x2, y2 = pad_box(*box, img_w, img_h)
  cropped_eye = img[y1:y2, x1:x2]
  cv2.imwrite('last_cropped_eye.jpg', cropped_eye)

  has_glare, glare_fraction = detect_large_glare(cropped_eye)
  print(f'Glare check: {glare_fraction:.2f} bright fraction (threshold {GLARE_AREA_FRACTION_THRESHOLD})')

  rgb_eye = cv2.cvtColor(cropped_eye, cv2.COLOR_BGR2RGB)
  input_tensor = transform(rgb_eye).unsqueeze(0).to(device)

  with torch.no_grad():
    outputs = classifier_model(input_tensor)
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    confidence, predicted_idx = torch.max(probabilities, 1)
    condition = class_names[predicted_idx.item()]
    conf_score = float(confidence.item())

  urgency_level = URGENCY_MAP.get(condition.lower(), 'Urgent')

  prob_dict = {cls: round(p, 4) for cls, p in zip(class_names, probabilities[0].tolist())}
  print(f'Model Probabilities: {prob_dict}')
  print(f'Predicted Condition: {condition} ({conf_score * 100:.1f}%)')

  readable_condition = condition.replace('_', ' ')

  if conf_score < CONFIDENCE_LOW:
    return {
        'condition': None,
        'confidence': round(conf_score, 2),
        'urgencyLevel': None,
        'eyeDetected': True,
        'message': 'Result inconclusive due to low confidence. Please consult an eye care professional or retake the photo in better lighting.',
    }

  message = None
  if has_glare and condition == 'cataract':
    message = (
        f'Possible {readable_condition} detected, but strong glare was also found — '
        'this can sometimes resemble cataract in photos. Consider retaking in softer lighting, '
        'and please consult an eye care professional either way.'
    )
  elif conf_score < CONFIDENCE_HIGH:
    message = f'Possible {readable_condition} detected with moderate confidence. Please consult an eye care professional to confirm.'

  return {
      'condition': condition,
      'confidence': round(conf_score, 2),
      'urgencyLevel': urgency_level,
      'eyeDetected': True,
      'message': message,
  }