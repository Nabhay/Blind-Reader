'''
import cv2
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import numpy as np
import time

# Load the image using OpenCV
image_path = 'result//Motion in a 2d plane_page-0002//box_28.jpg'
image1 = cv2.imread(image_path)



# Convert the OpenCV image (inverted1) to a PIL image for OCR
pil_image = Image.fromarray(image1)

# Load processor and model
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')

# Move model to GPU if available
device = 'cpu' #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
t = time.time()
# Process image and move pixel values to GPU
pixel_values = processor(images=pil_image, return_tensors="pt").pixel_values.to(device)

# Generate text
generated_ids = model.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

# Print the generated text
print(generated_text)
print(time.time() - t)
'''

from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import requests
from PIL import Image
import cv2
import time

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

image_path = 'result//Motion in a 2d plane_page-0002//box_28.jpg'
image1 = cv2.imread(image_path)


t = time.time()
# Convert the OpenCV image (inverted1) to a PIL image for OCR
pil_image = Image.fromarray(image1)

pixel_values = processor(pil_image, return_tensors="pt").pixel_values
generated_ids = model.generate(pixel_values)

generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(time.time() - t)