import cv2
import time
import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from collections import OrderedDict
from CRAFT import craft_utils
from CRAFT import imgproc
from CRAFT import file_utils
from CRAFT.craft import CRAFT
from PIL import Image
import easyocr
from Secret_Parser import get_Secrets
import google.generativeai as genai
import pyttsx3
import argparse

def copyStateDict(state_dict):
    """Copy state dict for CRAFT model."""
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def preprocess_image(image):
    # Convert the image to grayscale if it is not already
    if len(image.shape) == 3:  # Check if the image is color
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the image to get a binary image
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find the coordinates of the non-zero pixels
    coords = np.column_stack(np.where(binary_image > 0))

    # Calculate the angle of the skew
    angle = cv2.minAreaRect(coords)[-1]

    # Adjust the angle
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # Check if the angle is within the allowable range
    if abs(angle) > 45:
        return image  # Return the original image without rotation

    # Get the image dimensions
    (h, w) = image.shape[:2]

    # Calculate the center of the image
    center = (w // 2, h // 2)

    # Rotate the image to correct the skew
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated

def save_boxes_as_images(image, boxes, name):
    """Save detected text boxes as individual images."""
    if not os.path.exists(name):
        os.makedirs(name)
    rows = []
    for box in boxes:
        if box is not None:
            y_center = np.mean(box[:, 1])
            added = False
            for row in rows:
                if abs(y_center - np.mean([np.mean(b[:, 1]) for b in row])) < 10:
                    row.append(box)
                    added = True
                    break
            if not added:
                rows.append([box])

    for row in rows:
        row.sort(key=lambda box: np.min(box[:, 0]))

    sorted_boxes = [box for row in rows for box in row]

    for i, box in enumerate(sorted_boxes):
        if box is not None:
            x_min, y_min = np.min(box[:, 0]), np.min(box[:, 1])
            x_max, y_max = np.max(box[:, 0]), np.max(box[:, 1])
            box_image = preprocess_image(image[int(y_min):int(y_max), int(x_min):int(x_max)])
            box_filename = os.path.join(name, f"box_{i}.png")
            cv2.imwrite(box_filename, box_image)
            box_to_text(name, i)

def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None, canvas_size=1280, mag_ratio=1.5):
    """Perform text detection on an image."""
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)
    x = Variable(x.unsqueeze(0))
    if cuda:
        x = x.cuda()

    with torch.no_grad():
        y, feature = net(x)

    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    render_img = np.hstack((score_text, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    return boxes, polys, ret_score_text

def load_craft_model(trained_model='CRAFT/craft_mlt_25k.pth', cuda=False):
    """Load the CRAFT model with the given weights."""
    net = CRAFT()
    print(f'Loading weights from checkpoint ({trained_model})')
    if cuda:
        net.load_state_dict(copyStateDict(torch.load(trained_model)))
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False
    else:
        net.load_state_dict(copyStateDict(torch.load(trained_model, map_location='cpu')))
    return net

def process_image(image_path, net, refine_net=None, text_threshold=0.7, low_text=0.4, link_threshold=0.4, cuda=False, canvas_size=1280, mag_ratio=1.5, poly=False, show_time=True, output_dir='./result'):
    """Process a single image for text detection."""
    net.eval()

    if refine_net:
        refine_net.eval()
        poly = True

    image = imgproc.loadImage(image_path)
    bboxes, polys, score_text = test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net, canvas_size, mag_ratio)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename, file_ext = os.path.splitext(os.path.basename(image_path))
    mask_file = os.path.join(output_dir, f"res_{filename}_mask.jpg")
    cv2.imwrite(mask_file, score_text)

    image_result_folder = os.path.join(output_dir, filename)
    if not os.path.exists(image_result_folder):
        os.makedirs(image_result_folder)
    save_boxes_as_images(image, bboxes, image_result_folder)

    file_utils.saveResult(filename, image[:,:,::-1], polys, dirname=image_result_folder)

def box_to_text(name, box_no):
    results = reader.recognize(f'{name}/box_{box_no}.png')

    image_path = os.path.join(name, f'box_{box_no}.png')
    
    # Use '..' to go back one directory and construct the path to the content file
    parent_dir = os.path.abspath(os.path.join(name, '..'))
    content_file_path = os.path.join(parent_dir, 'content.txt')

    with open(content_file_path, 'a') as file:
        for bbox, text, prob in results:
            file.write(f"{text} (probability: {prob})\n")

def is_image_blurry(image, threshold=50.0):
    """Check if the image is blurry using the Laplacian variance method."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold

def capture_and_save_image(base_dir=r"result", camera_id=0, width=1920, height=1080):
    """Capture an image from the camera, check for blurriness, and save it."""
    # Initialize camera
    cap = cv2.VideoCapture(camera_id)

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    while True:
        # Capture image
        ret, photo = cap.read()
        if not ret:
            cap.release()
            raise RuntimeError("Failed to capture image")

        # Check for blurriness
        if is_image_blurry(photo):
            print("Captured image is blurry. Retrying...")
            continue  # Retry capturing if the image is blurry

        break  # Exit loop if image is not blurry
    # Release camera
    cap.release()

    # Generate timestamp and create the corresponding directory
    timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
    print(timestamp)
    save_dir = os.path.join(base_dir, timestamp)
    if not os.path.exists(save_dir):
        print(save_dir)
        os.makedirs(save_dir)

    # Create the full save path
    filename = os.path.join(save_dir, f"{timestamp}.png")

    # Save the image
    cv2.imwrite(filename, photo)

    # Return the timestamp (which corresponds to the directory name)
    return timestamp

def llm_refinement(name, answer_query=False):
    global model, secrets

    # Define the paths
    content_file_path = os.path.join('result', name, 'content.txt')
    modified_file_path = os.path.join('result', name, 'modified.txt')
    image_path = os.path.join('result', name, f'{name}res_{name}.jpg')
    # Load the text content from content.txt
    with open(content_file_path, 'r') as file:
        text_content = file.read()

    image = Image.open(image_path).convert('RGB')

    

    if answer_query:
        sys_prompt = ("""
                        You are an assistant tasked with analyzing text and providing accurate answers to questions included in the frame. Your primary responsibility is to focus on the question, using the provided text as the sole source of information to construct clear, concise, and accurate responses.
                        Key Guidelines:
                        Focus on the Question: Your primary goal is to answer the question posed in the frame directly. Ensure your response is clear and addresses the query fully. Avoid adding unrelated information.
                        Use the Provided Text: Base your answers strictly on the text in the frame. Do not infer information beyond what is explicitly stated or implied within the text.
                        Clarity and Accuracy: Ensure your response is simple and easy to understand. Correct any ambiguities or errors in the source text that might hinder comprehension.
                        Conciseness: Avoid verbosity. Provide only the information needed to answer the question effectively.
                        No External Context: Do not include details or context from outside the provided text. Stick strictly to what is available within the frame.
                        Additional Considerations:

                        If the frame contains symbols, mathematical notation, or complex language, interpret these into standard English or spoken equivalents to enhance clarity.
                        If the question cannot be answered using the provided text, clearly state that the information is not available.
                        The user prompt starts NOW:""")
    else:
        sys_prompt = ("""
                    You are an assistant for the blind, tasked with converting provided text into a format that can be clearly understood when read aloud to a deaf person.
                    Primary Responsibilities:

                    Minimal Adjustments: Convert the text into standard English, making only minor changes for clarity and accuracy. Preserve the original structure and meaning as much as possible, ensuring the text remains recognizable.
                    Math Symbols and Greek Letters: Accurately convey mathematical symbols and Greek letters as they would be spoken. For example, replace symbols like "∑" with "sigma" and articulate fractions or equations in an intuitive manner.
                    Accuracy: Correct grammatical errors while preserving the overall content and intent. Avoid unnecessary rephrasing or rewriting unless required for clarity or comprehension.
                    Preserve Original Text: The essence and structure of the original text must remain intact. Adjustments should be subtle and made only when essential for understanding.
                    No Detailed Descriptions: Avoid providing detailed explanations or transcriptions. Add only the minimal context needed to clarify errors or ambiguities without altering the original meaning.
                    Additional Considerations:

                    When working with OCR-generated text, use the provided probabilities for each word to guide corrections. Prioritize accuracy by selecting the most probable corrections while maintaining the text's coherence.
                    Ensure all adjustments support the needs of blind users, ensuring clarity when read aloud.
                    The user prompt starts NOW:
                  """
    )

    response = model.generate_content(
        [sys_prompt, text_content, image],
        generation_config=genai.types.GenerationConfig(temperature=1.0)
    )

    with open(modified_file_path, 'w') as modified_file:
        modified_file.write(response.text)
    return response.text

def tts(text):
    engine.say(text)
    engine.runAndWait()

def main():
    global secrets, reader, model, engine

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process images for text detection and accessibility.")
    parser.add_argument('-c', '--camera_id', type=int, default=0, help="ID of the camera to use (default: 0)")
    parser.add_argument('--use_gpu', type=bool, default=False, help='Use GPU if available (default: False)')
    parser.add_argument('-b', '--braille', action='store_true', help="Enable Braille processing mode")
    parser.add_argument('-a', '--answer_query', action='store_true', help="Answer the question in frame")

    args = parser.parse_args()

    camera_id = args.camera_id
    cuda = args.use_gpu

    # Load CRAFT model
    net = load_craft_model(cuda=cuda)

    # Load secrets and configure Google Generative AI
    secrets = get_Secrets()
    genai.configure(api_key=secrets["Google_Gemini_Api"])

    # Initialize the Generative Model
    model = genai.GenerativeModel(model_name="gemini-1.5-pro")

    # Initialize EasyOCR Reader
    reader = easyocr.Reader(['en'], gpu=cuda)

    # Initialize Text-to-Speech Engine
    engine = pyttsx3.init()

    # Start processing
    t = time.time()
    name = capture_and_save_image(camera_id=camera_id)
    process_image(
        f'./result/{name}/{name}.png',
        net,
        output_dir=f'./result/{name}',
        cuda=cuda
    )
    text_to_be_read = llm_refinement(name, answer_query=args.answer_query)
    tts(text_to_be_read)
    print(f"Processing time: {time.time() - t:.2f} seconds")

    if args.braille:
        import Braille
        Braille.process_braille_file(os.path.join('result', name, 'modified.txt'))

if __name__ == '__main__':
    main()