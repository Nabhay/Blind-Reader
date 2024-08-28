import os
import time
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from collections import OrderedDict

from CRAFT import craft_utils
from CRAFT import imgproc
from CRAFT import file_utils
from CRAFT.craft import CRAFT

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
    """Pre-process the image by enhancing contrast and applying adaptive thresholding."""


    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    threshold_value = np.percentile(gray, 23)
    _, binary_image = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
    cv2.imshow("Scanner Effect Image", binary_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return binary_image


def save_boxes_as_images(image, boxes, output_dir):
    """Save detected text boxes as individual images."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    #image = preprocess_image(image)
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
            box_filename = os.path.join(output_dir, f"box_{i}.jpg")
            cv2.imwrite(box_filename, box_image)

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

def process_image(image_path, trained_model='CRAFT\craft_mlt_25k.pth', text_threshold=0.7, low_text=0.4, link_threshold=0.4, cuda=False, canvas_size=1280, mag_ratio=1.5, poly=False, show_time=False, refine=False, refiner_model=None, output_dir='./result'):
    """Process a single image for text detection."""
    net = CRAFT()

    print(f'Loading weights from checkpoint ({trained_model})')
    if cuda:
        net.load_state_dict(copyStateDict(torch.load(trained_model)))
    else:
        net.load_state_dict(copyStateDict(torch.load(trained_model, map_location='cpu')))

    if cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()

    refine_net = None
    if refine:
        from CRAFT.refinenet import RefineNet
        refine_net = RefineNet()
        print(f'Loading weights of refiner from checkpoint ({refiner_model})')
        if cuda:
            refine_net.load_state_dict(copyStateDict(torch.load(refiner_model)))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(copyStateDict(torch.load(refiner_model, map_location='cpu')))
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

    file_utils.saveResult(image_path, image[:,:,::-1], polys, dirname=image_result_folder)

def main():
    pass

if __name__ == '__main__':
    main()