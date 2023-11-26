import cv2
import numpy as np
import random
import torch
import argparse
from torch import nn


class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super(Ensemble, self).__init__()


class Conv(nn.Module):
    # Standard convolution
    def __init__(
        self, c1, c2, k=1, s=1, p=None, g=1, act=True
    ):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, self.autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = (
            nn.SiLU()
            if act is True
            else (act if isinstance(act, nn.Module) else nn.Identity())
        )

    @staticmethod
    def autopad(k, p=None):  # kernel, padding
        # Pad to 'same'
        if p is None:
            p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
        return p


def attempt_load(weights, map_location=None):
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt = torch.load(w, map_location=map_location)  # load
        model.append(
            ckpt["ema" if ckpt.get("ema") else "model"].float().fuse().eval()
        )  # FP32 model

    # Compatibility updates
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True  # pytorch 1.7.0 compatibility
        elif type(m) is nn.Upsample:
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

    if len(model) == 1:
        return model[-1]  # return model
    else:
        print("Ensemble created with %s\n" % weights)
        for k in ["names", "stride"]:
            setattr(model, k, getattr(model[-1], k))
        return model  # return ensemble


def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    import time
    import torchvision

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        if nc == 1:
            x[:, 5:] = x[
                :, 4:5
            ]  # for models with one class, cls_loss is 0 and cls_conf is always 0.5,
            # so there is no need to multiplicate.
        else:
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3e3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(
                1, keepdim=True
            )  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f"WARNING: NMS time limit {time_limit}s exceeded")
            break  # time limit exceeded

    return output


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (
        (
            torch.min(box1[:, None, 2:], box2[:, 2:])
            - torch.max(box1[:, None, :2], box2[:, :2])
        )
        .clamp(0)
        .prod(2)
    )
    return inter / (
        area1[:, None] + area2 - inter
    )  # iou = inter / (area1 + area2 - inter)


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(
            img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1]
        )  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (
            img1_shape[0] - img0_shape[0] * gain
        ) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2


def letterbox(
    img,
    new_shape=(640, 640),
    color=(114, 114, 114),
    auto=True,
    scaleFill=False,
    scaleup=True,
    stride=32,
):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border
    return img, ratio, (dw, dh)


def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    import random

    tl = (
        line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )


def detect(model_path, orig_img, save_img=False):
    model = attempt_load(model_path, map_location=torch.device("cpu"))
    # image_path = "analog_meter/images/test/analog_meter_37.png"

    # Run the Inference and draw predicted bboxes
    img = orig_img.copy()
    img = letterbox(img, 640, 32)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img.copy()).to(torch.device("cpu"))
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    # print(img.shape)

    # Get names and colors
    names = model.module.names if hasattr(model, "module") else model.names
    # idx = names.index("clock")
    # names[idx] = "meter"
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Inference
    with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
        pred = model(img, augment=False)[0]

        pred = non_max_suppression(pred)

        bboxes = []
        for i, det in enumerate(pred):  # detections per image
            s, im0 = "", orig_img

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if True:  # Write to file
                        xywh = (
                            (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn)
                            .view(-1)
                            .tolist()
                        )  # normalized xywh
                        line = (
                            (cls, *xywh, conf) if False else (cls, *xywh)
                        )  # label format
                        with open("random" + ".txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")

                    if save_img:  # Add bbox to image
                        label = f"{names[int(cls)]} {conf:.2f}"
                        plot_one_box(
                            xyxy,
                            im0,
                            label=label,
                            color=colors[int(cls)],
                            line_thickness=1,
                        )

                    bboxes.append(list(map(int, xyxy)))

            if save_img:
                cv2.imshow(image_path, im0)
                cv2.waitKey(0)  # 1 millisecond

        return bboxes


def check_line(x1, y1, x2, y2, img):
    # height = img.shape[0]
    # if (
    #     abs(y1 - y2) > 0.4 * height
    #     or 0.3 * height <= y1 <= 0.7 * height
    #     or 0.3 * height <= y2 <= 0.7 * height
    # ):
    #     print(abs(y1 - y2) > 0.4 * height)
    #     print(0.3 * height <= y1 <= 0.7 * height)
    #     print(0.3 * height <= y2 <= 0.7 * height)
    #     return True
    # return False

    h, w = img.shape[:-1]

    cy, cx = h/2, w/2
    dist_to_center_1 = np.sqrt((x1 - cx)**2 + (y1 - cy)**2)
    dist_to_center_2 = np.sqrt((x2 - cx)**2 + (y2 - cy)**2)
    dist_to_center = dist_to_center_1 if dist_to_center_1 < dist_to_center_2 else dist_to_center_2
    # print()
    # print((cx, cy))
    # print((cy-y1)*(cy-y2))
    # print((cx-x1)*(cx-x2))
    # print(dist_to_center > 0.1*h)
    # print(((cy-y1)*(cy-y2) < 0 or (cx-x1)*(cx-x2) < 0))
    # cv2.line(crop_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # cv2.imshow("img", crop_img)
    # cv2.waitKey(0)
    if dist_to_center > 0.1*h and not ((cy-y1)*(cy-y2) < 0 and (cx-x1)*(cx-x2) < 0):
        # print("returned 0")
        return 0

    slope = (y2-y1)/(x2-x1 if x2-x1!=0 else x2-x1+1)
    y = y1 + slope*(w/2-x1)
    if abs(y-h/2) <= 0.1*h:
        return slope
    return 0

def identify_region(img, lines):
    h, w = img.shape[:-1]
    cx, cy = w/2, h/2

    longest_line = (cy, 0)
    for line in lines:
        x1, y1, x2, y2 = line
        dist1 = np.sqrt((x1 - cx)**2 + (y1 - cy)**2)
        dist2 = np.sqrt((x2 - cx)**2 + (y2 - cy)**2)
        if dist1 > dist2:
            dist = dist1
            y = y1
        else:
            dist = dist2
            y = y2

        if dist > longest_line[-1]:
            longest_line = (y, dist)
    
    if longest_line[0] < cy:
        return 1    # upper
    elif longest_line[0] > cy:
        return -1    # lower
    return 0  


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", default="yolov7.pt", help="path to the inference model")
    parser.add_argument("-i", "--image_path", help="path to input image")
    parser.add_argument("-v0", "--value_at_zero", help="Reading at needle angle of zero")
    parser.add_argument("-v1", "--value_at_one_eighty", help="Reading at needle angle of zero")
    parser.add_argument("-f", "--write_to_file", default=False, help="boolean for write read value to output file")
    args = parser.parse_args()
    image_path = args.image_path
    model_path = args.model_path
    value_at_zero = float(args.value_at_zero)
    value_at_one_eighty = float(args.value_at_one_eighty)
    write_to_file = bool(args.write_to_file)

    orig_img = cv2.imread(image_path)
    bboxes = detect(model_path, orig_img)

    bbox = bboxes[-1]
    crop_img = orig_img[bbox[1] : bbox[3], bbox[0] : bbox[2]]

    # h, w = crop_img.shape[:-1]
    # midpoint = h/2, w/2
    # crop_img = crop_img[int(midpoint[0]-h/3): int(midpoint[0]+h/3), int(midpoint[1]-w/3): int(midpoint[1]+w/3)]

    # crop_img = orig_img[bbox[1] : int(bbox[1] + (bbox[3] - bbox[1])*0.55), bbox[0] : bbox[2]]
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blurred, 50, 150, None, 3)

    # lsd = cv2.createLineSegmentDetector(0)
    # lines = lsd.detect(canny)
    # print(len(lines[0]))
    # y_dist = []
    # for dline in lines[0]:
    #     if np.sqrt((dline[0][3] - dline[0][1])**2 + (dline[0][2] - dline[0][0])**2) > 50:
    #         y_dist.append(dline[0])
    #     x1 = int(round(dline[0][0]))
    #     y1 = int(round(dline[0][1]))
    #     x2 = int(round(dline[0][2]))
    #     y2 = int(round(dline[0][3]))
    #     # cv2.line(crop_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # print(y_dist)
    # for line in y_dist:
    #     cv2.line(crop_img, (int(line[0]), int(line[1])), (int(line[2]), int(line[3])), (0, 255, 0), 2)

    # cv2.imshow("img", crop_img)
    # cv2.waitKey(0)
    # cv2.imwrite("output_imgs_final/lsd_image.png", crop_img)

    # lines = cv2.HoughLines(canny, 1, np.pi / 180, 150, None, 0, 0)
    # slopes = []
    # angles = []
    # for line in lines:
    #     rho, theta = line[0]
    #     a = np.cos(theta)
    #     b = np.sin(theta)
    #     x0 = a * rho
    #     y0 = b * rho
    #     x1 = int(x0 + 1000 * (-b))
    #     y1 = int(y0 + 1000 * (a))
    #     x2 = int(x0 - 1000 * (-b))
    #     y2 = int(y0 - 1000 * (a))

    lines = cv2.HoughLinesP(canny, 1, np.pi / 180, 15, None, 100, 20)
    filtered_lines = []
    slopes = []
    angles = []
    if lines is None:
        exit()
    for l in lines:
        line = l[0]
        x1 = line[0]
        y1 = line[1]
        x2 = line[2]
        y2 = line[3]

        slope = check_line(x1, y1, x2, y2, crop_img)
        if slope:
            filtered_lines.append([x1, y1, x2, y2])
            slopes.append(-slope)
            angle = np.arctan(-slope)
            angle = angle/np.pi * 180
            if angle < 0:
                angles.append(180 + angle)
            else:
                angles.append(angle)
    #         cv2.line(crop_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # print(filtered_lines, slopes, angles)
    # cv2.imshow("img", crop_img)
    # cv2.waitKey(0)

    region = identify_region(crop_img, filtered_lines)
    if not region:
        exit()
    
    approx_angle = sum(angles)/len(angles)
    # print(region)
    # print("slopes", slopes)
    # print("angles", angles)
    # print(approx_angle)

    if region == 1:
        final_angle = approx_angle
    elif approx_angle <= 90:
        final_angle = approx_angle + 180
    else:
        final_angle = -(180 - approx_angle)

    reading = value_at_zero - final_angle * (value_at_zero - value_at_one_eighty)/180
    print(f"reading: {reading}, final_angle: {final_angle}")

    # import os
    # filename = os.path.basename(image_path)
    # cv2.imwrite(f"output_imgs_final/{filename}", crop_img)

    if write_to_file:
        with open("reading.txt", "w") as file:
            file.write(f"{reading}, {final_angle}")

