from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
import cv2
import sys


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0], boxA[2], boxB[2])
    yA = max(boxA[1], boxB[1], boxA[3], boxB[3])
    xB = min(boxA[0], boxB[0], boxA[2], boxB[2])
    yB = min(boxA[1], boxB[1], boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, abs(xB - xA)) * max(0, abs(yB - yA))
    s1 = abs(boxA[0] - boxA[2]) * abs(boxA[1] - boxA[3])
    s2 = abs(boxB[0] - boxB[2]) * abs(boxB[1] - boxB[3])
    # compute the intersection over union by taking the intersection area and dividing it by the sum of
    # prediction + ground-truth areas - the interesection area
    iou = interArea / float(s1 + s2 - interArea)
    # return the intersection over union value
    return 1 / iou

net_type = 'mb1-ssd'
model_path = 'version1.3/mb1-ssd-Epoch-499-Loss-1.727292001247406-CrossVal-0_2_3_4.pth'
label_path = 'models/pedestrian-labels.txt'
image_path = '/home/borys/Documents/PEDESTRIAN DETECTION/PennFudanPed/yolo_data/images/PennPed00089.jpg'

class_names = [name.strip() for name in open(label_path).readlines()]

if net_type == 'vgg16-ssd':
    net = create_vgg_ssd(len(class_names), is_test=True)
elif net_type == 'mb1-ssd':
    net = create_mobilenetv1_ssd(len(class_names), is_test=True)
elif net_type == 'mb1-ssd-lite':
    net = create_mobilenetv1_ssd_lite(len(class_names), is_test=True)
elif net_type == 'mb2-ssd-lite':
    net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)
elif net_type == 'sq-ssd-lite':
    net = create_squeezenet_ssd_lite(len(class_names), is_test=True)
else:
    print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
    sys.exit(1)
net.load(model_path)

if net_type == 'vgg16-ssd':
    predictor = create_vgg_ssd_predictor(net, candidate_size=200)
elif net_type == 'mb1-ssd':
    predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)
elif net_type == 'mb1-ssd-lite':
    predictor = create_mobilenetv1_ssd_lite_predictor(net, candidate_size=200)
elif net_type == 'mb2-ssd-lite':
    predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200)
elif net_type == 'sq-ssd-lite':
    predictor = create_squeezenet_ssd_lite_predictor(net, candidate_size=200)
else:
    predictor = create_vgg_ssd_predictor(net, candidate_size=200)

f = open('./data/pedestrians/test_set.txt', 'r')

TP = 0
FP = 0
TN = 0
FN = 0

iou_threshold = 0.5
confidence_threshold = 0.5

for s in f:
    path_to_image = s
    img = cv2.imread(path_to_image[:-1])
    img_BGR = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    boxes, labels, probs = predictor.predict(img_BGR, 10, confidence_threshold)
    print(f"Found {len(probs)} objects.")


    h, w = img.shape[:2]
    path_to_txt = '/home/borys/Documents/PEDESTRIAN DETECTION/PennFudanPed/yolo_data/labels/%s.txt' % (
    s.split('/')[-1].split('.')[0])
    l = open(path_to_txt, 'r')
    color = (0, 255, 0)
    thickness = 2
    gt_boxes = []
    for o in l:
        x1, y1, x2, y2 = o.split(' ')
        print(int(x1), int(y1), int(x2), int(y2))
        gt_boxes.append([int(x1), int(y1), int(x2), int(y2)])
        img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

    for box in boxes:
        used = False
        for gt_box in gt_boxes:
            if bb_intersection_over_union(box, gt_box) >= iou_threshold:
                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 2)
                TP += 1
                used = True
            if used:
                break
        if not used:
            FP += 1

    for gt_box in gt_boxes:
        found = False
        for box in boxes:
            if bb_intersection_over_union(box, gt_box) >= iou_threshold:
                found = True
                break
        if not found:
            FN += 1

    cv2.imshow('sample with bb', img)
    cv2.waitKey(0)
cv2.destroyAllWindows()

print('TP: %d\nTN: %d\nFP: %d\nFN: %d\n' % (TP, TN, FP, FN))

recall = TP / (TP + FN)
print('Recall: %f\n' % recall)
fnr = FN / (FN + TP)

precision = TP / (TP + FP)
print('Precision: %f\n' % precision)

f1_score = 2 * (precision * recall) / (precision + recall)
print('F1 Score: %f' % f1_score)
