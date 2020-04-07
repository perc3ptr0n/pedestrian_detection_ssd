from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor

import cv2
import sys
from glob import glob

net_type = 'mb1-ssd'
model_path = 'version1.3/mb1-ssd-Epoch-499-Loss-1.4479286074638367-CrossVal-1_2_3_4.pth'
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

f = glob('/home/borys/Documents/PEDESTRIAN DETECTION/new_test/*.jpg')
f += glob('/home/borys/Documents/PEDESTRIAN DETECTION/new_test/*.jpeg')
f += glob('/home/borys/Documents/PEDESTRIAN DETECTION/new_test/*.webp')

for s in f:
    path_to_image = s
    img = cv2.imread(path_to_image)
    img_BGR = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    boxes, labels, probs = predictor.predict(img_BGR, 10, 0.4)
    print(f"Found {len(probs)} objects.")

    for i in range(boxes.size(0)):
        box = boxes[i, :]
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 2)
        # label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
        # cv2.putText(img, label, (box[0] + 20, box[1] + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)  # line type

    cv2.imshow('sample with bb', img)
    cv2.waitKey(0)
cv2.destroyAllWindows()
