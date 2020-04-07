import numpy as np
import pathlib
import cv2
import pandas as pd
import copy

class OpenImagesDataset:

    def __init__(self, root, k_folds,
                 transform=None, target_transform=None,
                 dataset_type="train", balance_data=False):
        self.root = pathlib.Path(root)
        self.transform = transform
        self.target_transform = target_transform
        self.dataset_type = dataset_type.lower()

        self.data, self.class_names, self.class_dict = self._read_data(k_folds)
        self.balance_data = balance_data
        self.min_image_num = -1
        if self.balance_data:
            self.data = self._balance_data()
        self.ids = [info['image_path'] for info in self.data]

        self.class_stat = None

    def _getitem(self, index):
        image_info = self.data[index]
        image = self._read_image(image_info['image_path'])
        # duplicate boxes to prevent corruption of dataset
        boxes = copy.copy(image_info['boxes'])

        # duplicate labels to prevent corruption of dataset
        labels = copy.copy(image_info['labels'])
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        return image_info['image_path'], image, boxes, labels

    def __getitem__(self, index):
        _, image, boxes, labels = self._getitem(index)
        return image, boxes, labels

    def get_annotation(self, index):
        """To conform the eval_ssd implementation that is based on the VOC dataset."""
        image_id, image, boxes, labels = self._getitem(index)
        is_difficult = np.zeros(boxes.shape[0], dtype=np.uint8)
        return image_id, (boxes, labels, is_difficult)

    def get_image(self, index):
        image_info = self.data[index]
        image = self._read_image(image_info['image_path'])
        if self.transform:
            image, _ = self.transform(image)
        return image

    def _read_data(self, k_folds):
        paths_to_images = []

        if self.dataset_type == 'train':
            for k in k_folds:
                annotation_file = open(f"{self.root}/{self.dataset_type}_set_%d.txt" % k, 'r')
                # annotations = pd.read_csv(annotation_file)
                for s in annotation_file:
                    paths_to_images.append(s[:-1])
                annotation_file.close()

        if self.dataset_type == 'test':
            annotation_file = open(f"{self.root}/{self.dataset_type}_set.txt", 'r')
            # annotations = pd.read_csv(annotation_file)
            for s in annotation_file:
                paths_to_images.append(s[:-1])
            annotation_file.close()

        # class_names = ['BACKGROUND'] + sorted(list(annotations['ClassName'].unique()))
        class_names = ['BACKGROUND', 'pedestrian']
        class_dict = {class_name: i for i, class_name in enumerate(class_names)}
        data = []
        for path in paths_to_images:
            txt_path = '/'.join(path.split('/')[:-2]) + '/labels/' + path.split('/')[-1][:-4] + '.txt'

            box_txt = open(txt_path, 'r')
            boxes = []
            for s in box_txt:
                box = s.split(' ')
                boxes.append([int(box[0]), int(box[1]), int(box[2]), int(box[3][:-1])])
            box_txt.close()

            # boxes = group.loc[:, ["XMin", "YMin", "XMax", "YMax"]].values.astype(np.float32)
            # make labels 64 bits to satisfy the cross_entropy function
            labels = []
            for i in range(len(boxes)):
                labels.append(1)
            boxes = np.asarray(boxes, dtype=np.float32)
            labels = np.asarray(labels, dtype=np.int32)
            data.append({
                'image_path': path,
                'boxes': boxes,
                'labels': labels
            })
        return data, class_names, class_dict

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.class_stat is None:
            self.class_stat = {name: 0 for name in self.class_names[1:]}
            for example in self.data:
                for class_index in example['labels']:
                    class_name = self.class_names[class_index]
                    self.class_stat[class_name] += 1
        content = ["Dataset Summary:"
                   f"Number of Images: {len(self.data)}",
                   f"Minimum Number of Images for a Class: {self.min_image_num}",
                   "Label Distribution:"]
        for class_name, num in self.class_stat.items():
            content.append(f"\t{class_name}: {num}")
        return "\n".join(content)

    def _read_image(self, image_path):
        image_file = image_path
        image = cv2.imread(str(image_file))
        if image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _balance_data(self):
        label_image_indexes = [set() for _ in range(len(self.class_names))]
        for i, image in enumerate(self.data):
            for label_id in image['labels']:
                label_image_indexes[label_id].add(i)
        label_stat = [len(s) for s in label_image_indexes]
        self.min_image_num = min(label_stat[1:])
        sample_image_indexes = set()
        for image_indexes in label_image_indexes[1:]:
            image_indexes = np.array(list(image_indexes))
            sub = np.random.permutation(image_indexes)[:self.min_image_num]
            sample_image_indexes.update(sub)
        sample_data = [self.data[i] for i in sample_image_indexes]
        return sample_data





