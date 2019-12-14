import os
import cv2
import cPickle
import numpy as np
import json


datapath = 'output/valresults/resnet50/2step/val_dt.json'
ground_truth_path = 'evaluation/val_crowd_gt.json'


images = []
data = []
with open(datapath) as json_file:
    data = json.load(json_file)

with open(ground_truth_path) as json_file:
    ground_truth = json.load(json_file)
    images = ground_truth['images']
    annotations = ground_truth['annotations']

    i = 0
    for image in images:
        i += 1
        if i == 20:
            break
        id = image['id']
        folder = image['im_name'].split("_")[0]
        path = "data/crowdHuman/val/Images/{}".format(image['im_name'])
        img = cv2.imread(path)

        annotation = filter(lambda annotation:annotation['image_id'] == id, annotations)
        anno_pred = filter(lambda annotation: annotation['image_id'] == id, data)
        img1 = img.copy()
        for box in annotation:
            bbox = box['bbox']
            ignore = box['ignore']
            if ignore == 1:
                continue
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])),
                          (0, 255, 0), 3);
        cv2.imwrite('output/images_gt/{}'.format(image['im_name']), img)


        for box in anno_pred:
            bbox = box['bbox']
            cv2.rectangle(img1, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])),
                          (0, 255, 0), 3);

        cv2.imwrite('output/images_dt/{}'.format(image['im_name']), img1)





# {"ignore": 0, "image_id": 1, "vis_ratio": 0.7912524850894632, "bbox": [696, 414, 163, 503], "vis_bbox": [696, 414, 163, 398], "category_id": 1, "iscrowd": 0, "id": 1, "height": 503}
# with open(data) as json_file:
#     data = json.load(json_file)
#
#     img = []
#     for i in range(len(data)-1):
#         # {"image_id": 1, "category_id": 1, "bbox": [933.6468, 394.9749, 244.1592, 555.5509], "score": 0.3457}
#         box = data[i]
#         id = box['image_id']
#
#         image = filter(lambda image: image['id'] == id, images)[0]
#         boxes = filter(lambda anotation:anotation['id'] == id, annotations)
#         path = "data/crowdHuman/val/Images/{}".format(image['im_name'])
#         if id != data[i + 1]['image_id'] or i == 0:
#             img = cv2.imread(path)
#         bbox = box['bbox']
#         cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), (0, 255, 0), 1);0
#         for bx in boxes:
#             bbx = bx['bbox']
#             cv2.rectangle(img, (int(bbx[0]), int(bbx[1])), (int(bbx[0] + bbx[2]), int(bbx[1] + bbx[3])), (0, 0, 255), 1);
#
#         if id != data[i + 1]['image_id']:
#             cv2.imwrite('output/images/{}'.format(image['im_name']),img)
