import os
import cv2
import cPickle
import numpy as np
import json


datapath = 'output/valresults/resnet50/2step/val_dt.json'
ground_truth_path = 'evaluation/val_gt.json'


images = []
data = []
for kk in range(0,3):
    if kk == 0:
        datapath = 'output/valresults/resnet50/2step/result_fog_02/val_dt.json'
    elif kk == 1:
        datapath = 'output/valresults/resnet50/2step/result_fog_005/val_dt.json'
    else:
        datapath = 'output/valresults/resnet50/2step/results_fog_01/val_dt.json'


    with open(datapath) as json_file:
        data = json.load(json_file)

    with open(ground_truth_path) as json_file:
        ground_truth = json.load(json_file)
        images = ground_truth['images']
        annotations = ground_truth['annotations']

        i = 0
        for image in images:
            i += 1
            if i == 10:
                break
            id = image['id']
            folder = image['im_name'].split("_")[0]
            impath = "data/cityperson/images/val/{}/{}".format(folder,image['im_name'])
            fogpath = "data/cityperson/images/val{}/{}/{}".format(kk+1,folder,image['im_name'])
            img = cv2.imread(impath)
            fimg = cv2.imread(fogpath)

            annotation = filter(lambda annotation:annotation['image_id'] == id, annotations)
            anno_pred = filter(lambda annotation: annotation['image_id'] == id, data)

            for box in annotation:
                ignore = box['ignore']
                if ignore == 1:
                    continue
                bbox = box['bbox']
                cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])),
                              (0, 0, 255), 4);


            cv2.imwrite('output/images_gt/val{}/{}'.format(kk+1,image['im_name']), img)

            img1 = fimg
            for box in anno_pred:
                bbox = box['bbox']
                cv2.rectangle(img1, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])),
                              (0, 255, 0), 3);

            cv2.imwrite('output/images_dt/val{}/{}'.format(kk+1,image['im_name']), img1)

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
