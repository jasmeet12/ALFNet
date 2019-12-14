import os
import json
import numpy as np
import cv2
import pickle

out = {'categories':[{"id":1,"name":"pedestrian"},
               {"id":2,"name":"rider"},
               {"id":3,"name":"sitting person"},
               {"id":4,"name":"other person"},
               {"id":5,"name":"people group"},
               {"id":0,"name":"ignore region"}],
        'images':[],
        'annotations':[]}

res_path1 = 'data/crowdHuman/val/annotation_object'
res_path = os.path.join('data/cache/crowdHuman', 'val')
out_path = 'evaluation/val_crowd_gt.json'
annos = []
with open(res_path, 'rb') as config_dictionary_file:
    annos = pickle.load(config_dictionary_file)

count = 0
for l in range(len(annos)):
    anno = annos[l]
    img_info = {"id":l+1,
                "im_name":'{}'.format(anno['filepath'].split('/')[-1]),
                "height":1024,"width":2048}
    out['images'].append(img_info)
    gts = anno['bboxes']


    if len(gts) == 0:
        continue
    for i in range(len(gts)):
        count += 1
        # {"id": 1, "image_id": 1, "category_id": 1, "iscrowd": 0, "ignore": 0, "bbox": [947, 406, 17, 40],
        #  #                 "vis_bbox":[950,407,14,39],"height":40,"vis_ratio":0.802941176471},


        val = gts[i]
        vbox = anno['vis_bboxes'][i]
        bbox = gts[i]
        wv = vbox[2]-vbox[0]
        hv = vbox[3]-vbox[1]
        wb = bbox[2]-vbox[0]
        hb = bbox[3]-bbox[1]
        vboxx = [vbox[0],vbox[1],wv,hv]
        bboxx = [bbox[0],bbox[1],wb,hb]
        try:
            vis_ratio = float(wv*hv) / (wb*hb)
        except:
            vis_ratio = 0
            print(0)

        # if val[0] == 1:
        annotation = {'id':count, 'image_id':l+1,'category_id':1,"iscrowd":0,'ignore':0,"bbox": bboxx ,"vis_bbox":vboxx, 'height':bbox[-1],'vis_ratio':vis_ratio}
        # else:
        #     annotation = {'id': count, 'image_id': l+1, 'category_id': 1, "iscrowd": 0, 'ignore': 1, "bbox": bbox,
        #                   "vis_bbox": vbox, 'height': val[4], 'vis_ratio': vis_ratio}

        out['annotations'].append(annotation)



with open(out_path, 'w') as f:
    json.dump(out, f)

#
#
# {"categories":[{"id":1,"name":"pedestrian"},
#                {"id":2,"name":"rider"},
#                {"id":3,"name":"sitting person"},
#                {"id":4,"name":"other person"},
#                {"id":5,"name":"people group"},
#                {"id":0,"name":"ignore region"}],
#  "images":[{"id":1,"im_name":"frankfurt_000000_000294_leftImg8bit.png","height":1024,"width":2048},
#            {"id":2,"im_name":"frankfurt_000000_000576_leftImg8bit.png","height":1024,"width":2048},
#            {"id":3,"im_name":"frankfurt_000000_001016_leftImg8bit.png","height":1024,"width":2048},
#            {"id":4,"im_name":"frankfurt_000000_001236_leftImg8bit.png","height":1024,"width":2048},
#            {"id":5,"im_name":"frankfurt_000000_001751_leftImg8bit.png","height":1024,"width":2048},
#
# "annotations":[{"id":1,"image_id":1,"category_id":1,"iscrowd":0,"ignore":0,"bbox":[947,406,17,40],
#                 "vis_bbox":[950,407,14,39],"height":40,"vis_ratio":0.802941176471},
#                {"id":2,"image_id":1,"category_id":1,"iscrowd":0,"ignore":0,"bbox":[1157,375,41,99],"vis_bbox":[1158,375,40,99],"height":99,"vis_ratio":0.975609756098},
#                {"id":3,"image_id":1,"category_id":1,"iscrowd":0,"ignore":0,"bbox":[1195,381,35,84],"vis_bbox":[1198,381,29,84],"height":84,"vis_ratio":0.828571428571},
#                {"id":4,"image_id":1,"category_id":1,"iscrowd":0,"ignore":0,"bbox":[1223,379,38,91],"vis_bbox":[1223,379,35,91],"height":91,"vis_ratio":0.921052631579},
#                {"id":5,"image_id":1,"category_id":1,"iscrowd":0,"ignore":1,"bbox":[1839,382,31,71],"vis_bbox":[1839,382,31,71],"height":71,"vis_ratio":1},
#                {"id":6,"image_id":1,"category_id":1,"iscrowd":0,"ignore":1,"bbox":[1617,354,24,31],"vis_bbox":[1617,354,24,31],"height":31,"vis_ratio":1},
#                {"id":7,"image_id":1,"category_id":1,"iscrowd":0,"ignore":1,"bbox":[1940,455,30,54],"vis_bbox":[1940,455,30,54],"height":54,"vis_ratio":1},
