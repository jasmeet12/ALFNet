import os
import json
import numpy as np
import cv2

root_dir = 'data/crowdHuman/val'
all_img_path = os.path.join(root_dir, 'test')

def readFile(p,type):
    with open(p, 'r+') as f:
        datalist = f.readlines()
        data = []
    for i in np.arange(len(datalist)):

        adata = json.loads(datalist[i])
        gtboxes = adata['gtboxes']
        inputfile = {}
        imgname = adata['ID']

        # img = cv2.imread('data/crowdHuman/{}/Images/{}.jpg'.format(type,imgname))

        im_pth = os.path.join(all_img_path, imgname)

        img = cv2.imread("{}.jpg".format(im_pth))
        if img is None:
            continue

        inputfile['imgHeight'] = 1024
        inputfile['imgWidth'] = 2048
        inputfile['objects'] = []
        count = 24000
        gts = []

        for gtbox in gtboxes:
            x1 = gtbox['fbox'][0]
            y1 = gtbox['fbox'][1]
            w = gtbox['fbox'][2]
            h = gtbox['fbox'][3]

            x1_vis = gtbox['vbox'][0]
            y1_vis = gtbox['vbox'][1]
            w_vis = gtbox['vbox'][2]
            h_vis = gtbox['vbox'][3]


            if gtbox['tag'] == 'person':

                gts.append([1, x1, y1, w, h, count, x1_vis, y1_vis, w_vis, h_vis],)

                count += 1
            elif gtbox['tag'] == 'mask':
                gts.append([4, x1, y1, w, h, 0, x1_vis, y1_vis, w_vis, h_vis],)

        obj = {'cityname':'random','imgname':'{}.jpg'.format(imgname),'bbs':gts,'imgH':1024,'imgW':2048}

        data.append(obj)
    return data





