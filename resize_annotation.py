import cv2
from keras_alfnet import config
import os
from readodgtFile import readFile
import pickle

C = config.Config()

desired_sizeH, desired_sizeW = C.random_crop


root_dir = 'data/crowdHuman/val'
all_img_path = os.path.join(root_dir, 'test')
out = os.path.join(root_dir, 'tt')
anno_path = os.path.join(root_dir, 'annotation_val.odgt')
annos = readFile(anno_path,'val')
i = 0
for anno in annos:


    img = anno['imgname']

    im_pth = os.path.join(all_img_path, img)
    out_path = os.path.join(out, img)
    im = cv2.imread(im_pth)
    if im is None:
        continue
    old_size = im.shape[:2] # old_size is in (height, width) format

    # ratio = float(desired_sizeW)/max(old_size)
    # new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format
    #

    ratio = float(old_size[1]) / float(old_size[0])
    if old_size[0] > desired_sizeH:
        width = int(ratio * desired_sizeH)
        height = desired_sizeH
        if width > desired_sizeW:
            width = int(desired_sizeW*.9)
            height = int(width/ratio)
        im = cv2.resize(im, (width, height))

    elif old_size[1] > desired_sizeW:
        height = int(desired_sizeW/ratio)
        width = desired_sizeW
        if height > desired_sizeH:
            height = int(desired_sizeH*.9)
            width = int(height *ratio)
        im = cv2.resize(im, (width, height))


    # if (old_size[0] < old_size[1]):
    #     ratio = float(desired_sizeH) / max(old_size)
    # else:
    #     ratio = float(desired_sizeW) / max(old_size)
    # new_size = tuple([int(x * ratio) for x in old_size])

    new_size = im.shape[:2]
    delta_w = desired_sizeW - new_size[1]
    delta_h = desired_sizeH - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    # im = cv2.resize(im, (new_size[1], new_size[0]))
    gts = anno['bbs']
    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    i += 1

    w_r = float(float(new_size[1]) / float(old_size[1]))
    h_r = float(float(new_size[0]) / float(old_size[0]))

    for j in range(len(gts)):

        label, x1, y1, w, h = gts[j][:5]
        w_n = int(w_r*w)
        h_n = int(h_r*h)

        x1_n = int(x1*w_r)
        y1_n = int(y1*h_r)

        x1_n = x1_n + left
        y1_n = y1_n + top
        gts[j][:5] = [label, x1_n, y1_n, w_n, h_n]


        xv1, yv1, wv, hv = gts[j][6:]

        wv_n = int(w_r * wv)
        hv_n = int(h_r * hv)


        xv1_n = int(xv1*w_r)
        yv1_n = int(yv1*h_r)

        xv1_n = xv1_n + left
        yv1_n = yv1_n + top

        gts[j][6:] = [xv1_n, yv1_n, wv_n, hv_n]
        cv2.rectangle(new_im, (xv1_n, yv1_n), (xv1_n + wv_n, yv1_n + hv_n), (0, 255, 0), 3);

    cv2.imwrite(out_path, new_im)
    print(out_path)
    # cv2.imwrite('output/images/{}'.format(img), new_im)

# res_path = 'data/crowdHuman/val/annotation_object'
# with open(res_path, 'wb') as config_dictionary_file:
#     pickle.dump(annos, config_dictionary_file)
#
# with open(res_path, 'rb') as config_dictionary_file:
#     config_dictionary = pickle.load(config_dictionary_file)
#
#     print(config_dictionary)



