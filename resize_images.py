import cv2
from keras_alfnet import config
import os

C = config.Config()

desired_sizeH, desired_sizeW = C.random_crop

root_dir = 'data/crowdHuman/val'
all_img_path = os.path.join(root_dir, 'Images',)

for image in os.listdir(all_img_path):


    im_pth = os.path.join(all_img_path, image)

    im = cv2.imread(im_pth)
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

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    cv2.imwrite(im_pth, new_im)
    cv2.imwrite('output/images/{}'.format(image['im_name']), new_im)