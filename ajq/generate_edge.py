import os

import cv2

image_path = './SYSU-CD/test/label'
img_list = os.listdir(image_path)
dir_name = './SYSU-CD/test/edge/'

for name in img_list:
    img = cv2.imread(os.path.join(image_path, name))
    canny = cv2.Canny(img, 50, 150)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    cv2.imwrite(dir_name + name, canny)
