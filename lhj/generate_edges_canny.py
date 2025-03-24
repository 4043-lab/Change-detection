import cv2
import glob
import os
from tqdm import tqdm

# type = 'train'
# #source path
# image_path = 'dataset/Earth_Paris_new-256/{}/'.format(type)
# image_list = glob.glob(image_path + 'label/*.png')
# #target path
# dir_name = 'dataset/Earth_Paris_new-256/{}/edge/'.format(type)
#
# for i in tqdm(range(len(image_list))):
#     img = cv2.imread(image_list[i])
#     canny = cv2.Canny(img, 50, 150)
#     basename = os.path.basename(image_list[i])
#     if not os.path.exists(dir_name):
#         os.makedirs(dir_name)
#     cv2.imwrite(dir_name + basename, canny)

# type = 'val'
# #source path
# image_path = 'dataset/dataset_test/{}/'.format(type)
# image_list = glob.glob(image_path + 'label/*.png')
# #target path
# dir_name = 'dataset/dataset_test/{}/edge/'.format(type)
#
# for i in range(len(image_list)):
#     img = cv2.imread(image_list[i])
#     canny = cv2.Canny(img, 50, 150)
#     basename = os.path.basename(image_list[i])
#     if not os.path.exists(dir_name):
#         os.makedirs(dir_name)
#     cv2.imwrite(dir_name + basename, canny)

#source path
# image_path = 'dataset/LEVIR-CD-256-1/train/label/'
# image_path = 'dataset/BCDD/test/label/'
image_path = 'dataset/SYSU-CD/test/label/'
image_list = glob.glob(image_path + '*.png')
# print(image_list)
#target path
# dir_name = 'dataset/LEVIR-CD-256-1/train/edge/'
# dir_name = 'dataset/BCDD/test/edge/'
dir_name = 'dataset/SYSU-CD/test/edge/'

for i in tqdm(range(len(image_list))):
    img = cv2.imread(image_list[i])
    canny = cv2.Canny(img, 50, 150)
    basename = os.path.basename(image_list[i])
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    cv2.imwrite(dir_name + basename, canny)