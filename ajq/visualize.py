import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from PIL import Image
import collections
import torch
from torchvision import transforms
import datetime
import numpy as np
import argparse
from tqdm import tqdm

from utils.logsetting import get_log
from utils.metrics import Metrics
import cv2

import warnings
warnings.filterwarnings('ignore')


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def predict(num_classes, net, files, save_path, device):
    trf = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.255])
    ])
    metrics = Metrics(range(num_classes))

    image_path1 = os.path.join(files, 'A')
    image_path2 = os.path.join(files, 'B')
    mask_path = os.path.join(files, 'label')

    save_dir = os.path.join(save_path, 'vis')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filenames = os.listdir(image_path1)

    for filename in tqdm(filenames):
        images1 = Image.open(os.path.join(image_path1, filename))
        images2 = Image.open(os.path.join(image_path2, filename))
        masks = Image.open(os.path.join(mask_path, filename))

        images1 = trf(images1).to(device)
        images2 = trf(images2).to(device)
        masks = trf(masks)
        masks = (masks > 0).type(torch.LongTensor).to(device)

        masks[masks < 0.01] = 0
        masks[masks >= 0.01] = 1

        images1 = images1.unsqueeze(0)
        images2 = images2.unsqueeze(0)
        # preds = net(images1, images2)
        preds, edge = net(images1, images2)

        for mask, output in zip(masks, preds):
            metrics.add(mask, output)

        preds = torch.argmax(preds, 1).cpu().detach().numpy()
        preds = preds * 255.0

        masks = masks.cpu().detach().numpy()
        tp = masks * preds
        fp = preds * (1 - masks)
        fn = masks * (1 - preds)

        one = torch.ones([1, 256, 256]) * 255.0
        zero = torch.zeros([1, 256, 256])

        c_blue = torch.cat([one, zero, zero], dim=0)
        # c_green = torch.cat([zero,one,zero],dim=0)
        c_red = torch.cat([zero, zero, one], dim=0)
        c_white = torch.cat([one, one, one], dim=0)

        images = c_red * fp + c_blue * fn + c_white * tp

        # vis = np.array(preds).transpose((1, 2, 0))
        vis = np.array(images).transpose((1, 2, 0))
        # vis = np.array(preds[0]).transpose((1, 2, 0))

        cv2.imwrite(os.path.join(save_dir, filename.split('.')[0] + '.png'), vis)

    return {
        "precision": metrics.get_precision(),
        "recall": metrics.get_recall(),
        "f_score": metrics.get_f_score(),
        "oa": metrics.get_oa(),
        "iou": metrics.get_miou()
    }


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', nargs='?', type=int, default=1,
                        help='Batch Size')
    arg = parser.parse_args()

    num_classes = 2
    img_size = 256
    batch_size = arg.batch_size

    history = collections.defaultdict(list)
    # test_datapath = './LEVIR-CD/test_s'
    # save_path = './LEVIR-CD/results'
    # test_datapath = './BCDD/test'
    # save_path = './BCDD/results'
    test_datapath = './SYSU-CD/test'
    save_path = './SYSU-CD/results'

    # net = torch.load("./checkpointsdpmSB_sgd_1e-2_bs4_LEVIR/best_model_epoch_71_9049.pth")
    # net = torch.load("./checkpointsatrousconvbranch_sgd_1e-2_bs4_LEVIR/best_model_epoch_65_9012.pth")
    # net = torch.load("./checkpointsdpmSB_DEpool_sgd_1e-2_bs4_LEVIR/0.9092425037940189_model_checkpoints.pth")
    # net = torch.load("./checkpointsdpmSB_crossattncat_sgd_1e-2_decay03_bs4_LEVIR/0.90916456362228_model_checkpoints.pth")
    # net = torch.load("./checkpointsdpmSB_DEpoolwithcrossattncat_sgd_1e-2_decay03_bs4_LEVIR/0.9093201048529292_model_checkpoints.pth")
    # net = torch.load("./checkpointsdpmSB_DEpoolwithcrossattncat_atrousconvbranch_sgd_1e-2_decay03_bs4_LEVIR/0.9125079684880728_model_checkpoints.pth")
    # net = torch.load("./checkpointstest_dpmSB_DEpoolwithcrossattncat_atrousconvwithedgebranch_sgd_1e-2_decay03_bs4_LEVIR/0.915323_best_model_epoch.pth")

    # net = torch.load("./checkpointstest_BIT_sgd_1e-2_decay03_bs4_LEVIR/0.8954229684292571_model_checkpoints.pth")
    # net = torch.load("./checkpointstest_SNUNet_sgd_1e-2_decay03_bs4_LEVIR/0.9090694677032202_model_checkpoints.pth")
    # net = torch.load("./checkpointstest_p2v_sgd_1e-2_decay03_bs4_LEVIR/0.9061162423246623_model_checkpoints.pth")
    # -----------------------------------------------------------------------------------------------------------------------------------------------------
    # net = torch.load("./checkpointsdpmSB_sgd_1e-2_decay06_bs4_BCDD/best_model_epoch_68_8971.pth")
    # net = torch.load("./checkpointsatrousconvbranch_sgd_1e-2_decay06_bs4_BCDD/best_model_epoch_57_8847.pth")
    # net = torch.load("./checkpointsdpmSB_DEpool_sgd_1e-2_decay06_bs4_BCDD/0.9097351407726062_model_checkpoints.pth")
    # net = torch.load("./checkpointsdpmSB_crossattncat_sgd_1e-2_decay06_bs4_BCDD/0.9092359174254884_model_checkpoints.pth")
    # net = torch.load("./checkpointsdpmSB_DEpoolwithcrossattncat_sgd_1e-2_decay06_bs4_BCDD/0.9208605694611395_model_checkpoints.pth")
    # net = torch.load("./checkpointsdpmSB_DEpoolwithcrossattncat_atrousconvbranch_sgd_1e-2_decay06_bs4_BCDD/0.9267688721178805_model_checkpoints.pth")
    # net = torch.load("./checkpointstest_dpmSB_DEpoolwithcrossattncat_atrousconvwithedgebranch_sgd_1e-2_decay06_bs4_BCDD/0.9321157586199532_model_checkpoints.pth")

    # net = torch.load("./checkpointstest_BIT_sgd_1e-2_decay06_bs4_BCDD/0.9063118139812076_model_checkpoints.pth")
    # net = torch.load("./checkpointstest_SNUNet_sgd_1e-2_decay06_bs4_BCDD/0.9200148497716618_model_checkpoints.pth")
    # net = torch.load("./checkpointstest_p2v_sgd_1e-2_decay06_bs4_BCDD/0.9207018814009115_model_checkpoints.pth")
    # -----------------------------------------------------------------------------------------------------------------------------------------------------
    # net = torch.load("./checkpointsdpmSB_sgd_1e-2_bs4_SYSU/best_model_epoch_57_7777.pth")
    # net = torch.load("./checkpointsatrousconvbranch_sgd_1e-2_bs4_SYSU/best_model_epoch_43_7588.pth")
    # net = torch.load("./checkpointsdpmSB_DEpool_sgd_1e-2_bs4_SYSU/0.766404291139197_model_checkpoints.pth")
    # net = torch.load("./checkpointsdpmSB_crossattncat_sgd_1e-2_bs4_SYSU/best_model_epoch_32_7645.pth")
    # net = torch.load("./checkpointsdpmSB_DEpoolwithcrossattncat_sgd_1e-2_bs4_SYSU/0.7699358427752393_model_checkpoints.pth")
    # net = torch.load("./checkpointsdpmSB_DEpoolwithcrossattncat_atrousconvbranch_sgd_1e-2_bs4_SYSU/0.7836330327330825_model_checkpoints.pth")
    net = torch.load("./checkpointstest_dpmSB_DEpoolwithcrossattncat_atrousconvwithedgebranch_sgd_1e-2_bs4_SYSU/0.8157853973299531_model_checkpoints.pth")

    # net = torch.load("./checkpointstest_BIT_sgd_1e-2_bs4_SYSU/0.7798897980596281_model_checkpoints.pth")
    # net = torch.load("./checkpointstest_SNUNet_sgd_1e-2_bs4_SYSU/0.8089086578597812_model_checkpoints.pth")
    # net = torch.load("./checkpointstest_p2v_sgd_1e-2_bs4_SYSU/0.7942319238039527_model_checkpoints.pth")

    net.eval()

    if not os.path.exists('./result'):
        os.makedirs('./result')
    today = str(datetime.date.today())

    # logger = get_log("checkpointsdpmSB_sgd_1e-2_bs4-LEVIR" + '_test_log.txt')
    # logger = get_log("checkpointsatrousconvbranch_sgd_1e-2_bs4-LEVIR" + '_test_log.txt')
    # logger = get_log("checkpointsdpmSB_DEpool_sgd_1e-2_bs4-LEVIR" + '_test_log.txt')
    # logger = get_log("checkpointsdpmSB_crossattncat_sgd_1e-2_decay03_bs4-LEVIR" + '_test_log.txt')
    # logger = get_log("checkpointsdpmSB_DEpoolwithcrossattncat_sgd_1e-2_decay03_bs4-LEVIR" + '_test_log.txt')
    # logger = get_log("checkpointsdpmSB_DEpoolwithcrossattncat_atrousconvbranch_sgd_1e-2_decay03_bs4-LEVIR" + '_test_log.txt')
    # logger = get_log("checkpointsdpmSB_DEpoolwithcrossattncat_atrousconvwithedgebranch_sgd_1e-2_decay03_bs4-LEVIR" + '_test_log.txt')
    # logger = get_log("checkpointstest_dpmSB_DEpoolwithcrossattncat_atrousconvwithedgebranch_sgd_1e-2_decay03_bs4-LEVIR" + '_test_log.txt')

    # logger = get_log("checkpointstest_BIT_sgd_1e-2_decay03_bs4-LEVIR" + '_test_log.txt')
    # logger = get_log("checkpointstest_SNUNet_sgd_1e-2_decay03_bs4-LEVIR" + '_test_log.txt')
    # logger = get_log("checkpointstest_p2v_sgd_1e-2_decay03_bs4-LEVIR" + '_test_log.txt')
    # -----------------------------------------------------------------------------------------------------------------------------------------------------
    # logger = get_log("checkpointsdpmSB_sgd_1e-2_bs4-BCDD" + '_test_log.txt')
    # logger = get_log("checkpointsatrousconvbranch_sgd_1e-2_bs4-BCDD" + '_test_log.txt')
    # logger = get_log("checkpointsdpmSB_DEpool_sgd_1e-2_decay06_bs4-BCDD" + '_test_log.txt')
    # logger = get_log("checkpointsdpmSB_crossattncat_sgd_1e-2_decay06_bs4-BCDD" + '_test_log.txt')
    # logger = get_log("checkpointsdpmSB_DEpoolwithcrossattncat_sgd_decay06_1e-2_bs4-BCDD" + '_test_log.txt')
    # logger = get_log("checkpointsdpmSB_DEpoolwithcrossattncat_atrousconvbranch_sgd_1e-2_decay06_bs4-BCDD" + '_test_log.txt')
    # logger = get_log("checkpointstest_dpmSB_DEpoolwithcrossattncat_atrousconvwithedgebranch_sgd_1e-2_decay06_bs4-BCDD" + '_test_log.txt')

    # logger = get_log("checkpointstest_BIT_sgd_1e-2_decay06_bs4-BCDD" + '_test_log.txt')
    # logger = get_log("checkpointstest_SNUNet_sgd_1e-2_decay06_bs4-BCDD" + '_test_log.txt')
    # logger = get_log("checkpointstest_p2v_sgd_1e-2_decay06_bs4-BCDD" + '_test_log.txt')
    # -----------------------------------------------------------------------------------------------------------------------------------------------------
    # logger = get_log("checkpointsdpmSB_sgd_1e-2_bs4-SYSU" + '_test_log.txt')
    # logger = get_log("checkpointsatrousconvbranch_sgd_1e-2_bs4-SYSU" + '_test_log.txt')
    # logger = get_log("checkpointsdpmSB_DEpool_sgd_1e-2_bs4-SYSU" + '_test_log.txt')
    # logger = get_log("checkpointsdpmSB_crossattncat_sgd_1e-2_bs4-SYSU" + '_test_log.txt')
    # logger = get_log("checkpointsdpmSB_DEpoolwithcrossattncat_sgd_1e-2_bs4-SYSU" + '_test_log.txt')
    # logger = get_log("checkpointsdpmSB_DEpoolwithcrossattncat_atrousconvbranch_sgd_1e-2_bs4-SYSU" + '_test_log.txt')
    logger = get_log("checkpointstest_dpmSB_DEpoolwithcrossattncat_atrousconvwithedgebranch_sgd_1e-2_bs4-SYSU" + '_test_log.txt')

    # logger = get_log("checkpointstest_BIT_sgd_1e-2_bs4-SYSU" + '_test_log.txt')
    # logger = get_log("checkpointstest_SNUNet_sgd_1e-2_bs4-SYSU" + '_test_log.txt')
    # logger = get_log("checkpointstest_p2v_sgd_1e-2_bs4-SYSU" + '_test_log.txt')


    test_hist = predict(num_classes, net, test_datapath, save_path, device)
    print('precision={}'.format(test_hist["precision"]),
          'recall={}'.format(test_hist["recall"]),
          'f_score={}'.format(test_hist["f_score"]),
          'oa={}'.format(test_hist["oa"]),
          'iou={}'.format(test_hist["iou"]))

    logger.info(('precision={}'.format(test_hist["precision"]),
                 'recall={}'.format(test_hist["recall"]),
                 'f_score={}'.format(test_hist["f_score"]),
                 'oa={}'.format(test_hist["oa"]),
                 'iou={}'.format(test_hist["iou"])))

    for k, v in test_hist.items():
        history["test " + k].append(v)