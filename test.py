from __future__ import print_function

"""
before start testing
set my_train_mode in ./data/voc7012.py and ./test.py to False
"""

my_train_mode = False

import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from data import VOC_ROOT, VOC_CLASSES as labelmap
from PIL import Image
from data import VOCAnnotationTransform, VOCDetection, BaseTransform, VOC_CLASSES
import torch.utils.data as data
from ssd import build_ssd
import os.path as osp
from datachange import DataChange
from draw_rectangle import DrawRectangle
from time import time as tm
import time

dataname = '2019_5000m'

rootpath = 'E:/chLi/ssd.pytorch/data/' + dataname
source_json_path = osp.join(rootpath + '/source/Annotations')
source_image_path = osp.join(rootpath + '/source/JPEGImages')

need_to_change_data = True
use_my_labels = False
need_to_change_image_size = False
image_width = 426
image_height = 240

need_to_evaluate_data = True

need_to_draw_rectangle = True
rectangle_color = [0, 0, 255]
rectangle_width = 5

need_to_show_name = True
name_color = (0, 255, 0)

need_to_evaluate_result = True

if my_train_mode:
    need_to_change_image_size = True
    need_to_evaluate_data = False
    need_to_draw_rectangle = False
    need_to_show_name = False
    need_to_evaluate_result = False

model_list = os.listdir('E:/chLi/ssd.pytorch/weights')
resume_num = 0
for model_name in model_list:
    if 'COCO' in model_name:
        if int(model_name.split('.')[0].split('_')[2]) > resume_num:
            resume_num = int(model_name.split('.')[0].split('_')[2])
# resume_num = 0
if resume_num == 0:
    resume_name = 'E:/chLi/ssd.pytorch/weights/ssd300_mAP_77.43_v2.pth'
else:
    resume_name = 'E:/chLi/ssd.pytorch/weights/ssd300_COCO_%d.pth' % resume_num

net_name = resume_name.split('weights/')[1].split('.pth')[0]

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default=resume_name,
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default=rootpath, type=str,
                    help='Dir to save results')
parser.add_argument('--visual_threshold', default=0.6, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to train model')
parser.add_argument('--voc_root', default=VOC_ROOT, help='Location of VOC root directory')
parser.add_argument('-f', default=None, type=str, help="Dummy arg so we can load in Jupyter Notebooks")
args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def test_net(save_folder, net, cuda, testset, transform, thresh):
    time_pull_image = 0
    time_pull_anno = 0
    time_np_to_torch = 0
    time_to_cuda = 0
    time_net = 0
    time_write = 0
    # dump predictions and assoc. ground truth to text file for now
    filename = save_folder+'/result_data.txt'
    num_images = len(testset)
    # -------- Here --------
    time_now = tm()
    if not osp.exists(filename):
        f = open(filename, 'w')
        f.close()
    else:
        with open(filename, mode='r+') as f:
            f.truncate()
    time_now = tm() - time_now
    time_write += (time_now * 1000)
    # -------- Over --------
    for i in range(num_images):
        print('Testing image {:d}/{:d}....'.format(i+1, num_images))
        # -------- Here --------
        time_now = tm()
        img = testset.pull_image(i)
        time_now = tm() - time_now
        time_pull_image += (time_now * 1000)
        # -------- Over --------
        # -------- Here --------
        time_now = tm()
        img_id, annotation = testset.pull_anno(i)
        time_now = tm() - time_now
        time_pull_anno += (time_now * 1000)
        # -------- Over --------
        # -------- Here --------
        time_now = tm()
        x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
        time_now = tm() - time_now
        time_np_to_torch += (time_now * 1000)
        # -------- Over --------
        # -------- Here --------
        time_now = tm()
        x = Variable(x.unsqueeze(0))
        time_now = tm() - time_now
        time_to_cuda += (time_now * 1000)
        # -------- Over --------

        # -------- Here --------
        time_now = tm()
        with open(filename, mode='a') as f:
            f.write('\nGROUND TRUTH FOR: '+img_id+'\n')
            for box in annotation:
                f.write('label: '+' || '.join(str(b) for b in box)+'\n')
        time_now = tm() - time_now
        time_write += (time_now * 1000)
        # -------- Over --------
        # -------- Here --------
        time_now = tm()
        if cuda:
            x = x.cuda()
        time_now = tm() - time_now
        time_to_cuda += (time_now * 1000)
        # -------- Over --------
        # -------- Here --------
        time_now = tm()
        y = net(x)      # forward pass
        time_now = tm() - time_now
        time_net += (time_now * 1000)
        # -------- Over --------
        # -------- Here --------
        time_now = tm()
        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor([img.shape[1], img.shape[0],
                             img.shape[1], img.shape[0]])
        time_now = tm() - time_now
        time_to_cuda += (time_now * 1000)
        # -------- Over --------
        # -------- Here --------
        time_now = tm()
        pred_num = 0
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= 0.6:
                if pred_num == 0:
                    with open(filename, mode='a') as f:
                        f.write('PREDICTIONS: '+'\n')
                score = detections[0, i, j, 0]
                label_name = labelmap[i-1]
                pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
                coords = (pt[0], pt[1], pt[2], pt[3])
                pred_num += 1
                with open(filename, mode='a') as f:
                    f.write(str(pred_num)+' label: '+label_name+' score: ' +
                            str(score) + ' '+' || '.join(str(c) for c in coords) + '\n')
                j += 1
        time_now = tm() - time_now
        time_write += (time_now * 1000)
        # -------- Over --------
    print('-------------------------------------------')
    print('---- time_pull_image          : %.2fms' % time_pull_image)
    print('---- time_pull_anno           : %.2fms' % time_pull_anno)
    print('---- time_np_to_torch         : %.2fms' % time_np_to_torch)
    print('---- time_to_cuda             : %.2fms' % time_to_cuda)
    print('---- time_net                 : %.2fms' % time_net)
    print('---- time_write               : %.2fms' % time_write)
    print('-------------------------------------------')
    print('---- average_time_pull_image  : %.2fms' % (time_pull_image / num_images))
    print('---- average_time_pull_anno   : %.2fms' % (time_pull_anno / num_images))
    print('---- average_time_np_to_torch : %.2fms' % (time_np_to_torch / num_images))
    print('---- average_time_to_cuda     : %.2fms' % (time_to_cuda / num_images))
    print('---- average_time_net         : %.2fms' % (time_net / num_images))
    print('---- average_time_write       : %.2fms' % (time_write / num_images))
    print('-------------------------------------------')
    f = open(rootpath + '/output_msg.txt', 'a+')
    f.write('\n')
    f.write('===========================================\n')
    f.write('===========================================\n')
    f.write('===========================================\n\n')
    f.write('      ---- Date ----          : ' + time.asctime(time.localtime(tm())) + '\n\n')
    f.write('      ---- Net  ----          : ' + resume_name.split(rootpath.split('/data')[0])[1].split('/')[2] + '\n\n')
    f.write('-------------------------------------------\n')
    f.write('---- time_pull_image          : %.2fms\n' % time_pull_image)
    f.write('---- time_pull_anno           : %.2fms\n' % time_pull_anno)
    f.write('---- time_np_to_torch         : %.2fms\n' % time_np_to_torch)
    f.write('---- time_to_cuda             : %.2fms\n' % time_to_cuda)
    f.write('---- time_net                 : %.2fms\n' % time_net)
    f.write('---- time_write               : %.2fms\n' % time_write)
    f.write('-------------------------------------------\n')
    f.write('---- average_time_pull_image  : %.2fms\n' % (time_pull_image / num_images))
    f.write('---- average_time_pull_anno   : %.2fms\n' % (time_pull_anno / num_images))
    f.write('---- average_time_np_to_torch : %.2fms\n' % (time_np_to_torch / num_images))
    f.write('---- average_time_to_cuda     : %.2fms\n' % (time_to_cuda / num_images))
    f.write('---- average_time_net         : %.2fms\n' % (time_net / num_images))
    f.write('---- average_time_write       : %.2fms\n' % (time_write / num_images))
    f.write('-------------------------------------------\n')
    f.close()


def test_voc():
    time_total = 0
    # -------- Here --------
    time_now = tm()
    # load net
    print('Start loading model ...')
    num_classes = len(VOC_CLASSES) + 1 # +1 background
    net = build_ssd('test', 300, num_classes) # initialize SSD
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    print('Loaded model : ' + args.trained_model)
    print('Finished loading model!')
    # load data
    print('Start loading data ...')
    testset = VOCDetection(args.voc_root, [(rootpath, 'test')], None, VOCAnnotationTransform())
    print('Finished loading data!')
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    time_now = tm() - time_now
    time_total += (time_now * 1000)
    # -------- Over --------
    # evaluation
    print('Start evaluating data ...')
    test_net(args.save_folder, net, args.cuda, testset,
             BaseTransform(net.size, (104, 117, 123)),
             thresh=args.visual_threshold)
    print('---- time_load_data   : %.2fms' % time_total)
    print('Finished evaluating data!')

if __name__ == '__main__':

    total_time = tm()
    if need_to_change_data:
        print('Start changing source data ...')
        time1 = tm()
        DataChange(rootpath, source_json_path, source_image_path, use_my_labels, need_to_change_image_size, image_width, image_height)
        time1 = tm() - time1
        print('Finished changing source data!')

    if need_to_evaluate_data:
        time2 = tm()
        test_voc()
        time2 = tm() - time2

    if need_to_draw_rectangle or need_to_show_name or need_to_evaluate_result:
        print('Start drawing rectangle ...')
        time3 = tm()
        DrawRectangle(rootpath, net_name, None, None, None, None, need_to_draw_rectangle, rectangle_color, rectangle_width, need_to_show_name, name_color, use_my_labels, need_to_evaluate_result)
        time3 = tm() - time3
        print('Finished drawing rectangle!')

    total_time = tm() - total_time

    f = open(rootpath + '/output_msg.txt', 'a+')
    if my_train_mode:
        f.write('===========================================\n')
        f.write('===========================================\n')
        f.write('===========================================\n\n')
        f.write('      ---- Date ----          : ' + time.asctime(time.localtime(tm())) + '\n\n')
    f.write('-------------------------------------------\n')
    if need_to_change_data:
        print('Spending time on changing source data : %.2fms' % (time1 * 1000))
        f.write('Spending time on changing source data : %.2fms\n' % (time1 * 1000))
    if need_to_evaluate_data:
        print('Spending time on evaluating data : %.2fms' % (time2 * 1000))
        f.write('Spending time on evaluating data : %.2fms\n' % (time2 * 1000))
    if need_to_draw_rectangle and need_to_show_name:
        print('Spending time on drawing rectangle and showing name : %.2fms' % (time3 * 1000))
        f.write('Spending time on drawing rectangle and showing name : %.2fms\n' % (time3 * 1000))
    elif need_to_draw_rectangle:
        print('Spending time on drawing rectangle : %.2fms' % (time3 * 1000))
        f.write('Spending time on drawing rectangle : %.2fms\n' % (time3 * 1000))
    elif need_to_show_name:
        print('Spending time on showing name : %.2fms' % (time3 * 1000))
        f.write('Spending time on showing name : %.2fms\n' % (time3 * 1000))

    print('Spending time on total process : %.2fms' % (total_time * 1000))
    f.write('Spending time on total process : %.2fms\n' % (total_time * 1000))
    f.write('-------------------------------------------\n')
    f.close()