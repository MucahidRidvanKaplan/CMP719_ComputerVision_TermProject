from PIL import Image, ImageOps, ImageFilter
import platform, os
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import random
import numpy as np
import torch
from torch.nn import init
from datetime import datetime
import argparse
import shutil
from matplotlib import pyplot as plt


"""
This class is responsible for file reading, file writing, dataloader management, saving results, and visualization 
in the project.
"""


def load_dataset (root, dataset, split_method):
    train_txt = root + '/' + dataset + '/' + split_method + '/' + 'train.txt'
    val_txt = root + '/' + dataset + '/' + split_method + '/' + 'val.txt'
    test_txt = root + '/' + dataset + '/' + split_method + '/' + 'test.txt'
    train_img_ids = []
    val_img_ids = []
    test_img_ids = []
    with open(train_txt, "r") as f:
        line = f.readline()
        while line:
            train_img_ids.append(line.split('\n')[0])
            line = f.readline()
        f.close()
    with open(val_txt, "r") as f:
        line = f.readline()
        while line:
            val_img_ids.append(line.split('\n')[0])
            line = f.readline()
        f.close()
    with open(test_txt, "r") as f:
        line = f.readline()
        while line:
            test_img_ids.append(line.split('\n')[0])
            line = f.readline()
        f.close()
    return train_img_ids, val_img_ids, test_img_ids


def get_test_txt(root, dataset, split_method):
    test_txt = root + '/' + dataset + '/' + split_method + '/' + 'test.txt'
    return test_txt


def load_param(channel_size):
    if channel_size == 'one':
        nb_filter = [4, 8, 16, 32, 64]
    elif channel_size == 'two':
        nb_filter = [8, 16, 32, 64, 128]
    elif channel_size == 'three':
        nb_filter = [16, 32, 64, 128, 256]
    elif channel_size == 'four':
        nb_filter = [32, 64, 128, 256, 512]
    return nb_filter


class TrainSetLoader(Dataset):

    def __init__(self, dataset_dir, img_id ,base_size=512,crop_size=480,suffix='.png'):
        super(TrainSetLoader, self).__init__()

        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])])

        self.transform = input_transform
        self._items = img_id
        self.masks = dataset_dir+'/'+'masks'
        self.images = dataset_dir+'/'+'images'
        self.base_size = base_size
        self.crop_size = crop_size
        self.suffix = suffix

    def _sync_transform(self, img, mask):
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        # random scale (short edge)
        long_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        # gaussian blur as in PSP
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
        # final transform
        img, mask = np.array(img), np.array(mask, dtype=np.float32)
        return img, mask

    def __getitem__(self, idx):

        img_id     = self._items[idx]  # idx：('../SIRST', 'Misc_41')
        img_path   = self.images+'/'+img_id+self.suffix  
        label_path = self.masks +'/'+img_id+self.suffix

        img = Image.open(img_path).convert('RGB')         
        mask = Image.open(label_path)

        # synchronized transform
        img, mask = self._sync_transform(img, mask)

        # general resize, normalize and toTensor
        img = self.transform(img)
        mask = np.expand_dims(mask, axis=0).astype('float32')/ 255.0

        return img, torch.from_numpy(mask) #img_id[-1]

    def __len__(self):
        return len(self._items)


class TestSetLoader(Dataset):

    def __init__(self, dataset_dir, img_id,base_size=512,crop_size=480,suffix='.png'):
        super(TestSetLoader, self).__init__()
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])])

        self.transform = input_transform
        self._items    = img_id
        self.masks     = dataset_dir+'/'+'masks'
        self.images    = dataset_dir+'/'+'images'
        self.base_size = base_size
        self.crop_size = crop_size
        self.suffix    = suffix

    def _testval_sync_transform(self, img, mask):
        base_size = self.base_size
        img  = img.resize ((base_size, base_size), Image.BILINEAR)
        mask = mask.resize((base_size, base_size), Image.NEAREST)

        # final transform
        img, mask = np.array(img), np.array(mask, dtype=np.float32)
        return img, mask

    def __getitem__(self, idx):
        # print('idx:',idx)
        img_id = self._items[idx]  # idx：('../SIRST', 'Misc_70')
        img_path   = self.images+'/'+img_id+self.suffix
        label_path = self.masks +'/'+img_id+self.suffix
        img  = Image.open(img_path).convert('RGB')
        mask = Image.open(label_path)
        # synchronized transform
        img, mask = self._testval_sync_transform(img, mask)

        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        mask = np.expand_dims(mask, axis=0).astype('float32') / 255.0

        return img, torch.from_numpy(mask)  # img_id[-1]

    def __len__(self):
        return len(self._items)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_ckpt(state, save_path, filename):
    torch.save(state, os.path.join(save_path, filename))


def save_train_log(args, save_dir):
    dict_args = vars(args)
    args_key = list(dict_args.keys())
    args_value = list(dict_args.values())
    with open('result/%s/train_log.txt' % save_dir, 'w') as f:
        now = datetime.now()
        f.write("time:--")
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        f.write(dt_string)
        f.write('\n')
        for i in range(len(args_key)):
            f.write(args_key[i])
            f.write(':--')
            f.write(str(args_value[i]))
            f.write('\n')
    return


def save_model_and_result(dt_string, epoch,train_loss, test_loss, best_iou, recall, precision, save_mIoU_dir, save_other_metric_dir):
    with open(save_mIoU_dir, 'a') as f:
        f.write('{} - {:04d}:\t - train_loss: {:04f}:\t - test_loss: {:04f}:\t mIoU {:.4f}\n' .format(dt_string, epoch,train_loss, test_loss, best_iou))
    with open(save_other_metric_dir, 'a') as f:
        f.write(dt_string)
        f.write('-')
        f.write(str(epoch))
        f.write('\n')
        f.write('Recall-----:')
        for i in range(len(recall)):
            f.write('   ')
            f.write(str(round(recall[i], 8)))
            f.write('   ')
        f.write('\n')

        f.write('Precision--:')
        for i in range(len(precision)):
            f.write('   ')
            f.write(str(round(precision[i], 8)))
            f.write('   ')
        f.write('\n')


def save_model(mean_IOU, best_iou, save_dir, save_prefix, train_loss, test_loss, recall, precision, epoch, net):
    if mean_IOU > best_iou:
        save_mIoU_dir = 'result/' + save_dir + '/' + save_prefix + '_best_IoU_IoU.log'
        save_other_metric_dir = 'result/' + save_dir + '/' + save_prefix + '_best_IoU_other_metric.log'
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        best_iou = mean_IOU
        save_model_and_result(dt_string, epoch, train_loss, test_loss, best_iou,
                              recall, precision, save_mIoU_dir, save_other_metric_dir)
        save_ckpt({
            'epoch': epoch,
            'state_dict': net,
            'loss': test_loss,
            'mean_IOU': mean_IOU,
        }, save_path='result/' + save_dir,
            filename='mIoU_' + '_' + save_prefix + '_epoch' + '.pth.tar')


def save_result_for_test(dataset_dir, st_model, epochs, best_iou, recall, precision):
    with open(dataset_dir + '/' + 'value_result' + '/' + st_model + '_best_IoU.log', 'a') as f:
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        f.write('{} - {:04d}:\t{:.4f}\n'.format(dt_string, epochs, best_iou))

    with open(dataset_dir + '/' + 'value_result' + '/' + st_model + '_best_other_metric.log', 'a') as f:
        f.write(dt_string)
        f.write('-')
        f.write(str(epochs))
        f.write('\n')
        f.write('Recall-----:')
        for i in range(len(recall)):
            f.write('   ')
            f.write(str(round(recall[i], 8)))
            f.write('   ')
        f.write('\n')

        f.write('Precision--:')
        for i in range(len(precision)):
            f.write('   ')
            f.write(str(round(precision[i], 8)))
            f.write('   ')
        f.write('\n')
    return


def make_dir(deep_supervision, dataset):
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    if deep_supervision:
        save_dir = "%s_%s_wDS" % (dataset, dt_string)
    else:
        save_dir = "%s_%s_woDS" % (dataset, dt_string)
    os.makedirs('result/%s' % save_dir, exist_ok=True)
    return save_dir


def make_visulization_dir(target_image_path):
    if os.path.exists(target_image_path):
        shutil.rmtree(target_image_path)
    os.mkdir(target_image_path)


def save_Pred_GT(pred, labels, target_image_path, val_img_ids, num, suffix):
    predsss = np.array((pred > 0).cpu()).astype('int64') * 255
    predsss = np.uint8(predsss)
    labelsss = labels * 255
    labelsss = np.uint8(labelsss.cpu())

    img = Image.fromarray(predsss.reshape(256, 256,4))
    img.save(target_image_path + '/' + '%s_Pred' % (val_img_ids[num]) + suffix)
    img = Image.fromarray(labelsss.reshape(256, 256,4))
    img.save(target_image_path + '/' + '%s_GT' % (val_img_ids[num]) + suffix)