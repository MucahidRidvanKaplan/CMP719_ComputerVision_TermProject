# Basic module
from tqdm import tqdm
from model.parse_args_test import parse_args
import scipy.io as scio

# Torch and visualization
from torchvision import transforms
from torch.utils.data import DataLoader

# Metric, loss .etc
from model.utils import *
from model.metric import *
from model.loss import *

# model
from model.Sobel import *
from model.model_ISNet import ISNet

"""
This class is responsible for testing and visualizing the results of the trained ISNet model.
"""


class Tester(object):
    def __init__(self, args):
        # Initialize arguments
        self.args  = args

        # metric, loss .etc
        self.PD_FA = PD_FA(1,10)
        self.mIoU  = mIoU(1)
        self.isnetLoss = ISNetLoss()

        # Directories
        self.save_prefix = '_'.join(["ISNet", args.dataset])
        nb_filter, num_blocks = load_param(args.channel_size)

        # Load img_ids
        dataset_dir = args.root + '/' + args.dataset
        train_img_ids, val_img_ids, test_img_ids = load_dataset(args.root, args.dataset, args.split_method)
        test_txt  = get_test_txt(args.root, args.dataset, args.split_method)

        testset         = TestSetLoader (dataset_dir, img_id=test_img_ids, base_size=args.base_size, crop_size=args.crop_size,suffix=args.suffix)
        self.test_data  = DataLoader(dataset=testset,  batch_size=args.test_batch_size, num_workers=args.workers,drop_last=False)

        # Sobel
        self.sobel3Channel = Find3ChannelEdgeCoarseEdge()
        self.sobel1Channel = Find1ChannelEdgeCoarseEdge()

        # Model Initialization
        model = ISNet(layer_blocks=[4] * 3, channels=[8, 16, 32, 64])

        model           = model.cuda()

        #model.apply(weights_init_xavier)
        print("Model Testing")
        self.model      = model

        # Evaluation metrics
        self.best_recall    = [0,0,0,0,0,0,0,0,0,0,0]
        self.best_precision = [0,0,0,0,0,0,0,0,0,0,0]

        # Checkpoint
        checkpoint        = torch.load("mIoU__ISNet_NUAA-SIRST_epoch.pth.tar")

        #checkpoint        = torch.load("mIoU__ISNet_NUAA-SIRST_epoch_MertPCyaridaKalan.pth.tar")

        target_image_path = "./visual_result_"+args.dataset

        make_visulization_dir(target_image_path)

        # Load trained model
        self.model.load_state_dict(checkpoint['state_dict'])

        # Test
        self.model.eval()
        tbar = tqdm(self.test_data)
        losses = AverageMeter()
        torch_sobel = Sobel()
        with torch.no_grad():
            num = 0
            for i, ( data, labels) in enumerate(tbar):
                data = data.cuda()
                labels = labels.cuda()
                edged = self.sobel3Channel(data)
                edged_groundTruth = self.sobel1Channel(labels)
                pred = self.model(data,edged.cuda())
                loss = self.isnetLoss(pred, edged_groundTruth, labels)
                save_Pred_GT(pred[0], labels,target_image_path, test_img_ids, num, args.suffix)
                num += 1

                losses.    update(loss.item())
                self.ROC.  update(pred[0], labels)
                self.mIoU. update(pred[0], labels)
                self.PD_FA.update(pred[0], labels)

                true_positive_rate, false_positive_rate, recall, precision= self.ROC.get()
                _, mean_IOU = self.mIoU.get()
            FA, PD = self.PD_FA.get(len(val_img_ids))
            test_loss = losses.avg
            scio.savemat(dataset_dir + '\\' +  'value_result'+ '\\' +args.st_model  + '_PD_FA_' + str(255),
                         {'number_record1': FA, 'number_record2': PD})

            print('test_loss, %.4f' % (test_loss))
            print('mean_IOU:', mean_IOU)
            self.best_iou = mean_IOU
            save_result_for_test(dataset_dir, args.st_model,args.epochs, self.best_iou, recall, precision)

            source_image_path = dataset_dir + '\\images'

            for i in range(len(test_img_ids)):
                source_image = source_image_path + '\\' + test_img_ids[i] + args.suffix
                target_image = target_image_path + '\\' + test_img_ids[i] + args.suffix
                shutil.copy(source_image, target_image)
            for i in range(len(test_img_ids)):
                source_image = target_image_path + '\\' + test_img_ids[i] + args.suffix
                img = Image.open(source_image)
                img = img.resize((256, 256), Image.ANTIALIAS)
                img.save(source_image)
            print("DONE")

def main(args):
    tester = Tester(args)

if __name__ == "__main__":
    args = parse_args()
    main(args)





