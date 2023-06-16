# torch and visualization
from tqdm import tqdm
import torch.optim as optim
from torch.optim import lr_scheduler

from torch.utils.data import DataLoader
from model.parse_args_train import parse_args
import torch.nn.functional as F

# metric, loss .etc
from model.utils import *
from model.metric import *
from model.loss import *

# model
from model.Sobel import *
from model.model_ISNet import ISNet

"""
This class is responsible for the train and validation operations of the ISNet model.

To do this, first the necessary IDs belonging to the train and validation sets were read from txt files, 
and then dataloaders were created.

Afterwards, the model was created and hyperparameters were set. The data inputs were provided to the model, 
and the training and validation stages were carried out.

Methods:

training(self, epoch): Performs the training operation for the ISNet model.
validate(self, epoch): Performs the validation operation for the ISNet model.
"""
class Trainer(object):
    def __init__(self, args):
        # Initialize arguments
        self.args = args

        # metric, loss .etc
        self.ROC = ROCMetric(1, 10)
        self.mIoU = mIoU(1)
        self.isnetLoss = ISNetLoss()

        # Directories
        self.save_prefix = '_'.join(["ISNet", args.dataset])
        self.save_dir = args.save_dir

        nb_filter = load_param(args.channel_size)

        # Load img_ids
        base_dataset_dir = "dataset"
        dataset_dir = base_dataset_dir + '/' + args.dataset
        train_img_ids, val_img_ids, test_img_ids = load_dataset(args.root, args.dataset, args.split_method)

        # DataLoader
        training_set = TrainSetLoader(dataset_dir, img_id=train_img_ids, base_size=args.base_size,
                                                    crop_size=args.crop_size,  suffix=args.suffix)

        validation_set = TestSetLoader(dataset_dir, img_id=val_img_ids, base_size=args.base_size,
                                               crop_size=args.crop_size, suffix=args.suffix)

        self.train_data = DataLoader(dataset=training_set, batch_size=args.train_batch_size, shuffle=True,
                                     num_workers=args.workers, drop_last=True)

        self. validation_data = DataLoader(dataset=validation_set, batch_size=args.test_batch_size,
                                           num_workers=args.workers, drop_last=False)

        # Sobel
        self.sobel3Channel = Find3ChannelEdgeCoarseEdge()
        self.sobel1Channel = Find1ChannelEdgeCoarseEdge()

        # Model Initialization
        model = ISNet(layer_blocks=[4] * 3, channels=[8, 16, 32, 64])

        model = model.cuda()

        print("ISNet Model Initializing")
        self.model = model

        # Optimizer and lr scheduling
        if args.optimizer == 'Adagrad':
            self.optimizer = torch.optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()),
                                                 lr=args.lr, weight_decay=args.weightDecay)

        # if args.scheduler == 'CosineAnnealingLR':
        #     self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.epochs, eta_min=args.min_lr)
        # self.scheduler.step()

        # Evaluation metrics
        self.best_iou = 0
        self.best_recall = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.best_precision = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # Training
    def training(self, epoch):
        tbar = tqdm(self.train_data)
        self.model.train()
        losses = AverageMeter()
        torch_sobel = Sobel()
        for i, (data, labels) in enumerate(tbar):
            data = data.cuda()
            labels = labels.cuda()
            edged = self.sobel3Channel(data)
            edged_groundTruth = self.sobel1Channel(labels)
            pred = self.model(data,edged.cuda())
            loss = self.isnetLoss(pred, edged_groundTruth, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.update(loss.item())
            tbar.set_description('Epoch %d, training loss %.4f' % (epoch, losses.avg))
        self.train_loss = losses.avg

    # Validating
    def validating(self, epoch):
        tbar = tqdm(self.validation_data)
        self.model.eval()
        self.mIoU.reset()
        losses = AverageMeter()

        with torch.no_grad():
            for i, (data, labels) in enumerate(tbar):
                data = data.cuda()
                labels = labels.cuda()
                edged = self.sobel3Channel(data)
                edged_groundTruth = self.sobel1Channel(labels)
                pred = self.model(data,edged.cuda())
                loss = self.isnetLoss(pred, edged_groundTruth, labels)
                losses.update(loss.item())
                self.ROC.update(pred[0], labels)
                self.mIoU.update(pred[0], labels)
                true_positive_rate, false_positive_rate, recall, precision = self.ROC.get()
                _, mean_IOU = self.mIoU.get()
                tbar.set_description('Epoch %d, validation loss %.4f, mean_IoU: %.4f' % (epoch, losses.avg, mean_IOU))
            validation_loss = losses.avg
        # save high-performance model
        save_model(mean_IOU, self.best_iou, self.save_dir, self.save_prefix,
                   self.train_loss, validation_loss, recall, precision, epoch, self.model.state_dict())

def main(args):
    trainer = Trainer(args)
    for epoch in range(args.start_epoch, args.epochs):
        trainer.training(epoch)
        trainer.validating(epoch)


if __name__ == "__main__":
    args = parse_args()
    main(args)
