"""SegmentationNN"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision import models

class SegmentationNN(pl.LightningModule):

    def __init__(self, num_classes=23, hparams=None):
        super().__init__()
        self.save_hyperparameters(hparams)
        ########################################################################
        # TODO - Train Your Model                                              #
        ########################################################################

        self.encoder = models.mobilenet_v2(pretrained=True).features

        for parma in self.encoder.parameters():
            parma.requires_grad = False
            
        self.decoder = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        nn.Conv2d(1280, 320, kernel_size=(1, 1), stride=(1, 1), bias=False),
        nn.Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        nn.BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU6(inplace=True),

        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        nn.Conv2d(320, 160, kernel_size=(1, 1), stride=(1, 1), bias=False),
        nn.Conv2d(160, 160, kernel_size=(3, 3), stride=(1, 1), bias=False),
        nn.BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU6(inplace=True),

        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        nn.Conv2d(160, 80, kernel_size=(1, 1), stride=(1, 1), bias=False),
        nn.Conv2d(80, 80, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        nn.BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU6(inplace=True),
                
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        nn.Conv2d(80, 40, kernel_size=(1, 1), stride=(1, 1), bias=False),
        nn.Conv2d(40, 40, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        nn.BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU6(inplace=True),

        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        nn.Conv2d(40, 23, kernel_size=(1, 1), stride=(1, 1), bias=False),
        nn.Conv2d(23, 23, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        nn.BatchNorm2d(23, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU6(inplace=True)
        )

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################

        x = self.encoder(x)
        x = self.decoder(x)

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return x

    def general_step(self, batch, batch_idx, mode):
        
        # def _to_one_hot(y, num_classes):
        #     scatter_dim = len(y.size())
        #     y_tensor = y.view(*y.size(), -1)
        #     zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

        #     return zeros.scatter(scatter_dim, y_tensor, 1)

        images, targets = batch
    
        # targets[targets == -1] = 1
        # targets = _to_one_hot(targets, 23).permute(0, 3, 1 ,2)

        # forward pass
        outputs = self.forward(images)

        # loss
        loss_function = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
        loss = loss_function(outputs, targets)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.general_step(batch, batch_idx, "train")
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.general_step(batch, batch_idx, "val")
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.decoder.parameters(), lr=self.hparams["learning_rate"],
                                                                    weight_decay=self.hparams["weight_decay"])
        StepLR = {'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optim, mode='min', 
                                                                    factor=self.hparams["lr_decay"], patience=2), 'monitor': 'train_loss'}
        return {"optimizer": optim, "lr_scheduler": StepLR}

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

        
class DummySegmentationModel(pl.LightningModule):

    def __init__(self, target_image):
        super().__init__()
        def _to_one_hot(y, num_classes):
            scatter_dim = len(y.size())
            y_tensor = y.view(*y.size(), -1)
            zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

            return zeros.scatter(scatter_dim, y_tensor, 1)

        target_image[target_image == -1] = 1

        self.prediction = _to_one_hot(target_image, 23).permute(2, 0, 1).unsqueeze(0)

    def forward(self, x):
        return self.prediction.float()
