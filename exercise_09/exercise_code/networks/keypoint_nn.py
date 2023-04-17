"""Models for facial keypoint detection"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F

# TODO: Choose from either model and uncomment that line
# class KeypointModel(nn.Module):
class KeypointModel(pl.LightningModule):
    """Facial keypoint detection model"""
    def __init__(self, hparams):
        """
        Initialize your model from a given dict containing all your hparams
        Warning: Don't change the method declaration (i.e. by adding more
            arguments), otherwise it might not work on the submission server
            
        NOTE: You could either choose between pytorch or pytorch lightning, 
            by switching the class name line.
        """
        super().__init__()
        self.save_hyperparameters(hparams)
        ########################################################################
        # TODO: Define all the layers of your CNN, the only requirements are:  #
        # 1. The network takes in a batch of images of shape (Nx1x96x96)       #
        # 2. It ends with a linear layer that represents the keypoints.        #
        # Thus, the output layer needs to have shape (Nx30),                   #
        # with 2 values representing each of the 15 keypoint (x, y) pairs      #
        #                                                                      #
        # Some layers you might consider including:                            #
        # maxpooling layers, multiple conv layers, fully-connected layers,     #
        # and other layers (such as dropout or batch normalization) to avoid   #
        # overfitting.                                                         #
        #                                                                      #
        # We would truly recommend to make your code generic, such as you      #
        # automate the calculation of the number of parameters at each layer.  #
        # You're going probably try different architecutres, and that will     #
        # allow you to be quick and flexible.                                  #
        ########################################################################
        
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, (4, 4), stride=1, padding=0),
            nn.ELU(),
            nn.MaxPool2d((2, 2), stride=None, padding=0),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(16, 32, (3, 3), stride=1, padding=0),
            nn.ELU(),
            nn.MaxPool2d((2, 2), stride=None, padding=0),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(32, 64, (2, 2), stride=1, padding=0),
            nn.ELU(),
            nn.MaxPool2d((2, 2), stride=None, padding=0),
            nn.Dropout2d(p=0.3),
            nn.Conv2d(64, 128, (1, 1), stride=1, padding=0),
            nn.ELU(),
            nn.MaxPool2d((2, 2), stride=None, padding=0),
            nn.Dropout2d(p=0.4),
            nn.Flatten(),
            nn.Linear(3200, 1000),
            nn.ELU(),
            nn.Dropout(p=0.5),
            nn.Linear(1000, 1000),
            # Linear Activation Function
            nn.Dropout(p=0.6),
            nn.Linear(1000, 30),
            nn.Tanh()
        )

        nn.init.xavier_uniform_(self.model[17].weight)
        nn.init.xavier_uniform_(self.model[20].weight)
        nn.init.xavier_uniform_(self.model[22].weight)


        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):
        
        # check dimensions to use show_keypoint_predictions later
        if x.dim() == 3:
            x = torch.unsqueeze(x, 0)
        ########################################################################
        # TODO: Define the forward pass behavior of your model                 #
        # for an input image x, forward(x) should return the                   #
        # corresponding predicted keypoints.                                   #
        # NOTE: what is the required output size?                              #
        ########################################################################

        x = self.model(x)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return x

    def general_step(self, batch, batch_idx, mode):
        images,  keypoints = batch["image"], batch["keypoints"]

        # forward pass
        predicted_keypoints = self.forward(images).view(-1,15,2)

        # loss
        loss_function = nn.MSELoss()
        loss = loss_function(torch.squeeze(keypoints), torch.squeeze(predicted_keypoints))
        # loss = F.mse_loss(torch.squeeze(keypoints), torch.squeeze(predicted_keypoints))

        return loss

    # def general_end(self, outputs, mode):
    #     # average over all batches aggregated during one epoch
    #     avg_loss = torch.stack([x[mode + '_loss'] for x in outputs]).mean()
    #     return avg_loss

    def training_step(self, batch, batch_idx):
        loss = self.general_step(batch, batch_idx, "train")
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.general_step(batch, batch_idx, "val")
        self.log("val_loss", loss)
        return loss

    # def validation_epoch_end(self, outputs):
    #     avg_loss = self.general_end(outputs, "val")
    #     self.log('avg_loss', avg_loss)
    #     return avg_loss

    # def train_dataloader(self):
    #     return torch.utils.data.DataLoader(self.hparams['train_dataset'], shuffle=True, batch_size=self.hparams['batch_size'])

    # def val_dataloader(self):
    #     return torch.utils.data.DataLoader(self.hparams['val_dataset'], shuffle=True, batch_size=self.hparams['batch_size'])

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.model.parameters(), lr=self.hparams["learning_rate"],
                                                                    weight_decay=self.hparams["weight_decay"])
        StepLR = {'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optim, mode='min', 
                                                                    factor=self.hparams["lr_decay"], patience=2), 'monitor': 'train_loss'}
        return {"optimizer": optim, "lr_scheduler": StepLR}
    

class DummyKeypointModel(pl.LightningModule):
    """Dummy model always predicting the keypoints of the first train sample"""
    def __init__(self):
        super().__init__()
        self.prediction = torch.tensor([[
            0.4685, -0.2319,
            -0.4253, -0.1953,
            0.2908, -0.2214,
            0.5992, -0.2214,
            -0.2685, -0.2109,
            -0.5873, -0.1900,
            0.1967, -0.3827,
            0.7656, -0.4295,
            -0.2035, -0.3758,
            -0.7389, -0.3573,
            0.0086, 0.2333,
            0.4163, 0.6620,
            -0.3521, 0.6985,
            0.0138, 0.6045,
            0.0190, 0.9076,
        ]])

    def forward(self, x):
        return self.prediction.repeat(x.size()[0], 1, 1, 1)
