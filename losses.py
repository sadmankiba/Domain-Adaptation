import segmentation_models_pytorch as smp
from torch import nn


def get_loss_function(loss_name):


    if loss_name == "cce":
        loss = nn.CrossEntropyLoss()

    
    elif loss_name == "dice":
        loss = smp.losses.DiceLoss(mode = "multiclass", from_logits=True)

    elif loss_name == "cce_dice":
        loss = Dice_CCE_Loss(weights=[0.5, 0.5])
    
    else:
        raise("Loss Function not found")
    

    return loss


class Dice_CCE_Loss():

    def __init__(self, weights):
        self.weights = weights
        self.cce_loss = nn.CrossEntropyLoss()
        self.dice_loss = smp.losses.DiceLoss(mode="multiclass", from_logits= True)

    
    def __call__(self, y_pred, y_true):
        return self.cce_loss(y_pred, y_true) * self.weights[0] + self.dice_loss(y_pred, y_true)*self.weights[1]
    
    
    

    




