"""Custom loss functions."""
import torch
from torch import nn, Tensor
from torch.nn import functional as F

EPSILON = 1e-8


def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Hard Intersection-over-Union (IoU) metric.
    
    Source: https://www.kaggle.com/iezepov/fast-iou-scoring-metric-in-pytorch-and-numpy
    """
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    dims = (1, 2)  # dimensions to sum over
    intersection = (outputs & labels).float().sum(dims)  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum(dims)         # Will be zero if both are 0

    # We smooth our devision to avoid 0/0
    iou = (intersection + EPSILON) / (union + EPSILON)
    return iou


def dice_score(input: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Dice score metric."""
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    input = input.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    dims = (1, 2)  # dimensions to sum over
    # Count zero whenever either prediction or truth = 0
    intersection = (input.float() * labels.float()).sum(dims)
    im_sum = (input + labels).float().sum()

    # We smooth our devision to avoid 0/0
    dice = 2 * intersection / (im_sum + EPSILON)
    return dice


def soft_dice_loss(input: torch.Tensor, target: torch.Tensor, softmax=True) -> torch.Tensor:
    """Mean soft dice loss over the batch. From Milletari et al. (2016) https://arxiv.org/pdf/1606.04797.pdf 
    
    Parameters
    ----------
    input : Tensor
        (N,C,H,W) Predicted classes for each pixel.
    target : LongTensor
        (N,K,H,W) Tensor of pixel labels where `K` is the no. of classes.
    softmax : bool
        Whether to apply `F.softmax` to input to get class probabilities.
    """
    target = F.one_hot(target).permute(0, 3, 1, 2)
    dims = (1, 2, 3)  # sum over C, H, W
    if softmax:
        input = F.softmax(input, dim=1)
    intersect = torch.sum(input * target, dim=dims)
    denominator = torch.sum(input**2 + target**2, dim=dims)
    ratio = (intersect + EPSILON) / (denominator + EPSILON)
    return torch.mean(1 - 2. * ratio)


class CombinedLoss(nn.Module):
    """Combined cross-entropy + soft-dice loss
    
    See nn-UNet paper: https://arxiv.org/pdf/1809.10486.pdf"""
    
    def __init__(self, weight: torch.Tensor=None):
        super().__init__()
        self.weight = weight

    def __call__(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """CE + Dice"""
        return F.cross_entropy(input, target, reduction="mean", weight=self.weight) \
            + soft_dice_loss(input, target)


def soft_iou_loss(input: torch.Tensor, target: torch.Tensor):
    """https://arxiv.org/pdf/1705.08790.pdf"""
    raise NotImplementedError("IoU loss not implemented")

def tanimoto_loss(input: torch.Tensor, target: torch.Tensor, softmax=True):
    """Tanimoto loss. 
    
    See eq. 3 of ResU-Net paper: https://arxiv.org/pdf/1904.00592.pdf
    """
    target = F.one_hot(target).permute(0, 3, 1, 2)
    dims = (1, 2, 3)  # sum over C, H, W
    if softmax:
        input = F.softmax(input, dim=1)
    intersect = torch.sum(input * target, dim=dims)
    denominator = torch.sum(input**2 + target**2, dim=dims)
    ratio = (intersect + EPSILON) / (denominator - intersect + EPSILON)
    return 1. - ratio


def tanimoto_complement_loss(input: torch.Tensor, target: torch.Tensor, softmax=True):
    """Tanimoto loss with complement. 
    
    See eq. 3 of ResU-Net paper: https://arxiv.org/pdf/1904.00592.pdf
    """
    if softmax:
        input = F.softmax(input, dim=1)
    t_ = tanimoto_loss(input, target, False)
    t_c = tanimoto_complement_loss(input, target, False)
    return 5 * (t_ + t_c)


def focal_loss(input: torch.Tensor, target: torch.Tensor, gamma=2, alpha=1):
    """Focal loss (auto-weighted cross entropy variant).
    
    See: https://arxiv.org/pdf/1708.02002.pdf 
    """
    target = target.detach()
    ce_loss = F.cross_entropy(input, target)
    loss = alpha * (1 - torch.exp(-ce_loss)) ** gamma * ce_loss
    return loss


def generalized_dice_loss(y_pred, y_true):
    """Multi-label generalization of the dice loss."""
    Nb_img = y_pred.shape[-1]
    r = torch.zeros((Nb_img, 2))
    for l in range(Nb_img):
        r[l, 0] = torch.sum(y_true[:, :, l] == 0)
    for l in range(Nb_img):
        r[l, 1] = torch.sum(y_true[:, :, l] == 1)
    p = torch.zeros((Nb_img, 2))
    for l in range(Nb_img):
        p[l, 0] = torch.sum(y_pred[:, :, l][y_true[:, :, l] > 0])
    for l in range(Nb_img):
        p[l, 1] = torch.sum(y_pred[:, :, l][y_true[:, :, l] < 0])

    w = torch.zeros((2,))
    w[0] = 1/(torch.sum(r[:, 0])**2)
    w[1] = 1/(torch.sum(r[:, 1])**2)

    num = (w[0]*torch.sum(r[:, 0]*p[:, 0]))+(w[1]*torch.sum(r[:, 1]*p[:, 1]))
    denom = (w[0]*torch.sum(r[:, 0]+p[:, 0]))+(w[1]*torch.sum(r[:, 1]+p[:, 1]))

    return 1-(2*(num/denom))


if __name__ == "__main__":

    # test
    model = nn.Linear(10, 10)
    x = torch.randn((1, 10), requires_grad=True)
    target = torch.randint(0, 2, (10,)).float()
    output = model(x)
    sig = nn.Sigmoid()
    output = sig(output)
    loss = focal_loss(output, target)
    loss.backward()
    print(model.weight.grad)

    # test
    x = torch.randn((10, 10, 3), requires_grad=True)
    target = torch.randint(0, 2, (10, 10, 3)).float()
    output = sig(x)
    loss = generalized_dice_loss(output, target)
    loss.backward()
    print(model.weight.grad)

    x = torch.randn((10, 10, 3), requires_grad=True)
    output = sig(x)
    loss = focal_loss(output, target)
    loss.backward()
    print(model.weight.grad)
