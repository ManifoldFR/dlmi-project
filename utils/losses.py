"""Custom loss functions."""
import torch
from torch import nn, Tensor
from torch.nn import functional as F

EPSILON = 1e-8


class DiceLoss(nn.Module):
    """
    
    Attributes
    ----------
    soft : bool
        Variant of the Dice loss to use.
    """
    
    def __init__(self, apply_softmax=True, variant=None):
        super().__init__()
        self.apply_softmax = apply_softmax
        self.variant = str(variant).lower()
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.variant == 'soft':
            return soft_dice_loss(input, target, self.apply_softmax)
        elif self.variant == 'none':
            return dice_loss(input, target, self.apply_softmax)


def dice_loss(input: torch.Tensor, target: torch.Tensor,
              softmax=True, smooth=EPSILON) -> torch.Tensor:
    """Regular Dice loss.
    
    Parameters
    ----------
    input: Tensor
    (N, K, H, W) Predicted classes for each pixel.
    target: LongTensor
    (N, K, H, W) Tensor of pixel labels where `K` is the no. of classes.
    softmax: bool
    Whether to apply `F.softmax` to input to get class probabilities.
    """
    target = F.one_hot(target).permute(0, 3, 1, 2)
    dims = (1, 2, 3)  # sum over C, H, W
    if softmax:
        input = F.softmax(input, dim=1)
    intersect = torch.sum(input * target, dim=dims)
    denominator = torch.sum(input + target, dim=dims)
    loss = 1 - (2 * intersect + smooth) / (denominator + smooth)
    return loss.mean()


def soft_dice_loss(input: torch.Tensor, target: torch.Tensor, softmax=True) -> torch.Tensor:
    """Mean soft dice loss over the batch. From Milletari et al. (2016) https://arxiv.org/pdf/1606.04797.pdf 
    
    Parameters
    ----------
    input : Tensor
        (N,K,H,W) Predicted classes for each pixel.
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
    """Combined cross-entropy + dice loss. The soft Dice loss can also be used.
    
    See nn-UNet paper: https://arxiv.org/pdf/1809.10486.pdf"""
    
    def __init__(self, weight: torch.Tensor=None, dice_variant=None):
        super().__init__()
        self._dice = DiceLoss(variant=dice_variant)
        self.weight = weight

    def __call__(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """CE + Dice"""
        return F.cross_entropy(input, target, reduction="mean", weight=self.weight) \
            + self._dice(input, target)


def soft_iou_loss(input: torch.Tensor, target: torch.Tensor):
    """https://arxiv.org/pdf/1705.08790.pdf"""
    raise NotImplementedError("IoU loss not implemented")


def tanimoto_loss(input: torch.Tensor, target: torch.Tensor, softmax=True):
    """Tanimoto loss. 
    
    See eq. (3) of ResU-Net paper: https://arxiv.org/pdf/1904.00592.pdf
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
    
    See eq. (3) of ResU-Net paper: https://arxiv.org/pdf/1904.00592.pdf
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
    ce_loss = F.cross_entropy(input, target, reduce=False)  # vector of loss terms
    loss = alpha * (1 - torch.exp(-ce_loss)) ** gamma * ce_loss
    return loss.mean()


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
