import torch


class BinarizeWithScaleFactor(torch.nn.Module):
    """Binarized a tensor image with scale factor
    This transform does not support PIL Image.

    Given tensor image; ``(pixel[1],...,pixel[n])`` for ``n`` channels,
    this transform will get scaling factor and binarized input

    .. math::
        \alpha^{*} = \frac{\Sigma{|I_{i}|}}{n},\ where\ \ n=c\times w\times h


    .. math::
        I_{b} = \bigg\{\begin{matrix}+1,\ \ \ if\ I\geq0, \ \ \\-1,\ \ \ otherwise,\end{matrix}

    .. note::
        This transform acts out of place, i.e., it does not mutate the input tensor.
    """

    def __init__(self):
        super().__init__()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tensor (torch.Tensor): Tensor image to be binarized input.
        Returns:
            (torch.Tensor, torch.Tensor): Binarized Tensor image and scaling factor.
        """

        scale_factor = torch.sum(torch.abs(tensor)) / sum(tensor.shape)
        binarized_input = tensor.sign()
        binarized_input[binarized_input == 0] = 1

        return binarized_input, scale_factor
