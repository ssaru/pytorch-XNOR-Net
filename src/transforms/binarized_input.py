import torch


class BinarizeWithScaleFactor(torch.nn.Module):
    """tensor 이미지를 scale factor와 binarized image로 변환한다.
    이 transform은 PIL Image를 지원하지 않는다.

    tensor image가 다음과 같이 주어지면: ``(pixel[1],...,pixel[n])`` for ``n`` channels,
    이 transform은 scale factor와 이진화된 tensor image를 반환할 것이다.

    scale factor와 이진화된 입력은 다음과 같다.
    .. math::
        \alpha^{*} = \frac{\Sigma{|I_{i}|}}{n},\ where\ \ n=c\times w\times h


    .. math::
        I_{b} = \bigg\{\begin{matrix}+1,\ \ \ if\ I\geq0, \ \ \\-1,\ \ \ otherwise,\end{matrix}

    """

    def __init__(self):
        super().__init__()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tensor (torch.Tensor): 이진화 입력과 scale factor을 구하기 위한 tensor image
        Returns:
            (torch.Tensor, torch.Tensor): 이진화된 입력과 scale factor
        """

        scale_factor = torch.sum(torch.abs(tensor)) / sum(tensor.shape)
        binarized_input = tensor.sign()
        binarized_input[binarized_input == 0] = 1

        return binarized_input, scale_factor
