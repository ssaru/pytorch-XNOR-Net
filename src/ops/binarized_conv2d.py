from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from src.ops.utils import deterministic_quantize, stochastic_quantize
from src.types import quantization
from src.utils import prod


class BinarizedConv2d(torch.autograd.Function):
    r"""
    binarized tensor를 입력으로 하여,
    scale factor와 binarized weights로 Conv2D 연산을 수행하는 operation function

    Binarize operation method는 BinaryConnect의 `Deterministic`과 `Stochastic` method를 사용하며,
    scale factor와 weights를  binarize하는 방법은 XNOR-Net의 method를 사용함.

    .. note:
        Weights binarize method는 forward에서 binarized wegiths를 사용하고,
        gradient update는 real-value weights에 적용한다.

        `Deterministic` method는 다음과 같다.
        .. math::
            W_{b} = \bigg\{\begin{matrix}+1,\ \ \ if\ W\geq0, \ \ \\-1,\ \ \ otherwise,\end{matrix}

        `Stochastic` method는 다음과 같다.
        .. math::
            W_{b} = \bigg\{\begin{matrix}+1,\ \ \ with\ probability\ p=\sigma(w) \ \ \\-1,\ \ \ with\ probability\ 1-p\ \ \ \ \ \ \ \ \ \end{matrix}

        `Scale factor`는 다음과 같다.
        .. math::
            \alpha^{*} = \frac{\Sigma{|W_{i}|}}{n},\ where\ \ n=c\times w\times h

    Refers:
        1). BinaryConnect : https://arxiv.org/pdf/1511.00363.pdf
        2). XNOR-Net : https://arxiv.org/pdf/1603.05279.pdf
    """

    @staticmethod
    def forward(
        ctx: object,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        mode: str = quantization.QType.DETER,
    ) -> torch.Tensor:
        r"""
        Real-value weights를 binarized weight와 scale factor로 변환한다.
        binarized tensor이를입력으로 받으면 이를 다음과 같이 계산한다.

        .. math::
            output=I_{b} \odot W_{b} \times \alpha_{W_{b}}

        Args:
            ctx         (object): forward/backward간 정보를 공유하기위한 데이터 컨테이너
            input       (torch.Tensor): binairzed tensor
            weight      (torch.Tensor): :math:`(out\_features, in\_features)`
            bias        (Optional[torch.Tensor]): :math:`(out\_features)`
            stride      (Union[int, Tuple[int, int]]): the stride of the convolving kernel. Can be a single number or a tuple `(sH, sW)`. Default: 1
            padding     (Union[int, Tuple[int, int]]): implicit paddings on both sides of the input. Can be a single number or a tuple `(padH, padW)`. Default: 0
            dilation    (Union[int, Tuple[int, int]]): the spacing between kernel elements. Can be a single number or a tuple `(dH, dW)`. Default: 1
            groups:     (int): split input into groups, :math:`\text{in\_channels}` should be divisible by the number of groups. Default: 1
            mode        (str): 이진화 종류

        Returns:
            (torch.Tensor) : :math:\text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) + \sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star \text{input}(N_i, k)
        """

        weight_scale_factor, n = None, None

        with torch.no_grad():
            if mode == quantization.QType.DETER:
                binarized_weight = deterministic_quantize(weight)

                s = torch.sum(torch.abs(weight))
                n = prod(weight.shape)
                weight_scale_factor = s / n

            elif mode == quantization.QType.STOCH:
                binarized_weight = stochastic_quantize(weight)

                s = torch.sum(
                    torch.matmul(
                        torch.transpose(weight, dim0=2, dim1=3), binarized_weight
                    )
                )
                n = prod(weight.shape)
                weight_scale_factor = s / n
            else:
                raise RuntimeError(f"{mode} not supported")

        if (not weight_scale_factor) or (not n):
            raise RuntimeError("`scale_factor` or `n` not allow `None` value")

        device = weight.device
        binarized_weight = binarized_weight.to(device)

        with torch.no_grad():
            output = F.conv2d(
                input, binarized_weight, bias, stride, padding, dilation, groups
            )
            output = output * weight_scale_factor

        # Save input, binarized weight, bias in context object
        ctx.save_for_backward(input, binarized_weight * weight_scale_factor, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups

        return output

    @staticmethod
    def backward(ctx: object, grad_output: Any) -> Tuple[Optional[torch.Tensor]]:
        r"""
        gradient에 binarized weight를 마스킹하여 grad를 전달한다.

        Args:
            ctx (object): forward/backward간 정보를 공유하기위한 데이터 컨테이너
            grad_output (Any): Compuational graph를 통해서 들어오는 gradient정보

        Returns:
            (torch.Tensor) : Computational graph 앞으로 보내기위한 gradient 정보
        """
        input, binarized_weight_with_scale_factor, bias = ctx.saved_tensors

        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        groups = ctx.groups

        grad_input = grad_weight = grad_bias = None

        with torch.no_grad():
            if ctx.needs_input_grad[0]:
                grad_input = torch.nn.grad.conv2d_input(
                    input.shape,
                    binarized_weight_with_scale_factor,
                    grad_output,
                    stride,
                    padding,
                    dilation,
                    groups,
                )
            if ctx.needs_input_grad[1]:
                grad_weight = torch.nn.grad.conv2d_weight(
                    input,
                    binarized_weight_with_scale_factor.shape,
                    grad_output,
                    stride,
                    padding,
                    dilation,
                    groups,
                )

            if bias is not None and ctx.needs_input_grad[2]:
                grad_bias = grad_output.sum((0, 2, 3))

        return grad_input, grad_weight, grad_bias, None, None, None, None, None


binarized_conv2d = BinarizedConv2d.apply
