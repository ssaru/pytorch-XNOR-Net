from typing import Any, Optional, Tuple

import torch

from src.types import quantization
from src.ops.utils import stochastic_quantize, deterministic_quantize


class BinarizedLinear(torch.autograd.Function):
    r"""
    Binary Operation에 대한 커스텀 Forward/Backward 정의
    Binary Operation Method는 `Deterministic`과 `Stochastic`으로 구분됨
    Refers:
        1). Custom Operation : https://pytorch.org/docs/stable/notes/extending.html
        2). Binary Operation Methods : https://arxiv.org/pdf/1511.00363.pdf
    """

    @staticmethod
    def forward(
        ctx: object,
        input: Tuple[torch.Tensor, torch.Tensor],
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        mode: Optional[str] = quantization.QType.DETER,
    ) -> torch.Tensor:
        r"""
        Binary forward operation을 정의한다.

        .. note:
            forward는 binarized wegiths를 이용하지만, backward는 real-value weights를 이용한다.
            torch.nn.functional.linear를 참고하였다.
            `Deterministic`과 `Stochastic`을 별도로 구현한다.
            Refs: https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#linear

            weights는 다음과 같이 scale factor와 binarized weights로 이진화될 수 있음
            
            .. math::
                \alpha^{*} = \frac{\Sigma{|I_{i}|}}{n},\ where\ \ n=c\times w\times h

            .. math::
                W_{b} = \bigg\{\begin{matrix}+1,\ \ \ if\ W\geq0, \ \ \\-1,\ \ \ otherwise,\end{matrix}


        Args:
            ctx (object): forward/backward간 정보를 공유하기위한 context 정보
            input (Tuple[torch.Tensor, torch.Tensor]): scale factor and binairzed input data.
            weight (torch.Tensor): :math:`(out\_features, in\_features)`
            bias (Optional[torch.Tensor]): :math:`(out\_features)`
            mode (str): `Deterministic`, `Stochastic` Method를 명시한다.

        Returns:
            (torch.Tensor) : binarized weights를 forwarding한 결과
        """
        bin_input, input_scale_factor = input
        weight_scale_factor, n = None, None

        # 별도로 Backward를 정의하므로 연산을 Computational Graph에 추가하지 않는다.
        with torch.no_grad():
            if mode == quantization.QType.DETER:
                binarized_weight = deterministic_quantize(weight)

                s = torch.sum(torch.abs(weight))
                n = sum(weight.shape)
                weight_scale_factor = s / n

            elif mode == quantization.QType.STOCH:
                binarized_weight = stochastic_quantize(weight)

                s = torch.sum(torch.abs(torch.matmul(weight.T, binarized_weight)))
                n = sum(weight.shape)
                weight_scale_factor = s / n

            else:
                raise RuntimeError(f"{mode} not supported")

            if (not weight_scale_factor) or (not n):
                raise RuntimeError("`scale_factor` or `n` not allow `None` value")

            device = weight.device
            binarized_weight = binarized_weight.to(device)
            output = bin_input.mm(binarized_weight.t()) * input_scale_factor * weight_scale_factor

            if bias is not None:
                output += bias.unsqueeze(0).expand_as(output)

        ctx.save_for_backward(input, binarized_weight, bias, weight_scale_factor, n)

        return output

    @staticmethod
    def backward(ctx: object, grad_output: Any):
        r"""
        Binary backward operation을 정의한다.

        Note:
            forward는 binarized wegiths를 이용하지만, backward는 real-value weights를 이용한다.

            gradient는 다음과 같이 계산된다. 이 때, alpha는 ``scale factor``이며 n은 ``width * height * channels``이다.
            
            .. math::
                \frac{\partial{C}}{\partial{\tilde{W_{i}}}}=\big(\frac{1}{n} + \frac{\partial{sign}}{\partial{W_{i}}} \alpha \big)

        Args:
            ctx (object): forward/backward간 정보를 공유하기위한 context 정보
            grad_output (Any): Compuational graph를 통해서 들어오는 gradient정보를 받는다.

        Returns:
            (torch.Tensor) : Computational graph 앞으로 보내기위한 gradient 정보
        """

        input, binarized_weight, bias, weight_scale_factor, n = ctx.saved_tensors
        bin_input, input_scale_factor = input

        grad_input = grad_weight = grad_bias = None
        grad = (1 / n) + (binarized_weight * weight_scale_factor)
        with torch.no_grad():
            if ctx.needs_input_grad[0]:
                grad_input = grad_output.mm(grad)
            if ctx.needs_input_grad[1]:
                grad_weight = grad_output.t().mm(bin_input) * input_scale_factor
            if (bias is not None) and (ctx.needs_input_grad[2]):
                grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias, None


binarized_linear = BinarizedLinear.apply
