import torch


def deterministic_quantize(weights: torch.Tensor) -> torch.Tensor:
    # torch.sign()함수는 Ternary로 Quantization 된다. [-1, 0, 1]
    # 따라서 `0`에 대한 별도 처리를 해야 Binary weight를 가질 수 있다.

    with torch.no_grad():
        binarized_weight = weights.sign()
        binarized_weight[binarized_weight == 0] = 1

    return binarized_weight


def stochastic_quantize(weights: torch.Tensor) -> torch.Tensor:
    # weights를 sigmoid 입력으로 넣어 이를 확률값으로 변환한다. `sigmoid(weights)`
    # 해당 확률값을 이용하여 [-1, 1]을 생성한다.
    # +1 if w >= p, where p = sigmoid(w)
    # -1 else 1 - p
    # p값을 먼저 구한다.
    # [0, 1]사이의 값을 갖는 데이터에서 uniform 확률 분포로 데이터를 샘플링한다.
    # sampling된 값들이 p값 이상을 갖으면 1, 그렇지 않으면 -1로 정의한다.
    with torch.no_grad():
        binarized_probability = torch.sigmoid(weights.to("cpu"))
        uniform_matrix = torch.empty(binarized_probability.shape).uniform_(0, 1)
        binarized_weight = (binarized_probability >= uniform_matrix).type(torch.float32)
        binarized_weight[binarized_weight == 0] = -1

    return binarized_weight
