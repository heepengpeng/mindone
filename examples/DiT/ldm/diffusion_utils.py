import mindspore as ms
import mindspore.nn.probability.distribution as msd
import numpy as np
from mindspore import ops


def normal_kl(mean1, logvar1, mean2, logvar2):
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, ms.Tensor):
            tensor = obj
            break
    assert tensor is not None

    logvar1, logvar2 = [
        x if isinstance(x, ms.Tensor) else ms.Tensor(x).to_tensor(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
            -1.0
            + logvar2
            - logvar1
            + ms.ops.exp(logvar1 - logvar2)
            + ((mean1 - mean2) ** 2) * ms.ops.exp(-logvar2)
    )


def approx_standard_normal_cdf(x):
    return 0.5 * (1.0 + ms.ops.tanh(
        ms.Tensor(np.sqrt(2.0 / np.pi + 1e-5), dtype=ms.float16) * (x + 0.044715 * ms.ops.pow(x, 3))))


def continuous_gaussian_log_likelihood(x, *, means, log_sales):
    centered_x = x - means
    inv_stdv = ms.Tensor.exp(-log_sales)
    normalized_x = centered_x * inv_stdv
    log_probs = msd.Normal(ms.ops.zeros_like(x), ms.ops.ones_like(x)).log_prob(normalized_x)
    return log_probs


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = ms.ops.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = ms.ops.log(ops.clip_by_value(cdf_plus, clip_value_min=ms.Tensor(1e-12)).astype(ms.float16))
    log_one_minus_cdf_min = ms.ops.log(
        ops.clip_by_value(1.0 - cdf_min, clip_value_min=ms.Tensor(1e-12)).astype(ms.float16))
    cdf_delta = cdf_plus - cdf_min
    log_probs = ms.numpy.where(
        x < -0.999,
        log_cdf_plus,
        ms.numpy.where(x > 0.999, log_one_minus_cdf_min,
                       ms.ops.log(ops.clip_by_value(cdf_delta, clip_value_min=ms.Tensor(1e-12)).astype(ms.float16))),
    )
    assert log_probs.shape == x.shape
    return log_probs
