#!/usr/bin/env python3

# Modified by Andes Y. L. Kei: Implemented alternative approximations for Sigmoid, Tanh, Erf, GELU, and Softmax
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import crypten
import torch
from crypten.config import cfg

__all__ = [
    "exp",
    "log",
    "reciprocal",
    "inv_sqrt",
    "sqrt",
    "_eix",
    "cossin",
    "cos",
    "sin",
    "sigmoid",
    "tanh",
    "erf",
    "gelu",
    "silu",
    "softmax",
    "log_softmax",
]


# Iterative methods:
def exp(self):
    r"""Approximates the exponential function using a limit approximation:

    .. math::

        exp(x) = \lim_{n \\rightarrow \\infty} (1 + x / n) ^ n

    Here we compute exp by choosing n = 2 ** d for some large d equal to
    `iterations`. We then compute (1 + x / n) once and square `d` times.

    Set the number of iterations for the limit approximation with
    config.exp_iterations.
    """  # noqa: W605
    method = cfg.functions.exp_method
    iters = cfg.functions.exp_iterations

    if method == "ideal":
        return crypten.cryptensor(torch.exp(self.get_plain_text()), device=self.device)
    elif method == "limit":
        result = 1 + self.div(2**iters)
        for _ in range(iters):
            result = result.square()
        return result
    else:
        raise ValueError(f"Invalid method {method} given for exp function")

def log(self, input_in_01=False):
    r"""
    Approximates the natural logarithm using 8th order modified
    Householder iterations. This approximation is accurate within 2% relative
    error on [0.0001, 250].

    Iterations are computed by: :math:`h = 1 - x * exp(-y_n)`

    .. math::

        y_{n+1} = y_n - \sum_k^{order}\frac{h^k}{k}

    Args:
        input_in_01 (bool) : Allows a user to indicate that the input is in the domain [0, 1],
            causing the function optimize for this domain. This is useful for computing
            log-probabilities for entropy functions.

            We shift the domain of convergence by a constant :math:`a` using the following identity:

            .. math::

                \ln{u} = \ln {au} - \ln{a}

            Since the domain of convergence for CrypTen's log() function is approximately [1e-4, 1e2],
            we can set :math:`a=100`.

    Configuration parameters:
        iterations (int): number of Householder iterations for the approximation
        exp_iterations (int): number of iterations for limit approximation of exp
        order (int): number of polynomial terms used (order of Householder approx)
    """
    if input_in_01:
        return log(self.mul(100)) - 4.605170

    # Initialization to a decent estimate (found by qualitative inspection):
    #                ln(x) = x/120 - 20exp(-2x - 1.0) + 3.0
    iterations = cfg.functions.log_iterations
    exp_iterations = cfg.functions.log_exp_iterations
    order = cfg.functions.log_order

    term1 = self.div(120)
    term2 = exp(self.mul(2).add(1.0).neg()).mul(20)
    y = term1 - term2 + 3.0

    # 8th order Householder iterations
    with cfg.temp_override({"functions.exp_iterations": exp_iterations}):
        for _ in range(iterations):
            h = 1 - self * exp(-y)
            y -= h.polynomial([1 / (i + 1) for i in range(order)])
    return y


def reciprocal(self, input_in_01=False):
    r"""
    Args:
        input_in_01 (bool) : Allows a user to indicate that the input is in the range [0, 1],
                    causing the function optimize for this range. This is useful for improving
                    the accuracy of functions on probabilities (e.g. entropy functions).

    Methods:
        'NR' : `Newton-Raphson`_ method computes the reciprocal using iterations
                of :math:`x_{i+1} = (2x_i - self * x_i^2)` and uses
                :math:`3*exp(1 - 2x) + 0.003` as an initial guess by default

        'log' : Computes the reciprocal of the input from the observation that:
                :math:`x^{-1} = exp(-log(x))`

    Configuration params:
        reciprocal_method (str):  One of 'NR' or 'log'.
        reciprocal_nr_iters (int):  determines the number of Newton-Raphson iterations to run
                        for the `NR` method
        reciprocal_log_iters (int): determines the number of Householder
            iterations to run when computing logarithms for the `log` method
        reciprocal_all_pos (bool): determines whether all elements of the
            input are known to be positive, which optimizes the step of
            computing the sign of the input.
        reciprocal_initial (tensor): sets the initial value for the
            Newton-Raphson method. By default, this will be set to :math:
            `3*exp(-(x-.5)) + 0.003` as this allows the method to converge over
            a fairly large domain

    .. _Newton-Raphson:
        https://en.wikipedia.org/wiki/Newton%27s_method
    """
    pos_override = {"functions.reciprocal_all_pos": True}
    if input_in_01:
        with cfg.temp_override(pos_override):
            rec = reciprocal(self.mul(64)).mul(64)
        return rec

    # Get config options
    method = cfg.functions.reciprocal_method
    all_pos = cfg.functions.reciprocal_all_pos
    initial = cfg.functions.reciprocal_initial

    if method == "ideal":
        return crypten.cryptensor(torch.reciprocal(self.get_plain_text()), device=self.device)

    if not all_pos:
        sgn = self.sign()
        pos = sgn * self
        with cfg.temp_override(pos_override):
            return sgn * reciprocal(pos)

    if method == "NR":
        nr_iters = cfg.functions.reciprocal_nr_iters
        if initial is None:
            # Initialization to a decent estimate (found by qualitative inspection):
            #                1/x = 3exp(1 - 2x) + 0.003
            result = 3 * (1 - 2 * self).exp() + 0.003
        else:
            result = initial
        for _ in range(nr_iters):
            if hasattr(result, "square"):
                result += result - result.square().mul_(self)
            else:
                result = 2 * result - result * result * self
        return result
    elif method == "log":
        log_iters = cfg.functions.reciprocal_log_iters
        with cfg.temp_override({"functions.log_iters": log_iters}):
            return exp(-log(self))
    else:
        raise ValueError(f"Invalid method {method} given for reciprocal function")


def inv_sqrt(self):
    r"""
    Computes the inverse square root of the input using the Newton-Raphson method.

    Configuration params:
        sqrt_nr_iters (int):  determines the number of Newton-Raphson iterations to run.
        sqrt_nr_initial (tensor): sets the initial value for the Newton-Raphson iterations.
                    By default, this will be set to allow the method to converge over a
                    fairly large domain.

    .. _Newton-Raphson:
        https://en.wikipedia.org/wiki/Fast_inverse_square_root#Newton's_method
    """
    initial = cfg.functions.sqrt_nr_initial
    iters = cfg.functions.sqrt_nr_iters
    method = cfg.functions.sqrt_method

    if method == "ideal":
        return crypten.cryptensor(torch.rsqrt(self.get_plain_text()), device=self.device)
    elif method == "NR":
        # Initialize using decent approximation
        if initial is None:
            y = exp(self.div(2).add(0.2).neg()).mul(2.2).add(0.2)
            y -= self.div(1024)
        else:
            y = initial

        # Newton Raphson iterations for inverse square root
        for _ in range(iters):
            y = y.mul_(3 - self * y.square()).div_(2)
        return y
    else:
        raise ValueError(f"Invalid method {method} given for inv_sqrt function")


def sqrt(self):
    r"""
    Computes the square root of the input by computing its inverse square root using
    the Newton-Raphson method and multiplying by the input.

    Configuration params:
        sqrt_nr_iters (int):  determines the number of Newton-Raphson iterations to run
        sqrt_initial (tensor): sets the initial value for the inverse square root
            Newton-Raphson iterations. By default, this will be set to allow convergence
            over a fairly large domain.

    .. _Newton-Raphson:
        https://en.wikipedia.org/wiki/Fast_inverse_square_root#Newton's_method
    """
    return inv_sqrt(self).mul_(self)


def _eix(self):
    r"""Computes e^(i * self) where i is the imaginary unit.
    Returns (Re{e^(i * self)}, Im{e^(i * self)} = cos(self), sin(self)
    """
    iterations = cfg.functions.trig_iterations

    re = 1
    im = self.div(2**iterations)

    # First iteration uses knowledge that `re` is public and = 1
    re -= im.square()
    im *= 2

    # Compute (a + bi)^2 -> (a^2 - b^2) + (2ab)i `iterations` times
    for _ in range(iterations - 1):
        a2 = re.square()
        b2 = im.square()
        im = im.mul_(re)
        im._tensor *= 2
        re = a2 - b2

    return re, im


def cossin(self):
    r"""Computes cosine and sine of input via exp(i * x).

    Args:
        iterations (int): for approximating exp(i * x)
    """
    return self._eix()


def cos(self):
    r"""Computes the cosine of the input using cos(x) = Re{exp(i * x)}

    Args:
        iterations (int): for approximating exp(i * x)
    """
    return cossin(self)[0]


def sin(self):
    r"""Computes the sine of the input using sin(x) = Im{exp(i * x)}

    Args:
        iterations (int): for approximating exp(i * x)
    """
    return cossin(self)[1]


# Logistic Functions
def sigmoid(self):
    r"""Computes the sigmoid function using the following definition

    .. math::
        \sigma(x) = (1 + e^{-x})^{-1}

    If a valid method is given, this function will compute sigmoid
        using that method:

    "chebyshev" - computes tanh via Chebyshev approximation with
        truncation and uses the identity:

    .. math::
        \sigma(x) = \frac{1}{2}tanh(\frac{x}{2}) + \frac{1}{2}

    "reciprocal" - computes sigmoid using :math:`1 + e^{-x}` and computing
        the reciprocal

    """  # noqa: W605
    method = cfg.functions.sigmoid_tanh_method

    if method == "ideal":
        return crypten.cryptensor(torch.sigmoid(self.get_plain_text()), device=self.device)
    elif method == "chebyshev":
        tanh_approx = tanh(self.div(2))
        return tanh_approx.div(2) + 0.5
    elif method == "reciprocal":
        ltz = self._ltz()
        sign = 1 - 2 * ltz

        pos_input = self.mul(sign)
        denominator = pos_input.neg().exp().add(1)

        # TODO: Set these with configurable parameters
        with cfg.temp_override(
            {
                "functions.exp_iterations": 9,
                "functions.reciprocal_nr_iters": 3,
                "functions.reciprocal_all_pos": True,
                "functions.reciprocal_initial": 0.75,
            }
        ):
            pos_output = denominator.reciprocal()

        result = pos_output.where(1 - ltz, 1 - pos_output)
        # TODO: Support addition with different encoder scales
        # result = pos_output + ltz - 2 * pos_output * ltz
        return result
    elif method == "fourier":    
        m = cfg.functions.sigmoid_fs_m
        width = 2 ** (m - 1)
        terms = cfg.functions.sigmoid_fs_terms

        # note that beta_cos = 0 for tanh
        alpha, _, beta_sin = crypten.common.util.fourier_series(torch.tanh, width, terms)
        return _fourier_series(self, terms, m, alpha=alpha, beta_sin=beta_sin)
    else:
        raise ValueError(f"Unrecognized method {method} for sigmoid")


def tanh(self):
    r"""Computes the hyperbolic tangent function using the identity

    .. math::
        tanh(x) = 2\sigma(2x) - 1

    If a valid method is given, this function will compute tanh using that method:

    "chebyshev" - computes tanh via Chebyshev approximation with truncation.

    .. math::
        tanh(x) = \sum_{j=1}^terms c_{2j - 1} P_{2j - 1} (x / maxval)

    where c_i is the ith Chebyshev series coefficient and P_i is ith polynomial.
    The approximation is truncated to +/-1 outside [-1, 1].

    Args:
        terms (int): highest degree of Chebyshev polynomials.
                        Must be even and at least 6.
    """
    method = cfg.functions.sigmoid_tanh_method

    if method == "ideal":
        return crypten.cryptensor(torch.tanh(self.get_plain_text()), device=self.device)
    if method == "reciprocal":
        return self.mul(2).sigmoid().mul(2).sub(1)
    elif method == "chebyshev":
        terms = cfg.functions.sigmoid_tanh_terms
        coeffs = crypten.common.util.chebyshev_series(torch.tanh, 1, terms)[1::2]
        tanh_polys = _chebyshev_polynomials(self, terms)
        tanh_polys_flipped = (
            tanh_polys.unsqueeze(dim=-1).transpose(0, -1).squeeze(dim=0)
        )
        out = tanh_polys_flipped.matmul(coeffs)

        # truncate outside [-maxval, maxval]
        return out.hardtanh()
    elif method == "poly":
        drelu_x = self >= 0
        sign_x = 2 * drelu_x - 1
        abs_x = sign_x * self
        do_poly = abs_x < 2.95
        # TODO: use numpy.polynomial.Polynomial.fit() to fit the function
        poly_x = abs_x.polynomial([1.1950192,-0.49313435,0.0737858,-0.00147019]) - 0.01758266
        out = sign_x * (do_poly * (poly_x - 1) + 1)
        return out
    elif method == "fourier":
        m = cfg.functions.tanh_fs_m
        width = 2 ** (m - 1)
        terms = cfg.functions.tanh_fs_terms

        # note that alpha, beta_cos = 0 for tanh
        _, _, beta_sin = crypten.common.util.fourier_series(torch.tanh, width, terms)
        return _fourier_series(self, terms, m, beta_sin=beta_sin)
    elif method == "ode":
        iter_num = cfg.functions.tanh_ode_iter_num
        x = self / iter_num
        y = self.new(torch.zeros_like(self.data), device=self.device)
        for _ in range(iter_num):
            y += (1 - y * y) * x
        return y
    else:
        raise ValueError(f"Unrecognized method {method} for tanh")


def _chebyshev_polynomials(self, terms):
    r"""Evaluates odd degree Chebyshev polynomials at x

    Chebyshev Polynomials of the first kind are defined as

    .. math::
        P_0(x) = 1, P_1(x) = x, P_n(x) = 2 P_{n - 1}(x) - P_{n-2}(x)

    Args:
        self (MPCTensor): input at which polynomials are evaluated
        terms (int): highest degree of Chebyshev polynomials.
                        Must be even and at least 6.
    Returns:
        MPCTensor of polynomials evaluated at self of shape `(terms, *self)`
    """
    if terms % 2 != 0 or terms < 6:
        raise ValueError("Chebyshev terms must be even and >= 6")

    polynomials = [self.clone()]
    y = 4 * self.square() - 2
    z = y - 1
    polynomials.append(z.mul(self))

    for k in range(2, terms // 2):
        next_polynomial = y * polynomials[k - 1] - polynomials[k - 2]
        polynomials.append(next_polynomial)

    return crypten.stack(polynomials)

def _fourier_series(self, terms, period, alpha=None, beta_cos=None, beta_sin=None):
    r"""Evaluates Fourier series at x
    Args:
        self (MPCTensor): input at which series are evaluated
        terms (int): highest degree of Fourier series
        period (int): period of Fourier series, e.g., half_period = 2*L for range [-L, L] 
        alpha (int): constant coefficient
        beta_cos (tensor): coefficients of cosine terms, currently not supported
        beta_sin (tensor): coefficients of sine terms
    Returns:
        MPCTensor of Fourier series evaluated at self of shape `(*self)`
    """
    if beta_cos is not None:
        raise NotImplementedError("fourier series with cosine is currently not supported")
    elif beta_sin is None:
        raise ValueError("beta_sin cannot be None")
    
    device = self.device

    beta_sin = beta_sin.view([-1 if _ == 0 else 1 for _ in range(self.dim() + 1)]).to(device)
    beta_sin.requires_grad = True

    k = [i * 2 * math.pi / period for i in range(1, terms + 1)]

    # obtain trigonometric triples (t, sin(t), cos(t)) from dealer
    provider = crypten.mpc.get_default_provider()
    t, u, v = provider.generate_trig_triple(self.size(), period, terms, device=self.device)

    # reconstruct masked input (x-t)
    delta = self - t
    delta = delta.get_plain_text()

    # compute sin(x-t), cos(x-t)
    delta_k = torch.stack([i * delta for i in k])
    p, q = torch.sin(delta_k).to(device), torch.cos(delta_k).to(device)

    # evaluate sin(x) = sin(x - t) cos(t) + cos(x-t) sin(t)
    res = ((v * p + u * q) * beta_sin).sum(dim=0)

    if alpha is not None:
        res.add_(alpha)
    return res

def erf(tensor):
    r"""
    Approximates the error function of the input tensor.
    """
    method = cfg.functions.erf_method

    if method == "ideal":
        return crypten.cryptensor(torch.erf(tensor.get_plain_text()), device=tensor.device)
    elif method == "taylor":
        iters = cfg.functions.erf_iterations

        output = tensor.clone()
        for n in range(1, iters + 1):
            multiplier = ((-1) ** n) / (math.factorial(n) * (2 * n + 1))
            output = output.add(tensor.pos_pow(2 * n + 1).mul(multiplier))
        return output.mul(2.0 / math.sqrt(math.pi))
        # NOTE: This approximation is not unstable for large tensor values.
    elif method == "tanh":
        return tanh(math.sqrt(4 / math.pi) * (tensor + 0.044715 * tensor.pow(3)))
    elif method == "fourier":
        period = cfg.functions.erf_fs_period
        width = period / 2
        terms = cfg.functions.erf_fs_terms

        # note that alpha, beta_cos = 0 for erf
        _, _, beta_sin = crypten.common.util.fourier_series(torch.erf, width, terms)
        return _fourier_series(tensor, terms, period, beta_sin=beta_sin)
    else:
        raise ValueError(f"Unrecognized method {method} for erf")

def _diff_gelu(x):
    return torch.sign(x) * (torch.nn.functional.gelu(x, approximate="none") - torch.nn.functional.relu(x))

def _diff_gelu_tanh(x):
    return torch.sign(x) * (torch.nn.functional.gelu(x, approximate="tanh") - torch.nn.functional.relu(x))

def _diff_silu(x):
    return torch.sign(x) * (torch.nn.functional.silu(x) - torch.nn.functional.relu(x))

def gelu(self, approximate="none"):
    r"""Compute the Gaussian error linear unit of a tensor"""
    method = cfg.functions.gelu_method
    if method == "ideal":
        return crypten.cryptensor(torch.nn.functional.gelu(self.get_plain_text(), approximate=approximate), device=self.device)
    elif method == "fourier":
        period = cfg.functions.gelu_fs_period
        width = period / 2
        terms = cfg.functions.gelu_fs_terms

        relu_x = self.relu()
        abs_x = 2 * relu_x - self
        do_fs = abs_x < width

        if approximate == "tanh":
            #_, _, beta_sin = crypten.common.util.fourier_series(_diff_gelu_tanh, width, terms)
            beta_sin = torch.tensor([-0.0817,-0.0812,-0.0424,-0.0175,-0.0079,-0.0043,-0.0026,-0.0017], device=self.device)
        else:
            #_, _, beta_sin = crypten.common.util.fourier_series(_diff_gelu, width, terms)
            beta_sin = torch.tensor([-0.0818,-0.0809,-0.0424,-0.0176,-0.0079,-0.0043,-0.0026,-0.0017], device=self.device)
        out = relu_x + do_fs * _fourier_series(abs_x, terms, period, beta_sin=beta_sin)
        return out
    elif method == "secformer":
        # set erf_fs_period: 20, erf_fs_terms: 7
        b0, b1 = self > -1.7 * math.sqrt(2), self < 1.7 * math.sqrt(2)
        b0 = b0 * b1
        b1 = 1 - b1
        x_ = self / math.sqrt(2)
        gelu_fs = 0.5 * self * (1 + x_.erf())
        return b0 * gelu_fs + b1 * self
    elif method == "poly":
        relu_x = self.relu()
        abs_x = 2 * relu_x - self
        do_poly = abs_x < 3
        # TODO: use numpy.polynomial.Polynomial.fit() to fit the function
        poly_x = abs_x.polynomial([-0.55386347,0.5658561,-0.19719836,0.02328962]) + 0.00410626
        return relu_x + do_poly * poly_x
    elif method == "bolt":
        relu_x = self.relu()
        abs_x = 2 * relu_x - self
        do_poly = abs_x < 2.7
        # Motzkin's polynomial preprocessing
        g = [0.14439048359960427, -0.7077117131613893, 4.5702822654246535, -8.15444702051307, 16.382265425072532]
        poly_x = (g[0] * abs_x + g[1]) * abs_x + g[2]
        poly_x = (poly_x + g[0] * abs_x + g[3]) * abs_x + g[4] + 0.5 * self
        # The g's provided by BOLT are wrong, uncomment the following line to get the correct approximation
        #poly_x = abs_x.polynomial([-0.53798164612714154,0.5410550166368381,-0.18352506127082727,0.020848611754127593]) + 0.001620808531841547
        return relu_x + do_poly * poly_x
    elif method == "erf":
        # set erf_fs_period: 16, erf_fs_terms: 5
        b0, b1 = self > -2, self < 2
        b0 = b0 * b1
        b1 = 1 - b1
        x_ = self / math.sqrt(2)
        gelu_fs = 0.5 * self * (1 + x_.erf())
        return b0 * gelu_fs + b1 * self
    else:
        raise ValueError(f"Unrecognized method {method} for gelu")

def silu(self):
    r"""Compute the Sigmoid linear unit of a tensor"""
    method = cfg.functions.silu_method
    if method == "ideal":
        return crypten.cryptensor(torch.nn.functional.silu(self.get_plain_text()), device=self.device)
    elif method == "fourier":
        period = cfg.functions.silu_fs_period
        width = period / 2
        terms = cfg.functions.silu_fs_terms

        relu_x = self.relu()
        abs_x = 2 * relu_x - self
        do_fs = abs_x < width

        #_, _, beta_sin = crypten.common.util.fourier_series(_diff_silu, width, terms)
        beta_sin = torch.tensor([-0.1299, -0.1220, -0.0743, -0.0394, -0.0216, -0.0118, \
                                 -0.0074, -0.0044, -0.0033, -0.0021, -0.0018, -0.0011], device=self.device)
        out = relu_x + do_fs * _fourier_series(abs_x, terms, period, beta_sin=beta_sin)
        return out
    else:
        raise ValueError(f"Unrecognized method {method} for silu")

def softmax(self, dim, **kwargs):
    r"""Compute the softmax of a tensor's elements along a given dimension"""
    method = cfg.functions.softmax_method

    # 0-d case
    if self.dim() == 0:
        assert dim == 0, "Improper dim argument"
        return self.new(torch.ones_like((self.data)))

    if self.size(dim) == 1:
        return self.new(torch.ones_like(self.data))

    if method == "ideal":
        return crypten.cryptensor(torch.softmax(self.get_plain_text(), dim=dim), device=self.device)
    if method == "reciprocal":
        maximum_value = self.max(dim, keepdim=True)[0]
        logits = self - maximum_value
        numerator = logits.exp()
        with cfg.temp_override({"functions.reciprocal_all_pos": True}):
            inv_denominator = numerator.sum(dim, keepdim=True).reciprocal()
        return numerator * inv_denominator
    elif method == "ode":
        iter_num = cfg.functions.softmax_ode_iter_num
        clip = cfg.functions.softmax_ode_clip
        upper, lower = cfg.functions.softmax_ode_ub, cfg.functions.softmax_ode_lb

        if clip:
            # clip the input within the range [lower, upper] for numerical stability
            diff = crypten.cat([self - upper, lower - self]).relu().split(self.shape[0])#.split([1,1])
            self += diff[1] - diff[0]

        # initialize ode approximation
        x = self / iter_num
        g = self.new(torch.ones_like(self.data) / self.size(dim), device=self.device)

        # compute ode update formula
        for _ in range(iter_num):
            g += (x - g.mul(x).sum(dim=dim).unsqueeze(-1)).squeeze(-1) * g
        return g
    else:
        raise ValueError(f"Unrecognized method {method} for softmax")

def log_softmax(self, dim, **kwargs):
    r"""Applies a softmax followed by a logarithm.
    While mathematically equivalent to log(softmax(x)), doing these two
    operations separately is slower, and numerically unstable. This function
    uses an alternative formulation to compute the output and gradient correctly.
    """
    # 0-d case
    if self.dim() == 0:
        assert dim == 0, "Improper dim argument"
        return self.new(torch.zeros((), device=self.device))

    if self.size(dim) == 1:
        return self.new(torch.zeros_like(self.data))

    maximum_value = self.max(dim, keepdim=True)[0]
    logits = self - maximum_value
    normalize_term = exp(logits).sum(dim, keepdim=True)
    result = logits - normalize_term.log()
    return result
