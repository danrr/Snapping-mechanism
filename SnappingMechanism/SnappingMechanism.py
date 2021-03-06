import math
import secrets

import crlibm
import numpy as np

from Util import bits_to_float, float_to_bits


# TODO: add tests
class SnappingMechanism:
    def __init__(self, minimum_bound, maximum_bound, epsilon, sensitivity=1.0):
        self.minimum_bound = minimum_bound
        self.maximum_bound = maximum_bound
        self.sensitivity = sensitivity
        self.epsilon = epsilon

        # TODO check epsilon exists and is positive
        # TODO check sensitivity is positive
        # TODO check min < max and exist

        # TODO: The distribution ˜ f(·) defined above satisfies (1/λ+2^{−49}B/λ)-differential privacy
        #  when λ < B < 2^{46} · λ -- so epsilon doesn't quite mean the same thing -- how to handle this?
        #  exp(1/λ + 12Bη/λ + 2η),
        # TODO: do I need to check λ < B < 2^{46} · λ? throw error?

        # Compute a symmetric bound scaled to sensitivity 1 -- B in Mironov paper
        bound = (self.maximum_bound - self.minimum_bound) / 2.0
        self.B = bound / self.sensitivity

    def compute_real_epsilon(self):
        machine_epsilon = np.finfo(float).epsneg
        return self.epsilon + 12.0 * self.B * self.epsilon + machine_epsilon * 2.0

    def _scale_and_offset_value(self, value):
        """
        Centre value around 0 with symmetric bound; scale to sensitivity 1
        :param value: input value to the mechanism
        :return: value offset to be centered on 0 and scaled to sensitivity 1
        """
        value_scaled = value / self.sensitivity
        return value_scaled - self.B - (self.minimum_bound / self.sensitivity)

    def _reverse_scale_and_offset_value(self, value):
        return (value + self.B) * self.sensitivity + self.minimum_bound

    def _clamp_value(self, value):
        if value > self.B:
            return self.B
        if value < -self.B:
            return -self.B
        return value

    def _sample_uniform(self):
        # https://docs.python.org/3/library/random.html
        # structure of a double - sign: 1 bit; exponent: 11 bits; mantissa: 52 bits
        # A uniform distribution over D ∩ (0, 1) can be generated by independently sampling an exponent
        # from the geometric distribution with parameter .5 - ie. coin flips
        exponent = -53
        x = 0
        while not x:
            x = secrets.randbits(32)
            exponent += x.bit_length() - 32
        # and a significand by drawing a uniform string from {0, 1}^52
        mantissa = 1 << 52 | secrets.randbits(52)
        return math.ldexp(mantissa, exponent)

    def _get_next_power_of_2(self, x):
        # since epsilon and sensitivity are positive, this only needs to work for x > 0
        b = float_to_bits(x)
        e = (b >> 52) & ((1 << 11) - 1)
        m = b % (1 << 52)
        if e == 0:  # subnormal, very unlikely
            if m & (m - 1) == 0:
                return x
            return bits_to_float(1 << m.bit_length())
        if m == 0:
            return x
        return bits_to_float((e + 1) << 52)

    def _round_to_next_power_of_2(self, value):
        # Λ is the smallest power of 2 (including negative powers) greater than or equal to λ, and ⌊·⌉_Λ
        # rounds to the closest multiple of Λ in D with ties resolved towards +∞.

        # no need for sensitivity; already scaled down to 1
        base = self._get_next_power_of_2(1.0 / self.epsilon)
        remainder = math.fmod(value, base)
        if remainder > base / 2:
            return value - remainder + base
        if remainder == base / 2:  # ties resolved towards +∞
            return value + remainder
        return value - remainder

    def _sample_laplace(self):
        sign = secrets.randbits(1)
        uniform = self._sample_uniform()
        # Using crlibm, as mentioned in Mironov paper
        # no need for sensitivity already scaled down to sensitivity 1
        laplace = 1.0 / self.epsilon * crlibm.log_rn(uniform)
        if sign:
            return -laplace
        return laplace

    def add_noise(self, value):
        if not self.minimum_bound <= value <= self.maximum_bound:
            raise ValueError(f"Value {value} not within interval [{self.minimum_bound},[{self.maximum_bound}]")
        value_scaled_offset = self._scale_and_offset_value(value)
        value_clamped = self._clamp_value(value_scaled_offset)
        laplace = self._sample_laplace()
        value_rounded = self._round_to_next_power_of_2(value_clamped + laplace)
        return self._reverse_scale_and_offset_value(self._clamp_value(value_rounded))
