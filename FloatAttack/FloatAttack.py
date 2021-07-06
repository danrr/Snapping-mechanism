import numpy as np
import diffprivlib.mechanisms.laplace as ibm_laplace

from Util import bits_to_float, float_to_bits
from SnappingMechanism.SnappingMechanism import SnappingMechanism


# modify the bit representation
def add_bits_float(num, add):
    return bits_to_float(float_to_bits(num) + add)


# Table 1 Mironov - multiples of 2^(−53)
base_float = 2 ** -53


def is_uniform(x):
    # is a multiple of the base_float used for uniform sampling in numpy
    # return x / base_float == x // base_float
    return x % base_float == 0


def is_base(x, y, scale):
    return scale * np.log(x) == y


def has_base_in_uniform(lap, scale):
    # symmetric distribution around 0, cast to negative
    if lap > 0:
        lap *= -1.0

    # does there exist an x such that scale * log(x) == y
    # AND it is sampled from the uniform distribution

    unscaled_lap = lap / scale
    uniform_base = np.exp(unscaled_lap)
    search_range = 100
    for j in range(-search_range, search_range):
        x = add_bits_float(uniform_base, j)
        # for all x around the search target, check if any could have been the uniformly sampled base
        if is_base(x, lap, scale) and is_uniform(x):
            return True
    return False


# from StatDP paper TODO: link and full name
def histogram(prng, queries, epsilon):
    noisy_array = np.asarray(queries, dtype=np.float64) + prng.laplace(scale=1.0 / epsilon, size=len(queries))
    return noisy_array[0]


def attack_numpy(scale):
    print("Naive Laplace implementation using Numpy:")
    count = 0
    for i in range(1000):
        result = np.random.laplace(scale=scale)

        if has_base_in_uniform(result, scale):
            count += 1
    print(f"True positive: {count}/1000")

    count = 0
    for i in range(1000):
        result = 1 + np.random.laplace(scale=scale)

        if has_base_in_uniform(result, scale):
            count += 1
    print(f"False positive: {count}/1000")


def attack_histogram(epsilon, scale):
    print("StatDP histogram implementation:")
    count = 0
    for i in range(1000):
        result = histogram(np.random.default_rng(), [0, 1, 1, 1, 1], epsilon)
        if has_base_in_uniform(result, scale):
            count += 1
    print(f"True positive: {count}/1000")

    count = 0
    for i in range(1000):
        result = histogram(np.random.default_rng(), [2, 1, 1, 1, 1], epsilon)
        if has_base_in_uniform(result, scale):
            count += 1
    print(f"False positive: {count}/1000")


def attack_ibm(sensitivity, epsilon, scale):
    print("IBM Laplace:")
    count = 0
    laplace = ibm_laplace.Laplace(epsilon=epsilon, sensitivity=sensitivity)
    for i in range(1000):
        result = laplace.randomise(0)

        if has_base_in_uniform(result, scale):
            count += 1
    print(f"True positive: {count}/1000")

    count = 0
    for i in range(1000):
        result = laplace.randomise(sensitivity)

        if has_base_in_uniform(result, scale):
            count += 1
    print(f"False positive: {count}/1000")


def attack_sm(sensitivity, epsilon, scale):
    print("Snapping Mechanism:")
    count = 0
    minimum = 0.
    maximum = 100.
    sm = SnappingMechanism(minimum, maximum, epsilon, sensitivity)
    for i in range(1000):
        result = sm.add_noise(0.)

        if has_base_in_uniform(result, scale):
            count += 1
    print(f"True positive: {count}/1000")

    count = 0
    for i in range(1000):
        result = sm.add_noise(sensitivity)

        if has_base_in_uniform(result, scale):
            count += 1
    print(f"False positive: {count}/1000")


def main():
    sensitivity = 1.0
    epsilon = .0001
    scale = sensitivity/epsilon

    attack_numpy(scale)
    attack_histogram(epsilon, scale=1.0/epsilon)  # histogram always has sensitivity 1 since it's a count
    attack_ibm(sensitivity, epsilon, scale)
    attack_sm(sensitivity, epsilon, scale)


if __name__ == '__main__':
    main()
