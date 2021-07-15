from random import RECIP_BPF

import numpy as np
import diffprivlib.mechanisms.laplace as ibm_laplace

from SnappingMechanism.SnappingMechanism import SnappingMechanism


# Table 1 Mironov - multiples of 2^(−53)
BASE_FLOAT = RECIP_BPF  # 2 ** -53

NUMBER_ITERATIONS = 1_000_000


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
    u = uniform_base - uniform_base % BASE_FLOAT

    for x in [u, u + BASE_FLOAT]:
        if is_base(x, lap, scale):
            return True
    return False


# from StatDP paper TODO: link and full name
def histogram(prng, queries, epsilon):
    noisy_array = np.asarray(queries, dtype=np.float64) + prng.laplace(scale=1.0 / epsilon, size=len(queries))
    return noisy_array[0]


def attack_numpy(scale):
    print("Naive Laplace implementation using Numpy:")
    count = 0
    for i in range(NUMBER_ITERATIONS):
        result = np.random.laplace(scale=scale)

        if has_base_in_uniform(result, scale):
            count += 1
    print(f"True positive: {count}/{NUMBER_ITERATIONS}")

    count = 0
    for i in range(NUMBER_ITERATIONS):
        result = 1 + np.random.laplace(scale=scale)

        if has_base_in_uniform(result, scale):
            count += 1
    print(f"False positive: {count}/{NUMBER_ITERATIONS}")


def attack_histogram(epsilon, scale):
    print("StatDP histogram implementation:")
    count = 0
    for i in range(NUMBER_ITERATIONS):
        result = histogram(np.random.default_rng(), [0, 1, 1, 1, 1], epsilon)
        if has_base_in_uniform(result, scale):
            count += 1
    print(f"True positive: {count}/{NUMBER_ITERATIONS}")

    count = 0
    for i in range(NUMBER_ITERATIONS):
        result = histogram(np.random.default_rng(), [2, 1, 1, 1, 1], epsilon)
        if has_base_in_uniform(result, scale):
            count += 1
    print(f"False positive: {count}/{NUMBER_ITERATIONS}")


def attack_ibm(sensitivity, epsilon, scale):
    print("IBM Laplace:")
    count = 0
    laplace = ibm_laplace.Laplace(epsilon=epsilon, sensitivity=sensitivity)
    for i in range(NUMBER_ITERATIONS):
        result = laplace.randomise(0)

        if has_base_in_uniform(result, scale):
            count += 1
    print(f"True positive: {count}/{NUMBER_ITERATIONS}")

    count = 0
    for i in range(NUMBER_ITERATIONS):
        result = laplace.randomise(sensitivity)

        if has_base_in_uniform(result, scale):
            count += 1
    print(f"False positive: {count}/{NUMBER_ITERATIONS}")


def attack_sm(sensitivity, epsilon, scale):
    print("Snapping Mechanism:")
    count = 0
    minimum = 0.
    maximum = 100.
    sm = SnappingMechanism(minimum, maximum, epsilon, sensitivity)
    for i in range(NUMBER_ITERATIONS):
        result = sm.add_noise(0.)

        if has_base_in_uniform(result, scale):
            count += 1
    print(f"True positive: {count}/{NUMBER_ITERATIONS}")

    count = 0
    for i in range(NUMBER_ITERATIONS):
        result = sm.add_noise(sensitivity)

        if has_base_in_uniform(result, scale):
            count += 1
    print(f"False positive: {count}/{NUMBER_ITERATIONS}")


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
