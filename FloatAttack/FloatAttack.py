from random import RECIP_BPF

import numpy as np
import diffprivlib.mechanisms.laplace as ibm_laplace
from matplotlib import pyplot as plt

from SnappingMechanism.SnappingMechanism import SnappingMechanism


# Table 1 Mironov - multiples of 2^(−53)
BASE_FLOAT = RECIP_BPF  # 2 ** -53

NUMBER_ITERATIONS = 1_000


def is_base(x, y, scale):
    return x > 0 and scale * np.log(x) == y


def has_base_in_uniform(lap, scale, *, base=BASE_FLOAT):
    # symmetric distribution around 0, cast to negative
    if lap > 0:
        lap *= -1.0

    # does there exist an x such that scale * log(x) == y
    # AND it is sampled from the uniform distribution

    unscaled_lap = lap / scale
    uniform_base = np.exp(unscaled_lap)
    u = uniform_base - uniform_base % base
    for x in [u - base - base - base,
              u - base - base,
              u - base,
              u,
              u + base,
              u + base + base,
              u + base + base + base,
              ]:
        if is_base(x, lap, scale):
            return True
    return False


# from StatDP paper TODO: link and full name
def histogram(prng, queries, epsilon):
    noisy_array = np.asarray(queries, dtype=np.float64) + prng.laplace(scale=1.0 / epsilon, size=len(queries))
    return noisy_array[0]


def iterate_attack(attack, *args, iterations=NUMBER_ITERATIONS, **kwargs):
    count = 0
    for i in range(iterations):
        if attack(*args, **kwargs):
            count += 1
    return count


def attack_numpy_single(scale, offset):
    result = offset + np.random.laplace(scale=scale)
    return has_base_in_uniform(result, scale)


def attack_numpy(scale):
    print("Naive Laplace implementation using Numpy:")
    count = iterate_attack(attack_numpy_single, scale, 0.)
    print(f"True positive: {count}/{NUMBER_ITERATIONS}")

    count = iterate_attack(attack_numpy_single, scale, 1.)
    print(f"False positive: {count}/{NUMBER_ITERATIONS}")


def attack_histogram_single(epsilon, scale, offset):
    result = histogram(np.random.default_rng(), [offset, 1, 1, 1, 1], epsilon)
    return has_base_in_uniform(result, scale)


def attack_histogram(epsilon, scale):
    print("StatDP histogram implementation:")
    count = iterate_attack(attack_histogram_single, epsilon, scale, 0.)
    print(f"True positive: {count}/{NUMBER_ITERATIONS}")

    count = iterate_attack(attack_histogram_single, epsilon, scale, 1.)
    print(f"False positive: {count}/{NUMBER_ITERATIONS}")


def attack_ibm_single(laplace, scale, offset):
    result = laplace.randomise(offset)
    return has_base_in_uniform(result, scale)


def attack_ibm(sensitivity, epsilon, scale):
    print("IBM Laplace:")
    laplace = ibm_laplace.Laplace(epsilon=epsilon, sensitivity=sensitivity)
    count = iterate_attack(attack_ibm_single, laplace, scale, 0.)
    print(f"True positive: {count}/{NUMBER_ITERATIONS}")

    count = iterate_attack(attack_ibm_single, laplace, scale, 1.)
    print(f"False positive: {count}/{NUMBER_ITERATIONS}")


def attack_sm_single(sm, scale, offset):
    result = sm.add_noise(offset)
    return has_base_in_uniform(result, scale)


def attack_sm(sensitivity, epsilon, scale):
    print("Snapping Mechanism:")
    minimum = 0.
    maximum = 100.
    sm = SnappingMechanism(minimum, maximum, epsilon, sensitivity)

    count = iterate_attack(attack_sm_single, sm, scale, 0.)
    print(f"True positive: {count}/{NUMBER_ITERATIONS}")

    count = iterate_attack(attack_sm_single, sm, scale, sensitivity)
    print(f"False positive: {count}/{NUMBER_ITERATIONS}")


def plot_results():
    percentage = []
    scales = np.arange(0, 3., .01)[1:]
    for scale in scales:
        count = 0
        for i in range(NUMBER_ITERATIONS):
            result = 1 + np.random.laplace(scale=scale)
            if has_base_in_uniform(result, scale):
                count += 1
        percentage.append(1.0 - count/NUMBER_ITERATIONS)
    plt.plot(scales, percentage)
    plt.axis([0, 3, 0, 1.02])
    plt.xlabel('Scale of the Laplace distribution')
    plt.ylabel('Distinguisher probability of success')
    plt.margins(x=0, y=5)
    plt.show()


def laplace(sign, uniform, scale):
    return sign * scale * np.log(uniform)


def attack_all_floats(scale):
    print("Naive Laplace implementation - running through all floats")
    exponent = 24
    base = np.float32(2)**-np.float32(exponent)
    iterations = 2**exponent

    counts = {
        0.0: 0,
        1.0: 0,
    }

    scale = np.float32(scale)
    for i in range(1, iterations + 1):
        uniform = np.float32(i) * base
        lap = laplace(np.float32(1), uniform, scale)
        for offset in counts.keys():
            result = np.float32(offset) + lap
            if has_base_in_uniform(result, scale, base=base):
                counts[offset] += 1
    for offset, count in counts.items():
        print(f'{offset}: {count}; {iterations - count}')


def main():
    sensitivity = 1.0
    epsilon = .0001
    scale = sensitivity/epsilon

    attack_numpy(scale)
    attack_histogram(epsilon, scale=1.0/epsilon)  # histogram always has sensitivity 1 since it's a count
    attack_ibm(sensitivity, epsilon, scale)
    attack_sm(sensitivity, epsilon, scale)
    attack_all_floats(scale)


if __name__ == '__main__':
    main()
