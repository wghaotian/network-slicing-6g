from typing import Any

import numpy as np
from numpy import sqrt, dtype
from scipy import optimize, special
import matplotlib.pyplot as plt
from queue import PriorityQueue
import pickle
from datetime import datetime


def inv_Q(x: float) -> float:
    """
    Inverse of Gaussian Q Function
    :param x: x in Q^{-1}(x)
    :return: Q^{-1}(x)
    """
    return sqrt(2) * special.erfinv(1 - 2 * x)


def calc_V(gammas: np.ndarray[Any, dtype[float]]) -> np.ndarray[Any, dtype[float]]:
    """
    Calculate V[u] given gamma[u]
    :param gammas: SINR of users, gamma[u] is $$\gamma_u$$.
    """
    return (1 - 1 / (1 + gammas) ** 2) * (np.log2(np.e) ** 2)


def dB2power(dB: np.ndarray[Any, dtype[float]]) -> np.ndarray[Any, dtype[float]]:
    """
    Calculate power in Watt given its level in dB
    """
    return np.power(10, dB / 10)


def calc_pathloss(distance: np.ndarray[Any, dtype[float]], freqency: float = 2400) \
        -> np.ndarray[Any, dtype[float]]:
    """
    Calculate path loss given distance and frequency
    :param distance: distance in meters
    :param freqency: frequency in Hz
    :return pathloss: pathloss coefficient
    """
    pathloss_db = 20 * np.log10(distance) + 20 * np.log10(freqency) - 27.55
    pathloss = dB2power(pathloss_db)
    return pathloss


def calc_bus_and_cus(B: float, delta: float, gamma: np.ndarray[Any, dtype[float]], epsilon: float,
                     kus: np.ndarray[Any, dtype[int]] | None = None) -> \
        (np.ndarray[Any, dtype[float]], np.ndarray[Any, dtype[float]]):
    """
    Calculates the b value for the given $$V, B, \delta, \gamma and Q^{-1}(\epsilon)$$
    :param B: Value of the B
    :param delta: Value of the delta
    :param gamma: a 1-dimensional array, where gamma[u] is the value of the $$\gamma_{u, k}^{(t)}$$ for all $k$
    :param epsilon: Float, the value of $$\epsilon$$
    :param kus: Array of integers, where kus[u] is the RB $$k$$ associated to user $$u$$

    :return bus: Array of floats, where b[u] is the value of $$b_u$$
    :return cus: Array of floats, where c[u] is the value of $$c_u$$
    """
    # gammas = gamma[:, kus].diagonal()
    gammas = gamma.copy()
    Vs = calc_V(gammas)
    inv_q_epsilon = inv_Q(epsilon)
    bus = sqrt(Vs) * inv_q_epsilon / (2 * sqrt(B * delta) * np.log(1 + gammas))
    cus = 1 / (B * delta * np.log(1 + gammas))
    return bus, cus


def calc_K_chernoff_given_mu_star(mu_star: float, bus: np.ndarray[Any, dtype[float]],
                                  cus: np.ndarray[Any, dtype[float]], lambdas: np.ndarray[Any, dtype[float]],
                                  derivative: bool) -> float:
    """
    Calculates the $$K'$$ for the given $$\mu^*$$
    :param mu_star: Value of the $$\mu^*$$
    :param bus: Array of floats, where b[u] is the value of $$b_u$$
    :param cus: Array of floats, where c[u] is the value of $$c_u$$
    :param derivative: Boolean, whether to calculate the derivative
    :return K_chernoff: Value of $$K'$$
    :return dK_chernoff: Value of $$\frac{d\mu^*}{dK'}$$
    """
    if derivative:
        dK_chernoff = 4 * np.sum(lambdas * (cus ** 2) * np.exp(2 * mu_star * cus))
        return dK_chernoff
    else:
        K_chernoff = 4 * np.sum(bus ** 2) + 2 * np.sum(lambdas * cus * np.exp(2 * mu_star * cus))
        # K_chernoff = 4 * np.sum(bus ** 2) + 2 * np.sum(lambdas * cus * np.exp(2 * mu_star * cus)) + cus.shape[0]
        return K_chernoff


def substituted_inequality_43(mu: float, cus: np.ndarray[Any, dtype[float]],
                              lambdas: np.ndarray[Any, dtype[float]], epsilon: float) -> float:
    """
    Calculates the left-hand-side of substituted inequality (43)
    :param mu: float, the value of $$\mu^*$$
    :param cus: Array of floats, where c[u] is the value of $$c_u$$
    :param lambdas: Array of floats, where lambdas[u] is the value of $$\lambda_u$$
    :param epsilon: float, the value of $$\epsilon_{\text{puncture}}$$
    """
    return (-2 * mu * np.dot(lambdas, cus * np.exp(2 * cus * mu)) + np.dot(lambdas, np.exp(2 * mu * cus))
            - np.sum(lambdas) - np.log(epsilon))
    # return (-2 * mu * np.dot(lambdas, cus * np.exp(2 * cus * mu)) + np.dot(lambdas, np.exp(2 * mu * cus))
    #         - np.sum(lambdas + mu) - np.log(epsilon))


def solve_mu_star_given_epsilon(cus: np.ndarray[Any, dtype[float]], lambdas: np.ndarray[Any, dtype[float]],
                                epsilon: float) -> float:
    """
    Solves the modified equation (43) by substituting $$K'$$ with equation (41)
    :param cus: Array of floats, where c[u] is the value of $$c_u$$
    :param lambdas: Array of integers, where lambdas[u] is the value of $$lambdas[u]$$
    :param epsilon: float, value of $$\epsilon_\text{puncture}$$
    :return mu: Value of $$\mu^*$$ solved in the inequality.
    """
    num_users = cus.shape[0]
    func = lambda mu: substituted_inequality_43(mu, cus, lambdas, epsilon)
    # derivative = lambda mu: -4 * mu * np.dot(lambdas, cus ** 2 * np.exp(2 * cus * mu))
    derivative = lambda mu: -4 * mu * np.dot(lambdas, cus ** 2 * np.exp(2 * cus * mu))
    mu = optimize.newton(func, 1 / np.max(cus), fprime=derivative)
    return mu


def solve_mu_star(K_chernoff: float, bus: np.ndarray[Any, dtype[float]], cus: np.ndarray[Any, dtype[float]],
                  lambdas: np.ndarray[Any, dtype[float]]) -> float:
    """
    Solves the equation (41) where
        $$ K\' = 4 \sum_{u \in \mathcal{U} b_u^2 + 2 \sum_{u \in \mathcal{U} \lambda_u c_u e^{2 c_u \mu^*} $$
    :param K_chernoff: Value of $$K'$$
    :param bus: Array of floats, where bus[u] is the value of $$b_u$$
    :param cus: Array of floats, where cus[u] is the value of $$c_u$$
    :param lambdas: Array of integers, where lambdas[u] is the value of $$\lambda_u$$

    :return mu_star: Value of $$\mu^*$$ solved from the equation'
    """
    func = lambda mu: calc_K_chernoff_given_mu_star(mu, bus, cus, lambdas, False) - K_chernoff
    f_chernoff = lambda mu: calc_K_chernoff_given_mu_star(mu, bus, cus, lambdas, True)
    mu_star = optimize.newton(func, 0.0, f_chernoff)
    return mu_star


def find_K_chernoff(lambdas: np.ndarray[Any, dtype[float]], bus: np.ndarray[Any, dtype[float]],
                    cus: np.ndarray[Any, dtype[float]], epsilon: float) -> (int, float):
    """
    Finds the smallest integer of $$K'$$ such that the inequality (43) holds where
        $$\exp(-\mu^* K' - \sum_{u \in \mathcal{U}} \lambda_u + 4 \mu^* \sum_{u \in \mathcal{U}} b_u^2 +
            \sum_{u \in \mathcal{U}}\lambda_u e^{2 \mu^* c_u}) \le \epsilon$$
    :param lambdas: Array of integers, where lambdas[u] is the value of $$\lambda_u$$
    :param bus: Array of floats, where bus[u] is the value of $$b_u$$
    :param cus: Array of floats, where cus[u] is the value of $$c_u$$
    :param epsilon: Value of $$\epsilon_\mathrm{puncture}$$

    :return floored_K_chernoff: the smallest integer of $$K'$$ such that the inequality (43) holds
    :return K_chernoff: the not floored bound of $$K_chernoff$$
    """
    mu = solve_mu_star_given_epsilon(cus, lambdas, epsilon)
    K_chernoff = calc_K_chernoff_given_mu_star(mu, bus, cus, lambdas, False)
    floored_K_Chernoff = -np.floor(-(K_chernoff + cus.shape[0]))
    return int(floored_K_Chernoff), K_chernoff


# TO IMPLEMENT 2024.08.30
#     Gaussian Q-Function
#     Consider flat rate, where gamma[u][k] = gamma[u], V[u][k] = V[u]
#     Compare results with Markov results (30), plot it
#      Plot 1:
#          Fixed epsilon in [1e-5, 1e-3]
#          x-axis: the expectation of lambdas, which follows a certain distribution
#          y-axis: K' calculated by implemented function and markov inequality
#          Plot different curves with different mean value of SINR (gamma), can follow rayleigh distribution
#      Plot 2:
#          Fixed gamma following rayleigh distribution
#          x-axis: epsilon
#          y-axis: K' and K from (30)
#          Plot different curves with different lambda
#
#     Verify (25) using Monte Carlo
#       K_puncture: (10)


def markov_ineq_lower_bound(bus: np.ndarray[Any, dtype[float]], cus: np.ndarray[Any, dtype[float]], epsilon: float,
                            lambdas: np.ndarray[Any, dtype[float]]) -> float:
    """
    Lower bound of K given by Markov inequality
    :param bus: Array of floats, where bus[u] is the value of $$b_u$$
    :param cus: Array of floats, where cus[u] is the value of $$c_u$$
    :param epsilon: Value of $$\epsilon_\mathrm{puncture}$$
    :param lambdas: Array of integers, where lambdas[u] is the value of $$\lambda_u$$
    """
    return (4 * np.dot(bus ** 2, 1 - np.exp(-lambdas)) + 2 * np.dot(cus, lambdas)) / epsilon  # lambda or -lambda?


def sample_data(bus: np.ndarray[Any, dtype[float]], cus: np.ndarray[Any, dtype[float]], epsilon: float,
                mean_lambda: float) -> (int, float):
    """
    Sample data for plot 1
    :param bus: Array of floats, where bus[u] is the value of $$b_u$$
    :param cus: Array of floats, where cus[u] is the value of $$c_u$$
    :param epsilon: Value of $$\epsilon_\mathrm{puncture}$$
    :param mean_lambda: mean value of lambda\
    """
    num_users = bus.shape[0]
    lambdas = np.random.poisson(mean_lambda, num_users)
    K_chernoff, _ = find_K_chernoff(lambdas, bus, cus, epsilon)
    K_markov = markov_ineq_lower_bound(bus, cus, epsilon, lambdas)
    return K_chernoff, K_markov


def plot_1(epsilon: float, num_users: int, B: float, delta: float,
           lambda_lower_bound: float, lambda_upper_bound: float, num_samples: int, scale_sinr: float = 1.0,
           with_latex: bool = True):
    K_chernoffs = np.array([])
    K_markovs = np.array([])
    gammas = np.random.rayleigh(scale=scale_sinr, size=num_users)
    bus, cus = calc_bus_and_cus(B, delta, gammas, epsilon)
    # print(f"bus: {bus}")
    # print(f"cus: {cus}")
    mean_lambdas = np.linspace(lambda_lower_bound, lambda_upper_bound, num_samples)
    for mean_lambda in mean_lambdas:
        K_chernoff, K_markov = sample_data(bus, cus, epsilon, mean_lambda)
        K_chernoffs = np.append(K_chernoffs, K_chernoff)
        K_markovs = np.append(K_markovs, K_markov)
    if with_latex:
        plt.rcParams['text.usetex'] = True
        plt.plot(mean_lambdas, K_chernoffs, label=r"$K'$")
        plt.plot(mean_lambdas, K_markovs, label=r'$K_{min}$')
        plt.xlabel(r"$\bar{\lambda}$")
        plt.title(rf"$\epsilon = {epsilon}, \gamma \sim "r"\mathrm{Rayleigh}"f"({scale_sinr})"r"$")
    else:
        plt.rcParams['text.usetex'] = False
        plt.plot(mean_lambdas, K_chernoffs, label="K'")
        plt.plot(mean_lambdas, K_markovs, label='K_min')
        plt.xlabel("$lambda")
        plt.title(f"epsilon = {epsilon}, gamma: Rayleigh({scale_sinr})")
    plt.legend()
    plt.show()


def plot_2(num_users: int, B: float, delta: float, mean_lambda: float, epsilon_lower_bound: float,
           epsilon_upper_bound: float, num_samples: int, scale_sinr: float = 1.0, with_latex: bool = True):
    K_chernoffs = np.array([])
    K_markovs = np.array([])
    gammas = np.random.rayleigh(scale=scale_sinr, size=num_users)
    epsilons = np.linspace(epsilon_lower_bound, epsilon_upper_bound, num_samples)
    for epsilon in epsilons:
        bus, cus = calc_bus_and_cus(B, delta, gammas, epsilon)
        K_chernoff, K_markov = sample_data(bus, cus, epsilon, mean_lambda)
        K_chernoffs = np.append(K_chernoffs, K_chernoff)
        K_markovs = np.append(K_markovs, K_markov)

    if with_latex:
        plt.rcParams['text.usetex'] = True
        plt.plot(epsilons, K_chernoffs, label=r"$K'$")
        plt.plot(epsilons, K_markovs, label=r'$K_{min}$')
        plt.xlabel(r"$\epsilon$")
        plt.title(r"$\bar{\lambda} = "f"{mean_lambda}"r", \gamma \sim \mathrm{Rayleigh}("f"{scale_sinr})$")
    else:
        plt.rcParams['text.usetex'] = False
        plt.plot(epsilons, K_chernoffs, label="K'")
        plt.plot(epsilons, K_markovs, label='K_min')
        plt.xlabel("epsilon")
        plt.title("lambda = "f"{mean_lambda}"", gamma: Rayleigh("f"{scale_sinr})")

    plt.legend()
    plt.show()


def generate_aus(lambdas: np.ndarray[Any, dtype[float]], stochastic: bool = False) -> np.ndarray[Any, np.dtype[int]]:
    """
    Generate aus[u] given lambdas[u]
    :param lambdas: Arrival rate of each user
    :param stochastic: Whether to generate stochastic aus[u] sim lambdas[u] or let aus[u] = lambdas[u]
    :return aus: generated arrival datasize
    """
    if not stochastic:
        return lambdas
    else:
        aus = np.zeros_like(lambdas)
        for i, lambd in enumerate(lambdas):
            aus[i] = np.random.poisson(lambd)
        return aus


def calc_omegas(epsilon: float, gammas: np.ndarray[Any, dtype[float]],
                aus: np.ndarray[Any, dtype[float]], B: float, delta: float) -> (
np.ndarray[Any, dtype[float]], np.ndarray[Any, dtype[float]]):
    """
    Calculates omegas[u]
    :param epsilon: Value of $$\epsilon$$
    :param gammas: SINR of each user
    :param aus: Data arrivals of each user
    :param B: Bandwidth in Hz
    :param delta: length of mini-slots
    """
    V = calc_V(gammas)
    term1 = sqrt(V) * inv_Q(epsilon)
    log_sinr = np.log(1 + gammas)
    omegas = (term1 + sqrt(term1 ** 2 + 4 * log_sinr * aus)) / (2 * sqrt(B * delta) * log_sinr)
    omegas = omegas ** 2
    return -np.floor(-omegas), omegas  # takes the smallest integer larger than omegas


def pole2cart(rs: np.ndarray[Any, dtype[float]],
              thetas: np.ndarray[Any, dtype[float]]) -> (np.ndarray[Any, dtype[float]],
                                                         np.ndarray[Any, dtype[float]]):
    return rs * np.cos(thetas), rs * np.sin(thetas)


def cart2pole(xs: np.ndarray[Any, dtype[float]],
              ys: np.ndarray[Any, dtype[float]]) -> (np.ndarray[Any, dtype[float]],
                                                     np.ndarray[Any, dtype[float]]):
    return sqrt(xs ** 2 + ys ** 2), np.arctan2(ys, xs)


def generate_user_positions(num_users: int,
                            maximum_radius: float, type: str = "pole") -> (np.ndarray[Any, dtype[float]],
                                                                           np.ndarray[Any, dtype[float]]):
    """
    Generate user positions given number of users. Positions are uniformly distributed within maximum_radius.
    It is assumed that the only basestation lies at the original.
    :param num_users: Number of users.
    :param maximum_radius: Maximum radius.
    :param type: "pole" or "cartesian"
    :return rus: Radius of users.
    :return thetas: Angles of users.
    """
    rus = np.random.uniform(low=0.0, high=maximum_radius, size=num_users)
    thetas = np.random.uniform(low=0.0, high=2 * np.pi, size=num_users)
    if type == "pole":
        return rus, thetas
    elif type == "cartesian":
        return pole2cart(rus, thetas)
    else:
        raise ValueError("Type must be either 'pole' or 'cartesian'")


def generate_gammas(rus: np.ndarray[Any, dtype[float]],
                    scale_gamma: float) -> np.ndarray[Any, dtype[float]]:
    """
    Generate gamma given radius of users and random channel factors following a rayleigh distribution.
    """
    num_users = rus.shape[0]
    # gammas = np.random.rayleigh(size=num_users, scale=scale_gamma)
    gammas = np.ones_like(rus)
    pathlosses = calc_pathloss(rus)
    gammas /= pathlosses
    return gammas


def generate_lambdas(mean_lambda: float, num_users: int) -> np.ndarray[Any, dtype[float]]:
    """
    Generate lambdas[u] given mean lambda and number of users following poisson distribution.
    :param mean_lambda: Mean lambda of poisson distribution
    :param num_users: Number of users.
    """
    # return np.random.poisson(mean_lambda, size=num_users)
    lambdas = (np.ones(num_users) * mean_lambda).astype(float)
    return lambdas


def monte_carlo_prob(num_samples: int, num_users: int, maximum_radius: float,
                     B: float, delta: float, gamma_scale: float, mean_lambda: float,
                     epsilon_error: float, epsilon_puncture: float,
                     stochastic_aus: bool = False) -> (int, int, int):
    num_hit_chernoff = 0
    num_hit_markov = 0
    rus, thetas = generate_user_positions(num_users, maximum_radius, type="pole")
    gammas = generate_gammas(rus, gamma_scale)
    bus, cus = calc_bus_and_cus(B, delta, gammas, epsilon_error)
    lambdas = generate_lambdas(mean_lambda, num_users)
    K_markov = markov_ineq_lower_bound(bus, cus, epsilon_puncture, lambdas)
    K_real_sum = 0
    try:
        K_chernoff, _ = find_K_chernoff(lambdas, bus, cus, epsilon_puncture)
    except RuntimeError as e:
        K_chernoff = 0
    for i in range(int(num_samples / epsilon_puncture)):
        aus = generate_aus(lambdas, stochastic_aus)
        omegas, _ = calc_omegas(epsilon_error, gammas, aus, B, delta)
        K_real = np.sum(omegas)
        K_real_sum += K_real
        if K_real > K_chernoff:
            num_hit_chernoff += 1
        if K_real > K_markov:
            num_hit_markov += 1
        print_progress(i, int(num_samples / epsilon_puncture), 15)
    return num_hit_chernoff, num_hit_markov, num_samples


def monte_carlo_expectations(num_samples: int, rus: np.ndarray[Any, dtype[float]],
                             B: float, delta: float, gammas: np.ndarray[Any, dtype[float]],
                             bus: np.ndarray[Any, dtype[float]], cus: np.ndarray[Any, dtype[float]],
                             lambdas: np.ndarray[Any, dtype[float]],
                             epsilon_error: float, epsilon_puncture: float,
                             stochastic_aus: bool = True) -> (float, float, float):
    K_markov = markov_ineq_lower_bound(bus, cus, epsilon_puncture, lambdas)
    K_real_sum = 0
    try:
        K_chernoff, _ = find_K_chernoff(lambdas, bus, cus, epsilon_puncture)
    except RuntimeError as e:
        K_chernoff = 0
    for i in range(num_samples):
        aus = generate_aus(lambdas, stochastic_aus)
        omegas, _ = calc_omegas(epsilon_error, gammas, aus, B, delta)
        K_real = np.sum(omegas)
        K_real_sum += K_real
        # if (i+1) % 100 == 0:
        #     print(f"average K_real:{K_real_sum / (i + 1)}, K_markov:{K_markov}, K_chernoff:{K_chernoff}")
    return K_real_sum / num_samples, K_chernoff, K_markov


def monte_carlo_bounds(num_samples: int, rus: np.ndarray[Any, dtype[float]],
                       B: float, delta: float, gammas: np.ndarray[Any, dtype[float]],
                       bus: np.ndarray[Any, dtype[float]], cus: np.ndarray[Any, dtype[float]],
                       lambdas: np.ndarray[Any, dtype[float]],
                       epsilon_error: float, epsilon_puncture: float,
                       stochastic_aus: bool = True, floored_bound: bool = False) -> (float, float, float, int, float):
    """

    :param num_samples:
    :param rus:
    :param B:
    :param delta:
    :param gammas:
    :param bus:
    :param cus:
    :param lambdas:
    :param epsilon_error:
    :param epsilon_puncture:
    :param stochastic_aus:
    :param floored_bound:
    :return:
        K_real, K_real_not_floored, K_chernoff, floored_K_chernoff, K_markov
    """
    K_markov = markov_ineq_lower_bound(bus, cus, epsilon_puncture, lambdas)
    num_iterations = (np.floor(num_samples / epsilon_puncture)).astype(int)
    ranked_index = -np.floor(-num_iterations * (epsilon_puncture)).astype(int)
    K_reals = PriorityQueue(ranked_index)
    try:
        floored_K_chernoff, K_chernoff = find_K_chernoff(lambdas, bus, cus, epsilon_puncture)
    except RuntimeError as e:
        floored_K_chernoff, K_chernoff = 0, 0
    flag = False
    for i in range(num_iterations):
        aus = generate_aus(lambdas, stochastic_aus)
        omegas, omegas_not_floored = calc_omegas(epsilon_error, gammas, aus, B, delta)
        K_real = np.sum(omegas)
        K_real_not_floored = np.sum(omegas_not_floored)
        K_tuple = (K_real, K_real_not_floored)
        if not K_reals.full():
            K_reals.put(K_tuple)
        else:
            cur_min_K_reals = K_reals.get()
            if cur_min_K_reals < K_tuple:
                K_reals.put(K_tuple)
            else:
                K_reals.put(cur_min_K_reals)
        if not flag:
            print(f"current epsilon_puncture = {epsilon_puncture}, lambda mean = {np.mean(lambdas)}")
            flag = True
        print_progress(i, num_iterations, 20)
    K_real, K_real_not_floored = K_reals.get()
    return K_real, K_real_not_floored, K_chernoff, floored_K_chernoff, K_markov


def print_progress(cur_step: int, total_step: int, progress_num: int, bar_size: int = 20, finished_char: str = '=',
                   unfinished_char: str = '.') -> bool:
    big_step_size = int(-np.floor(-total_step / progress_num))
    if cur_step % big_step_size == 0 and cur_step:
        progress = int(cur_step / total_step * bar_size)
        print(f"Current Progress {finished_char * progress + unfinished_char * (bar_size - progress)}"
              f" {cur_step}/{total_step}")
        return True
    else:
        return False


def plot_k_bounds_lambda(num_samples: int, num_users: int, num_plot_samples: int, maximum_radius: float,
                         B: float, delta: float, gamma_scale: float, lambda_min: float, lambda_max: float,
                         epsilon_error: float, epsilon_puncture: float, timestamp: str,
                         stochastic_aus: bool = False, with_latex: bool = True, filename="K_bounds_lambda",
                         floored_bounds: bool = False
                         ):
    rus, thetas = generate_user_positions(num_users, maximum_radius, type="pole")
    gammas = generate_gammas(rus, gamma_scale)
    bus, cus = calc_bus_and_cus(B, delta, gammas, epsilon_error)
    mean_lambdas = np.linspace(lambda_min, lambda_max, num_plot_samples)
    K_reals, K_chernoffs, floored_K_chernoffs, K_markovs, K_reals_not_floored = [], [], [], [], []
    env_vars = {"bus": bus, "cus": cus, "rus": rus, "gammas": gammas}
    for (i, mean_lambda) in enumerate(mean_lambdas):
        lambdas = generate_lambdas(mean_lambda, num_users)
        K_real, K_real_not_floored, K_chernoff, floored_K_chernoff, K_markov = monte_carlo_bounds(num_samples, rus, B, delta, gammas, bus,
                                                                              cus, lambdas,
                                                                              epsilon_error, epsilon_puncture,
                                                                              stochastic_aus)
        K_reals.append(K_real)
        K_chernoffs.append(K_chernoff)
        K_markovs.append(K_markov)
        K_reals_not_floored.append(K_real_not_floored)
        floored_K_chernoffs.append(floored_K_chernoff)
        if print_progress(i, num_plot_samples, 15, finished_char="#", unfinished_char="*"):
            if with_latex:
                plt.rcParams['text.usetex'] = True
                # plt.plot(mean_lambdas, K_markovs, label=r'$K_\mathrm{markov}$')
                if floored_bounds:
                    plt.plot(mean_lambdas[0: i + 1], floored_K_chernoffs, label=r"$\lceil K_\mathrm{chernoff} \rceil$")
                    plt.plot(mean_lambdas[0: i + 1], K_reals_not_floored, label=r"$\lceil K_\mathrm{puncture} \rceil$")
                # plt.plot(mean_lambdas[0: i + 1], K_reals, label=r"$K_\mathrm{puncture}$")
                else:
                    plt.plot(mean_lambdas[0: i + 1], K_chernoffs, label=r"$K_\mathrm{chernoff}$")
                    plt.plot(mean_lambdas[0: i + 1], K_reals_not_floored, label=r"$K_\mathrm{puncture}$")
                plt.xlabel(r"$\bar{\lambda}$")
                plt.title(r"${\epsilon_\mathrm{puncture}} = "f"{epsilon_puncture}"r"$")
            else:
                plt.rcParams['text.usetex'] = False
                if floored_bounds:
                    plt.plot(mean_lambdas[0: i + 1], K_reals, label="K_puncture_ceiled")
                    plt.plot(mean_lambdas[0: i + 1], floored_K_chernoffs, label="K_chernoff_ceiled")
                else:
                    plt.plot(mean_lambdas[0: i + 1], K_chernoffs, label="K_chernoff")
                    plt.plot(mean_lambdas[0: i + 1], K_reals_not_floored, label="K_puncture_not_ceiled")
                plt.xlabel("mean_lambda")
                plt.title("epsilon_puncture = "f"{epsilon_puncture}")
            plt.legend()
            plt.savefig(f"{filename}_{timestamp}.png")
            plt.close()
            with open(f"{filename}_{timestamp}.pkl", "wb") as file:
                data = {"lambdas": mean_lambdas, "K_reals": K_reals, "K_chernoffs": K_chernoffs, "K_markov": K_markov,
                        "K_not_floored": K_reals_not_floored, "env_vars": env_vars,
                        "floored_K_chernoffs": floored_K_chernoffs}
                pickle.dump(data, file)


def plot_k_bounds_epsilon(num_samples: int, num_users: int, num_plot_samples: int, maximum_radius: float,
                          B: float, delta: float, gamma_scale: float, mean_lambda: float,
                          epsilon_error: float, epsilon_puncture_min: int, epsilon_puncture_max: int, timestamp: str,
                          stochastic_aus: bool = False, with_latex: bool = True, filename: str = "K_bounds_epsilon",
                          floored_bound: bool = False):
    rus, thetas = generate_user_positions(num_users, maximum_radius, type="pole")
    gammas = generate_gammas(rus, gamma_scale)
    bus, cus = calc_bus_and_cus(B, delta, gammas, epsilon_error)
    epsilons = np.logspace(epsilon_puncture_min, epsilon_puncture_max, num_plot_samples)
    lambdas = generate_lambdas(mean_lambda, num_users)
    env_vars = {"bus": bus, "cus": cus, "rus": rus, "gammas": gammas}
    K_reals, K_chernoffs, floored_K_chernoffs, K_markovs, K_reals_not_floored = [], [], [], [], []
    sample_length = len(epsilons)
    for (i, epsilon_puncture) in enumerate(epsilons):
        (K_real, K_real_not_floored, K_chernoff,
         floored_K_chernoff, K_markov) = monte_carlo_bounds(num_samples, rus, B,
                                                            delta, gammas, bus,
                                                            cus, lambdas,
                                                            epsilon_error, epsilon_puncture,
                                                            stochastic_aus)
        K_reals.append(K_real)
        K_chernoffs.append(K_chernoff)
        K_markovs.append(K_markov)
        K_reals_not_floored.append(K_real_not_floored)
        floored_K_chernoffs.append(K_chernoffs)
        if print_progress(i, sample_length, 20, finished_char='#', unfinished_char='*'):
            if with_latex:
                plt.rcParams['text.usetex'] = True
                if floored_bound:
                    plt.plot(epsilons[0: i + 1], K_reals, label=r'$\lceil K_\mathrm{puncture} \rceil$')
                    plt.plot(epsilons[0: i + 1], floored_K_chernoffs, label=r'$\lceil K_\mathrm{chernoff} \rceil$')
                else:
                    plt.plot(epsilons[0: i + 1], K_reals_not_floored, label=r'${K_\mathrm{puncture}}$')
                    plt.plot(epsilons[0: i + 1], K_chernoffs, label=r"$K_\mathrm{chernoff}$")
                plt.xlabel(r"$\epsilon$")
                plt.title(r"$\bar{\lambda} = "f"{mean_lambda}"r")$")
            else:
                plt.rcParams['text.usetex'] = False
                if not floored_bound:
                    plt.plot(epsilons[0: i + 1], K_chernoffs, label="K_chernoff_not_floored")
                    plt.plot(epsilons[0: i + 1], K_reals_not_floored, label="K_puncture_not_floored")
                else:
                    plt.plot(epsilons[0: i + 1], K_reals, label='K_puncture_floored')
                    plt.plot(epsilons[0: i + 1], floored_K_chernoffs, label="K_chernoff_floored")
                plt.xlabel("epsilon")
                plt.title("mean_lambda = "f"{mean_lambda}")
            plt.xscale("log")
            plt.legend()
            plt.savefig(f"{filename}_{timestamp}.png")
            plt.close()
            with open(f"{filename}_{timestamp}.pkl", 'wb') as file:
                data = {"epsilons": epsilons, "K_chernoffs": K_chernoffs, "K_reals_not_floored": K_reals_not_floored,
                        "K_reals": K_reals, "K_markovs": K_markovs, "env_vars": env_vars}
                pickle.dump(data, file)


if __name__ == '__main__':
    # num_users = 1000
    # bus = np.random.rand(num_users)
    # cus = np.random.rand(num_users)
    # lambdas = np.random.rand(num_users)
    # K_chernoff = np.random.randint(num_users) + np.floor( 4 * np.sum(bus ** 2)) + 1
    # mu_star = solve_mu_star(K_chernoff=K_chernoff, bus=bus, cus=cus, lambdas=lambdas)
    # # print(f"bus: {bus}, cus: {cus}")
    # print(f"K_chernoff: {K_chernoff}, lambdas: {lambdas}, mu_star: {mu_star}")
    # print(f"K_chernoff(mu_star):{calc_K_chernoff_given_mu_star(mu_star, bus, cus, lambdas, False)}")
    # epsilon = np.random.rand()
    # print(f"epsilon: {epsilon}")
    # # K_chernoff_1 = find_K_chernoff(lambdas=lambdas, bus=bus, cus=cus, epsilon=epsilon)  # Didn't work out!!!
    # # print(f"K_chernoff_1: {K_chernoff_1}")
    # K_chernoff_1 = find_K_chernoff(lambdas, bus, cus, epsilon)
    # mu_star_1 = solve_mu_star(K_chernoff_1, bus, cus, lambdas)
    # lefthand = substituted_inequality_43(mu_star_1, cus, lambdas, epsilon)
    # assert(lefthand <= 0)
    # print(f"lefthand: {lefthand}")
    # print(f"K_chernoff_1: {K_chernoff_1}")
    # K_chernoff_2 = K_chernoff_1 - 1
    # mu_star_2 = solve_mu_star(K_chernoff_2, bus, cus, lambdas)
    # lefthand = substituted_inequality_43(mu_star_2, cus, lambdas, epsilon)
    # assert(lefthand > 0)
    # print(f"lefthand: {lefthand}, K_chernoff_2: {K_chernoff_2}")
    # print("Test successful!")
    #
    # B = 1
    # delta = 1e-3
    # epsilon = 1e-1
    # lambda_lower_bound = 1
    # lambda_upper_bound = 10
    # num_samples = 100
    # scale_sinr = 100
    # plot_1(epsilon, num_users, B, delta, lambda_lower_bound, lambda_upper_bound, num_samples, scale_sinr)
    # epsilon_lower_bound = 1e-3
    # epsilon_upper_bound = 1e-1
    # mean_lambda = 6
    # plot_2(num_users, B, delta, mean_lambda, epsilon_lower_bound, epsilon_upper_bound, num_samples, scale_sinr)

    num_users = 1000
    B = 1e15
    delta = .5e-3
    epsilon_puncture = 1e-3
    epsilon_error = 1e-5
    lambda_lower_bound = 1
    lambda_upper_bound = 10
    mean_lambda = 0.5
    num_samples = 100
    gamma_scale = 1
    maximum_radius = 1000
    # num_hit_chernoff, num_hit_markov, _ = monte_carlo_prob(num_samples, num_users,
    #                                                              maximum_radius, B, delta, gamma_scale,
    #                                                              mean_lambda, epsilon_error, epsilon_puncture,
    #                                                     True)
    # print(f"num_hit_chernoff: {num_hit_chernoff}, num_hit_markov: {num_hit_markov},"
    #       f" num_samples: {num_samples}")
    # print(f"markov_prob:{num_hit_markov / num_samples}")
    # print(f"chernoff_prob:{num_hit_chernoff / num_samples}")

    epsilon_puncture_min = -3
    epsilon_puncture_max = -1
    num_plot_samples = 99
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plot_k_bounds_epsilon(num_samples, num_users, num_plot_samples, maximum_radius, B, delta,
                          gamma_scale, mean_lambda, epsilon_error, epsilon_puncture_min, epsilon_puncture_max,
                          timestamp=timestamp, stochastic_aus=True, with_latex=True, floored_bound=False)
    plot_k_bounds_lambda(num_samples, num_users, num_plot_samples, maximum_radius, B, delta,
                         gamma_scale, lambda_lower_bound, lambda_upper_bound, epsilon_error, epsilon_puncture,
                         timestamp=timestamp, stochastic_aus=True,
                         with_latex=True, floored_bounds=False)

    # To Do 10.07:
    #   Save data as pkl
    #   Calculate linear regression to check if it has some linear relationship
