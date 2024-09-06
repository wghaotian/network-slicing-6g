import numpy as np
from scipy import optimize, special
import matplotlib.pyplot as plt


def inv_Q(x: float) -> float:
    """
    Inverse of Gaussian Q Function
    :param x: x in Q^{-1}(x)
    :return: Q^{-1}(x)
    """
    return np.sqrt(2) * special.erfinv(1 - 2 * x)


def calc_V(gamma: np.ndarray) -> np.ndarray:
    """
    Calculate V[u] given gamma[u]
    :param gamma: SINR of users, gamma[u] is $$\gamma_u$$.
    """
    return (1 - 1 / (1 - gamma) ** 2) * np.log2(np.e) ** 2


def calc_bus_and_cus(B: float, delta: float, gamma: np.ndarray, epsilon: float,
                     kus: np.ndarray = None) -> (np.ndarray, np.ndarray):
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
    bus = np.sqrt(Vs) * inv_q_epsilon / (2 * np.sqrt(B * delta) * np.log(1 + gammas))
    cus = 1 / (B * delta * np.log(1 + gammas))
    return bus, cus


def calc_K_prime_given_mu_star(mu_star: float, bus: np.ndarray, cus: np.ndarray, lambdas: np.ndarray,
                               derivative: bool) -> float:
    """
    Calculates the $$K'$$ for the given $$\mu^*$$
    :param mu_star: Value of the $$\mu^*$$
    :param bus: Array of floats, where b[u] is the value of $$b_u$$
    :param cus: Array of floats, where c[u] is the value of $$c_u$$
    :param derivative: Boolean, whether to calculate the derivative
    :return K_prime: Value of $$K'$$
    :return dK_prime: Value of $$\frac{d\mu^*}{dK'}$$
    """
    if derivative:
        dK_prime = 4 * np.sum(lambdas * (cus ** 2) * np.exp(2 * mu_star * cus))
        return dK_prime
    else:
        K_prime = 4 * np.sum(bus ** 2) + 2 * np.sum(lambdas * cus * np.exp(2 * mu_star * cus))
        return K_prime


def substituted_inequality_43(mu, cus, lambdas, epsilon) -> float:
    """
    Calculates the left-hand-side of substituted inequality (43)
    :param mu: float, the value of $$\mu^*$$
    :param cus: Array of floats, where c[u] is the value of $$c_u$$
    :param lambdas: Array of floats, where lambdas[u] is the value of $$\lambda_u$$
    :param epsilon: float, the value of $$\epsilon_{\text{puncture}}$$
    """
    return (-2 * mu * np.dot(lambdas, cus * np.exp(2 * cus * mu)) + np.dot(lambdas, np.exp(2 * mu * cus))
            - np.sum(lambdas) - np.log(epsilon))


def solve_mu_star_given_epsilon(cus, lambdas, epsilon) -> float:
    """
    Solves the modified equation (43) by substituting $$K'$$ with equation (41)
    :param cus: Array of floats, where c[u] is the value of $$c_u$$
    :param lambdas: Array of floats, where lambdas[u] is the value of $$lambdas[u]$$
    :param epsilon: float, value of $$\epsilon_\text{puncture}$$
    :return mu: Value of $$\mu^*$$ solved in the inequality.
    """
    num_users = cus.shape[0]
    func = lambda mu: substituted_inequality_43(mu, cus, lambdas, epsilon)
    derivative = lambda mu: -4 * mu * np.dot(lambdas, cus ** 2 * np.exp(2 * cus * mu))
    mu = optimize.newton(func, 1 / num_users, fprime=derivative)
    return mu


def solve_mu_star(K_prime: float, bus: np.ndarray, cus: np.ndarray, lambdas: np.ndarray) -> float:
    """
    Solves the equation (41) where
        $$ K\' = 4 \sum_{u \in \mathcal{U} b_u^2 + 2 \sum_{u \in \mathcal{U} \lambda_u c_u e^{2 c_u \mu^*} $$
    :param K_prime: Value of $$K'$$
    :param bus: Array of floats, where bus[u] is the value of $$b_u$$
    :param cus: Array of floats, where cus[u] is the value of $$c_u$$
    :param lambdas: Array of floats, where lambdas[u] is the value of $$\lambda_u$$

    :return mu_star: Value of $$\mu^*$$ solved from the equation'
    """
    func = lambda mu: calc_K_prime_given_mu_star(mu, bus, cus, lambdas, False) - K_prime
    f_prime = lambda mu: calc_K_prime_given_mu_star(mu, bus, cus, lambdas, True)
    mu_star = optimize.newton(func, 0.0, f_prime)
    return mu_star


def find_K_prime(lambdas: np.ndarray, bus: np.ndarray, cus: np.ndarray, epsilon: float) -> int:
    """
    Finds the smallest integer of $$K'$$ such that the inequality (43) holds where
        $$\exp(-\mu^* K' - \sum_{u \in \mathcal{U}} \lambda_u + 4 \mu^* \sum_{u \in \mathcal{U}} b_u^2 +
            \sum_{u \in \mathcal{U}}\lambda_u e^{2 \mu^* c_u}) \le \epsilon$$
    :param lambdas: Array of floats, where lambdas[u] is the value of $$\lambda_u$$
    :param bus: Array of floats, where bus[u] is the value of $$b_u$$
    :param cus: Array of floats, where cus[u] is the value of $$c_u$$
    :param epsilon: Value of $$\epsilon_\mathrm{puncture}$$

    :return K_prime: the smallest integer of $$K'$$ such that the inequality (43) holds
    """
    mu = solve_mu_star_given_epsilon(cus, lambdas, epsilon)
    K_prime = calc_K_prime_given_mu_star(mu, bus, cus, lambdas, False)
    K_prime_solved = np.floor(K_prime).astype(int)
    mu_star = solve_mu_star(K_prime_solved, bus, cus, lambdas)
    if substituted_inequality_43(mu_star, cus, lambdas, epsilon) > 0:
        K_prime_solved += 1
    return K_prime_solved

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


def markov_ineq_lower_bound(bus: np.ndarray, cus: np.ndarray, epsilon: float, lambdas: np.ndarray) -> float:
    """
    Lower bound of K given by Markov inequality
    :param bus: Array of floats, where bus[u] is the value of $$b_u$$
    :param cus: Array of floats, where cus[u] is the value of $$c_u$$
    :param epsilon: Value of $$\epsilon_\mathrm{puncture}$$
    :param lambdas: Array of floats, where lambdas[u] is the value of $$\lambda_u$$
    """
    return (4 * np.dot(bus ** 2, 1 - np.exp(lambdas)) + 2 * np.dot(cus, lambdas)) / epsilon  # lambda or -lambda?


def sample_data(bus: np.ndarray, cus: np.ndarray, epsilon: float, mean_lambda: float) -> (int, float):
    """
    Sample data for plot 1
    :param bus: Array of floats, where bus[u] is the value of $$b_u$$
    :param cus: Array of floats, where cus[u] is the value of $$c_u$$
    :param epsilon: Value of $$\epsilon_\mathrm{puncture}$$
    :param mean_lambda: mean value of lambda\
    """
    num_users = bus.shape[0]
    lambdas = np.random.poisson(mean_lambda, num_users)
    K_prime = find_K_prime(lambdas, bus, cus, epsilon)
    K_markov = markov_ineq_lower_bound(bus, cus, epsilon, lambdas)
    return K_prime, K_markov


def plot_1(epsilon: float, num_users: int, B:float, delta:float,
           lambda_lower_bound: float, lambda_upper_bound: float, num_samples: int, scale_sinr: float = 1.0):
    K_primes = np.array([])
    K_markovs = np.array([])
    gammas = np.random.rayleigh(scale=scale_sinr, size=num_users)
    bus, cus = calc_bus_and_cus(B, delta, gammas, epsilon)
    # print(f"bus: {bus}")
    # print(f"cus: {cus}")
    mean_lambdas = np.linspace(lambda_lower_bound, lambda_upper_bound, num_samples)
    for mean_lambda in mean_lambdas:
        K_prime, K_markov = sample_data(bus, cus, epsilon, mean_lambda)
        K_primes = np.append(K_primes, K_prime)
        K_markovs = np.append(K_markovs, K_markov)
    plt.rcParams['text.usetex'] = True
    plt.plot(mean_lambdas, K_primes, label=r"$K'$")
    plt.plot(mean_lambdas, K_markovs, label=r'$K_{min}$')
    plt.xlabel(r"$\bar{\lambda}$")
    plt.title(rf"$\epsilon = {epsilon}, \gamma \sim Rayleigh({scale_sinr})$")
    plt.legend()
    plt.show()


def plot_2(num_users: int, B:float, delta:float, mean_lambda:float, epsilon_lower_bound:float, epsilon_upper_bound:float
           , num_samples:int, scale_sinr: float = 1.0):
    K_primes = np.array([])
    K_markovs = np.array([])
    gammas = np.random.rayleigh(scale=scale_sinr, size=num_users)
    epsilons = np.linspace(epsilon_lower_bound, epsilon_upper_bound, num_samples)
    for epsilon in epsilons:
        bus, cus = calc_bus_and_cus(B, delta, gammas, epsilon)
        K_prime, K_markov = sample_data(bus, cus, epsilon, mean_lambda)
        K_primes = np.append(K_primes, K_prime)
        K_markovs = np.append(K_markovs, K_markov)

    plt.rcParams['text.usetex'] = True
    plt.plot(epsilons, K_primes, label=r"$K'$")
    plt.plot(epsilons, K_markovs, label=r'$K_{min}$')
    plt.xlabel(r"$\epsilon$")
    plt.title(r"$\bar{\lambda} = "f"{mean_lambda}"r", \gamma \sim \mathrm{Rayleigh}("f"{scale_sinr})$")
    plt.legend()
    plt.show()



if __name__ == '__main__':
    num_users = 1000
    # bus = np.random.rand(num_users)
    # cus = np.random.rand(num_users)
    # lambdas = np.random.rand(num_users)
    # K_prime = np.random.randint(num_users) + np.floor( 4 * np.sum(bus ** 2)) + 1
    # mu_star = solve_mu_star(K_prime=K_prime, bus=bus, cus=cus, lambdas=lambdas)
    # # print(f"bus: {bus}, cus: {cus}")
    # print(f"K_prime: {K_prime}, lambdas: {lambdas}, mu_star: {mu_star}")
    # print(f"K_prime(mu_star):{calc_K_prime_given_mu_star(mu_star, bus, cus, lambdas, False)}")
    # epsilon = np.random.rand()
    # print(f"epsilon: {epsilon}")
    # # K_prime_1 = find_K_prime(lambdas=lambdas, bus=bus, cus=cus, epsilon=epsilon)  # Didn't work out!!!
    # # print(f"K_prime_1: {K_prime_1}")
    # K_prime_1 = find_K_prime(lambdas, bus, cus, epsilon)
    # mu_star_1 = solve_mu_star(K_prime_1, bus, cus, lambdas)
    # lefthand = substituted_inequality_43(mu_star_1, cus, lambdas, epsilon)
    # assert(lefthand <= 0)
    # print(f"lefthand: {lefthand}")
    # print(f"K_prime_1: {K_prime_1}")
    # K_prime_2 = K_prime_1 - 1
    # mu_star_2 = solve_mu_star(K_prime_2, bus, cus, lambdas)
    # lefthand = substituted_inequality_43(mu_star_2, cus, lambdas, epsilon)
    # assert(lefthand > 0)
    # print(f"lefthand: {lefthand}, K_prime_2: {K_prime_2}")
    # print("Test successful!")
    B = 1
    delta = 1e-3
    epsilon = 1e-1
    lambda_lower_bound = 1
    lambda_upper_bound = 10
    num_samples = 100
    scale_sinr = 100
    plot_1(epsilon, num_users, B, delta, lambda_lower_bound, lambda_upper_bound, num_samples, scale_sinr)
    epsilon_lower_bound = 1e-3
    epsilon_upper_bound = 1e-1
    mean_lambda = 6
    plot_2(num_users, B, delta, mean_lambda, epsilon_lower_bound, epsilon_upper_bound, num_samples, scale_sinr)