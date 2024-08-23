import numpy as np
from scipy import optimize


def calc_bus_and_cus(V: np.array, B: float, delta: float, gamma: np.array, Q_inv_epsilon: float,
                     kus: np.array) -> (np.array, np.array):
    """
    Calculates the b value for the given $$V, B, \delta, \gamma and Q^{-1}(\epsilon)$$
    params:
    V: Array of floats, where V[u][k] represents the $$V_{u, k} = (1 - \frac{1}{(1 - \gamma_{u, k})^2} * \log_2^2(e))$$
    B: Value of the B
    delta: Value of the delta
    gamma: a 2-dimensional array, where gamma[u][k] is the value of the $$\gamma_{u, k}^{(t)}$$
    Q_inv_epsilon: Precalculated value of $$Q^{-1}(\epsilon)$$
    kus: Array of integers, where kus[u] is the RB $$k$$ associated to user $$u$$
    return:
    bus: Array of floats, where b[u] is the value of $$b_u$$
    cus: Array of floats, where c[u] is the value of $$c_u$$
    """
    gammas = gamma[:, kus].diagonal()
    Vs = V[:, kus].diagonal()
    bus = np.sqrt(Vs) * Q_inv_epsilon / (2 * np.sqrt(B * delta) * np.log(1 + gammas))
    cus = 1 / (B * delta * np.log(1 + gammas))
    return bus, cus


def calc_K_prime_given_mu_star(mu_star: float, bus: np.array, cus: np.array, lambdas: np.array,
                               derivative: bool) -> float:
    """
    Calculates the $$K'$$ for the given $$\mu^*$$
    params:
    mu_star: Value of the $$\mu^*$$
    bus: Array of floats, where b[u] is the value of $$b_u$$
    cus: Array of floats, where c[u] is the value of $$c_u$$
    derivative: Boolean, whether to calculate the derivative
    return:
    K_prime: Value of $$K'$$
    dK_prime: Value of $$\frac{d\mu^*}{dK'}$$
    """
    if derivative:
        dK_prime = 4 * np.sum(lambdas * (cus ** 2) * np.exp(2 * mu_star * cus))
        return dK_prime
    else:
        K_prime = 4 * np.sum(bus ** 2) + 2 * np.sum(lambdas * cus * np.exp(2 * mu_star * cus))
        return K_prime


def solve_mu_star(K_prime: float, bus: np.array, cus: np.array, lambdas: np.array) -> float:
    """
    Solves the equation (41) where
        $$ K\' = 4 \sum_{u \in \mathcal{U} b_u^2 + 2 \sum_{u \in \mathcal{U} \lambda_u c_u e^{2 c_u \mu^*} $$
    params:
    K_prime: Value of $$K'$$
    bus: Array of floats, where bus[u] is the value of $$b_u$$
    cus: Array of floats, where cus[u] is the value of $$c_u$$
    lambdas: Array of floats, where lambdas[u] is the value of $$\lambda_u$$

    return:
    mu_star: Value of $$\mu^*$$ solved from the equation'
    """
    f_K_prime = lambda mu: calc_K_prime_given_mu_star(mu, bus, cus, lambdas, False) - K_prime
    f_dK_prime = lambda mu: calc_K_prime_given_mu_star(mu, bus, cus, lambdas, True)
    mu_star = optimize.newton(f_K_prime, 0.0, f_dK_prime)
    return mu_star


def find_K_prime(lambdas: np.array, bus: np.array, cus: np.array, epsilon: float) -> int:
    """
    Finds the smallest integer of $$K'$$ such that the inequality (43) holds where
        $$\exp(-\mu^* K' - \sum_{u \in \mathcal{U}} \lambda_u + 4 \mu^* \sum_{u \in \mathcal{U}} b_u^2 +
            \sum_{u \in \mathcal{U}}\lambda_u e^{2 \mu^* c_u}) \le \epsilon$$
    params:
    lambdas: Array of floats, where lambdas[u] is the value of $$\lambda_u$$
    bus: Array of floats, where bus[u] is the value of $$b_u$$
    cus: Array of floats, where cus[u] is the value of $$c_u$$

    return:
    K_prime: the smallest integer of $$K'$$ such that the inequality (43) holds
    """
    def func_lefthand_side(K):
        mu = solve_mu_star(K, bus, cus, lambdas)
        return (-mu * K - np.sum(lambdas) + 4 * mu * np.sum(bus ** 2)
                + np.sum(lambdas * np.exp(2 * mu * cus)) - np.log(epsilon))
    f = lambda x: func_lefthand_side(x)
    df_dx = lambda x: -solve_mu_star(x, bus, cus, lambdas)
    K_prime = np.floor(optimize.newton(f, 0.0, df_dx))
    if f(K_prime) > 0:
        K_prime -= 1
    return K_prime


if __name__ == '__main__':
    bus = np.random.rand(10)
    cus = np.random.rand(10)
    lambdas = np.random.rand(10)
    K_prime = np.random.randint(10) + 4 * np.sum(bus ** 2)
    mu_star = solve_mu_star(K_prime=K_prime, bus=bus, cus=cus, lambdas=lambdas)
    print(f"K_prime: {K_prime}, bus: {bus}, cus: {cus}, lambdas: {lambdas}, mu_star: {mu_star}")
    print(f"K_prime(mu_star):{calc_K_prime_given_mu_star(mu_star, bus, cus, lambdas, False)}")
    epsilon = np.random.rand()
    print(f"epsilon: {epsilon}")
    K_prime_1 = find_K_prime(lambdas=lambdas, bus=bus, cus=cus, epsilon=epsilon) # Didn't work out!!!
    print(f"K_prime_1: {K_prime_1}")
