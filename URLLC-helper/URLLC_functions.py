import numpy as np
from scipy import optimize


def calc_bus_and_cus(V: np.ndarray, B: float, delta: float, gamma: np.ndarray, Q_inv_epsilon: float,
                     kus: np.ndarray) -> (np.ndarray, np.ndarray):
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


def calc_K_prime_given_mu_star(mu_star: float, bus: np.ndarray, cus: np.ndarray, lambdas: np.ndarray,
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


def substituted_inequality_43(mu, cus, lambdas, epsilon) -> float:
    """
    Calculates the left-hand-side of substituted inequality (43)
    params:
    mu: float, the value of $$\mu^*$$
    cus: Array of floats, where c[u] is the value of $$c_u$$
    lambdas: Array of floats, where lambdas[u] is the value of $$\lambda_u$$
    epsilon: float, the value of $$\epsilon_{\text{puncture}}$$
    """
    return (-2 * mu * np.dot(lambdas, cus * np.exp(2 * cus * mu)) + np.dot(lambdas, np.exp(2 * mu * cus))
            - np.sum(lambdas) - np.log(epsilon))


def solve_mu_star_given_epsilon(cus, lambdas, epsilon) -> float:
    """
    Solves the modified equation (43) by substituting $$K'$$ with equation (41)
    params:
    cus: Array of floats, where c[u] is the value of $$c_u$$
    lambdas: Array of floats, where lambdas[u] is the value of $$lambdas[u]$$
    epsilon: float, value of $$\epsilon_\text{puncture}$$
    return:
    mu: Value of $$\mu^*$$ solved in the inequality.
    """
    func = lambda mu: substituted_inequality_43(mu, cus, lambdas, epsilon)
    derivative = lambda mu: -4 * mu * np.dot(lambdas, cus ** 2 * np.exp(2 * cus * mu))
    mu = optimize.newton(func, 1.0, fprime=derivative)
    return mu


def solve_mu_star(K_prime: float, bus: np.ndarray, cus: np.ndarray, lambdas: np.ndarray) -> float:
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
    func = lambda mu: calc_K_prime_given_mu_star(mu, bus, cus, lambdas, False) - K_prime
    f_prime = lambda mu: calc_K_prime_given_mu_star(mu, bus, cus, lambdas, True)
    mu_star = optimize.newton(func, 0.0, f_prime)
    return mu_star


def find_K_prime(lambdas: np.ndarray, bus: np.ndarray, cus: np.ndarray, epsilon: float) -> int:
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
    mu = solve_mu_star_given_epsilon(cus, lambdas, epsilon)
    K_prime = calc_K_prime_given_mu_star(mu, bus, cus, lambdas, False)
    K_prime_solved = np.floor(K_prime).astype(int)
    mu_star = solve_mu_star(K_prime_solved, bus, cus, lambdas)
    if substituted_inequality_43(mu_star, cus, lambdas, epsilon) > 0:
        K_prime_solved += 1
    return K_prime_solved


if __name__ == '__main__':
    num_users = 100
    bus = np.random.rand(num_users)
    cus = np.random.rand(num_users)
    lambdas = np.random.rand(num_users)
    K_prime = np.random.randint(num_users) + np.floor( 4 * np.sum(bus ** 2)) + 1
    mu_star = solve_mu_star(K_prime=K_prime, bus=bus, cus=cus, lambdas=lambdas)
    print(f"K_prime: {K_prime}, bus: {bus}, cus: {cus}, lambdas: {lambdas}, mu_star: {mu_star}")
    print(f"K_prime(mu_star):{calc_K_prime_given_mu_star(mu_star, bus, cus, lambdas, False)}")
    epsilon = np.random.rand()
    print(f"epsilon: {epsilon}")
    # K_prime_1 = find_K_prime(lambdas=lambdas, bus=bus, cus=cus, epsilon=epsilon)  # Didn't work out!!!
    # print(f"K_prime_1: {K_prime_1}")
    K_prime_1 = find_K_prime(lambdas, bus, cus, epsilon)
    mu_star_1 = solve_mu_star(K_prime_1, bus, cus, lambdas)
    lefthand = substituted_inequality_43(mu_star_1, cus, lambdas, epsilon)
    assert(lefthand <= 0)
    print(f"lefthand: {lefthand}")
    print(f"K_prime_1: {K_prime_1}")
    K_prime_2 = K_prime_1 - 1
    mu_star_2 = solve_mu_star(K_prime_2, bus, cus, lambdas)
    lefthand = substituted_inequality_43(mu_star_2, cus, lambdas, epsilon)
    assert(lefthand > 0)
    print(f"lefthand: {lefthand}, K_prime_2: {K_prime_2}")
    print("Test successful!")