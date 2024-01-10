"""
Markov Process Tools for ISEN 340
=================================

This module provides various classes and functions that are useful for simulating Markov processes.
Each class has a `summary` method that displays the results in a neat format
and some functions have a `verbose` parameter that does the same.

This module was developed for ISEN 340 (Operations Research II) taught by Mark Lawley at Texas A&M University.

Created By: Christopher Abib \n
Credits: Mark Lawley, James Guse, and Oudah Mortaje
"""
import sklearn

__author__ = 'Christopher Abib'
__contact__ = 'christopher.abib@gmail.com'
__copyright__ = "Copyright 2024, Christopher Abib"
__credits__ = ['Mark Lawley', 'Christopher Abib', 'James Guse', 'Oudah Mortaje']
__date__ = '2022/01/01'
__deprecated__ = False
__email__ = 'christopher.abib@gmail.com'
__license__ = 'MIT'
__version__ = '1.0'

import numpy as np
import pandas as pd
import math
import sympy
import matplotlib.pyplot as plt
from IPython.display import display


class NonConvergenceError(Exception):
    """Exception raised for an operation that does not converge within `n` loops."""
    def __init__(self, loops: int):
        super().__init__(f"Loop did not converge after {loops} iterations.")


def verify_markov_matrix(P: np.matrix, digits=4) -> None:
    """Verifies if P is a valid Markov Chain matrix and raises an error if it is not."""

    # Check that the matrix is a square matrix.
    if P.shape[0] != P.shape[1]:
        raise ValueError("Expected a square probability matrix.")

    # Check that for each row, values sum to 1 when rounded to the `digits` digit.
    for row in P.sum(axis=1):
        if round(row[0, 0], digits) != 1:
            raise ValueError(f"Each row of the probability matrix should add up to 1.\n{P.sum(axis=1)}")


class MarkovChainSteadyState:
    """
    This class computes and visualizes the steady state of a Markov process.

    Args:
        P: A square matrix representing a Markov chain.
        significance: The value that the difference between two iterations must be less than.
        max_iterations: The maximum number of iterations that can be performed.

    Attributes:
        P: From args.
        significance: From args.
        max_iterations: From args.
        pi: The resultant steady state matrix.
        steps: The total number of iterations that were performed.
        deltas: A `list` of the difference between each iteration.
    """
    P: np.matrix
    significance: float
    max_iterations: int

    pi: np.matrix
    steps: int
    deltas: list
    _summary: str

    def __init__(self,
                 P: np.matrix,
                 significance: float = 10e-8,
                 max_iterations: int = 1_000):
        verify_markov_matrix(P)
        self.P = P
        self.significance = significance
        self.max_iterations = max_iterations
        self.compute_steady_state()
        self._summary = f"π = {self.pi}\nsteps = {self.steps}"

    def __repr__(self):
        return self._summary

    def __str__(self):
        return self._summary

    def compute_steady_state(self) -> None:
        """Computes the steady state probabilities."""
        self.deltas = []
        # The P matrix is raised to incrementing exponents
        # until the difference between iterations is smaller than `self.significance`.
        for i in range(1, self.max_iterations):
            self.deltas.append(np.sum(abs(self.P ** (i + 1) - self.P ** i)))

            if self.deltas[-1] < self.significance:
                P_stable = self.P ** i
                self.pi = P_stable[0]
                self.steps = len(self.deltas)
                return

        raise NonConvergenceError(self.max_iterations)

    def summary(self) -> None:
        """Displays a summary of the results."""
        print(self._summary)
        plt.plot(self.deltas)
        plt.xlabel('Steps')
        plt.ylabel('Difference Between Iterations')
        plt.show()


def expected_visits(P: np.matrix,
                    start: int,
                    stop: int,
                    n: int) -> float:
    """
    Calculates the expected number of visits to a state `stop` starting from state `start`
    over the course of `n` state transitions for a given probability matrix.

    Indices for `start` and `stop` begin at zero.

    Args:
        P: The initial probability matrix.
        start: State you start at.
        stop: State you end up at.
        n: Number of state transitions.

    Returns:
        The expected number of visits to state `stop`.
    """
    return sum([P**epochs for epochs in range(1, n+1)])[start, stop]


def first_passage(P: np.matrix,
                  start: int,
                  stop: int,
                  steps: int) -> float:
    """
    Calculates the first passage probability for a given probability matrix.

    Indices for `start` and `stop` begin at zero.

    Args:
        P: The initial probability matrix.
        start: State you start at.
        stop: State you end up at.
        steps: Number of state transitions that will occur.

    Returns:
        The probability that you start at state `start` and arrive at state `stop` for the first time in `steps` steps.
    """
    verify_markov_matrix(P)

    if steps < 1:
        raise ValueError("Steps should be an integer > 0")

    f_passages = []
    for step in range(1, steps + 1):
        if step == 1:
            f_passages.append((P ** step)[start, stop])
        else:
            cum_f_passage = sum([f_passages[k - 1] * (P ** (step - k))[stop, stop] for k in range(1, step)])
            f_passages.append((P ** step)[start, stop] - cum_f_passage)
    return f_passages[-1]


def mean_first_passage(P: np.matrix,
                       start: int,
                       stop: int,
                       verbose: bool = False) -> float:
    """
    Calculates the expected number of state transitions it takes to get from state `start` to state `stop`.

    Indices for `start` and `stop` begin at zero.

    Args:
        P:
        P: The initial probability matrix.
        start: State you start at.
        stop: State you end up at.
        verbose: If `True`, prints out all the intermediate steps of the calculations.

    Returns:
        The mean number of steps required to get from state `start` to state `stop`.
    """

    # mean matrix created from:
    # m_ij = 1 + 𝛴_{k≠j} p_ik * m_kj
    mean_matrix = []
    for i in range(P.shape[0]):
        row = []
        if i == stop:
            continue
        else:
            for j in range(P.shape[0]):
                if j != stop:
                    if i == j:
                        row.append(P[i, j] - 1)
                    else:
                        row.append(P[i, j])
            row.append(-1)
        mean_matrix.append(row)

    lin_eqs = sympy.Matrix(mean_matrix)
    if verbose:
        display(lin_eqs)
    solved_eqs = lin_eqs.rref()[0]
    if verbose:
        display(solved_eqs)

    # Mean Passage Time for start≠stop
    if start != stop:
        # columns list contains the variable associated with each column of the matrix
        columns = list(range(P.shape[0]))
        columns.remove(stop)

        return float(solved_eqs[columns.index(start), -1])

    # Mean Reoccurrence Time
    else:
        m_ki = np.array(solved_eqs[:, -1]).astype(np.float64).flatten()
        if verbose:
            display('m_ki', m_ki)
        P_ik = np.delete(P.A[start, :], start)
        if verbose:
            display('P_ik', P_ik)
        return 1 + sum(m_ki * P_ik)


def random_P_matrix(m: int,
                    seed: int = None,
                    digits: int = None) -> np.matrix:
    """
    Creates a random `np.matrix` of size m x m whose rows sum to 1.

    Args:
        m: The shape of the m x m matrix.
        seed: Sets the seed for `np.random.default_rng(seed)`.
        digits: If provided, restricts each element to `digits` decimal places.

    Returns:
        A valid probability matrix.
    """
    if m < 1:
        raise ValueError("`m` must be a positive integer.")
    if digits is not None:
        if digits > 14:
            raise ValueError(f"Cannot have {digits} digits of precision.")
        elif digits < 1:
            raise ValueError(f"`digits` must be a positive integer.")

    P = np.random.default_rng(seed).uniform(0, 1, m * m).reshape((m, m))
    P = P / P.sum(axis=1, keepdims=True)

    if digits is not None:
        P = np.round(P, digits)
        try:
            verify_markov_matrix(np.matrix(P), digits)
        except ValueError:
            # Deals with cases where after rounding, the sum of a row is not equal to 1.
            for i in range(len(P)):
                dif = 1 - sum(P[i])
                if dif > 0:
                    P[i][P[i].argmin()] += dif
                elif dif < 0:
                    P[i][P[i].argmax()] += dif
            P = np.round(P, digits)
    return np.matrix(P)


def build_lam_matrix(partial_lam_matrix: np.matrix) -> np.matrix:
    """
    Creates a valid lambda matrix when given a matrix that has diagonal values of 0.

    Args:
        partial_lam_matrix: A lambda matrix whose diagonals have yet to be computed.

    Returns:
        A valid lambda matrix.
    """
    m = partial_lam_matrix.shape[0]
    for i in range(m):
        if partial_lam_matrix[i, i] != 0:
            raise ValueError(f"Diagonals of the partial lambda matrix should all be 0.")
        partial_lam_matrix[i, i] = -partial_lam_matrix[i, :].sum()
    return partial_lam_matrix


class CTMC:
    """
    This class models a Continuous Time Markov Chain and solves for its steady state.

    Args:
        lambda_matrix:
        T: Time.
        P: Initial probability matrix.
        iterations: Number of iterations to be performed.
        base: The value of the first index; either 0 or 1.

    Attributes:
        lambda_matrix: From args.
        T: From args.
        P: From args.
        iterations: From args.
        base: From args.
        pi: Calculated steady state value.
        df: A `pandas.Dataframe` that holds records for each iteration of the calculations.
    """
    lambda_matrix: np.matrix
    T: float
    P: np.matrix
    iterations: int
    base: int

    pi: np.matrix
    df: pd.DataFrame
    _summary: str

    def __init__(self,
                 lambda_matrix: np.matrix,
                 T: float,
                 P: np.matrix = None,
                 iterations: int = 1_000,
                 base: int = 1):
        self.lambda_matrix = lambda_matrix
        if P is None:
            self.P = random_P_matrix(m=lambda_matrix.shape[0], seed=0, digits=2)
        else:
            self.P = P

        if lambda_matrix.shape != self.P.shape:
            raise ValueError('`lambda_matrix` and `P` must have the same shape.')

        self.T = T
        self.iterations = iterations
        self.base = base

        self.compute_steady_state()

        self._summary = ''
        for i in range(self.lambda_matrix.shape[0]):
            self._summary += f'The steady state probability for state {i + self.base} is {self.pi[0, i]:.4}\n'
        self._summary = self._summary[:-1]

    def __repr__(self):
        return self._summary

    def __str__(self):
        return self._summary

    def compute_steady_state(self) -> None:
        """Computes the steady state probabilities."""
        m = self.lambda_matrix.shape[0]

        t = np.linspace(0, self.T, self.iterations)
        delta_t = t[1] - t[0]

        matrix_list = list()
        kP = self.P
        for _ in t:
            kP = kP + delta_t * np.dot(kP, self.lambda_matrix)
            matrix_list.append(kP)

        self.pi = kP[0, :]

        p_at_t = []
        for mat in matrix_list:
            p_at_t.append(mat.flatten().tolist()[0])

        columns = []
        for i in range(m):
            for j in range(m):
                columns.append(f'P_{i + self.base}{j + self.base}')

        df = pd.DataFrame(p_at_t,
                          columns=columns)
        df.insert(0, 'Time', t)

        self.df = df

    def summary(self) -> None:
        """Displays a summary of the results."""
        print(self._summary)
        col = 0
        for j in range(self.lambda_matrix.shape[0]):
            column = f'P_{self.base}{j + self.base}'
            plt.plot('Time', column, data=self.df)
            col += 1
        plt.legend()
        plt.show()


class MMC:
    """
    This class calculates the performance measures of an M/M/C queuing model.

    Args:
        lam: Arrive rate.
        mu: Service rate.
        c: Number of servers.
        k: Number of `pi` values to calculate.

    Attributes:
        rho: Mean utilization.
        pi: Probability of j customers present in the system.
        u: Mean utilization
        L: Mean length of the system.
        L_q: Mean length of the queue.
        W: Mean waiting time in the system.
        W_q: Mean waiting time in the queue.
    """
    lam: float
    mu: float
    c: int
    k: int

    rho: float
    pi: np.ndarray
    u: float
    L: float
    L_q: float
    W: float
    W_q: float

    _summary: str

    def __init__(self,
                 lam: float,
                 mu: float,
                 c: int,
                 k: int = 0):
        self.lam, self.mu, self.c, self.k = lam, mu, c, k

        self.rho = lam / (c * mu)
        pi = [0] * (k + 1)
        self.u = self.rho

        part1 = (c * self.rho) ** c / ((1 - self.rho) * math.factorial(c))
        part2 = sum([(c * self.rho) ** i / math.factorial(i) for i in range(c)])
        pi[0] = 1 / (part1 + part2)

        for i in range(1, self.k + 1):
            if c <= i:
                pi[i] = (((c * self.rho) ** i) / ((c ** (i - c)) * math.factorial(c))) * pi[0]
            else:
                pi[i] = (((c * self.rho) ** i) / math.factorial(i)) * pi[0]

        self.L_q = self.rho * pi[0] * ((c * self.rho) ** c) / (((1 - self.rho) ** 2) * math.factorial(c))
        self.L = self.L_q + lam / mu
        self.W_q = self.L_q / lam
        self.W = self.W_q + 1 / mu

        self.pi = np.array(pi)

        if len(pi) > 1:
            pi_k_text = f"π_{self.k}: {self.pi[self.k - 1]:.4f}\n"
        else:
            pi_k_text = ""

        self._summary = f"ρ: {self.rho:.4f}\n" \
                        f"π_0: {self.pi[0]:.4f}\n" \
                        f"{pi_k_text}" \
                        f"u: {self.u:.4f}\n" \
                        f"L: {self.L:.4f}\n" \
                        f"L_q: {self.L_q:.4f}\n" \
                        f"W: {self.W:.4f}\n" \
                        f"W_q: {self.W_q:.4f}"

    def __repr__(self):
        return self._summary

    def __str__(self):
        return self._summary

    def summary(self) -> None:
        """Displays a summary of the results."""
        print(self._summary)

    def to_dataframe(self) -> pd.DataFrame:
        """Returns the performance measures as a dataframe"""
        items = {k: v for k, v in self.__dict__.items() if k not in ('k', 'u', 'pi', '_summary')}
        items['pi_0'] = self.pi[0]
        return pd.DataFrame(items, index=[0])


if __name__ == '__main__':
    mmc = MMC(lam=.4, mu=.5, c=2, k=9)
    print(mmc.to_dataframe())
