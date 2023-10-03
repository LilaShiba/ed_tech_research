import numpy as np
import matplotlib.pyplot as plt


class SIRSModel:
    def __init__(self, beta: float, gamma: float, xi: float, N: int, I0: int) -> None:
        """
        Initializes the SIRS model with the given parameters.

        Parameters:
        beta (float): Transmission rate.
        gamma (float): Recovery rate.
        xi (float): Loss of immunity rate.
        N (int): Total population.
        I0 (int): Initial number of infected individuals.
        """

        self.beta = beta
        self.gamma = gamma
        self.xi = xi
        self.N = N
        self.I = I0
        self.S = N - I0
        self.R = 0

    def step(self) -> None:
        """
        Update the S, I, R values for one time step based on the transition rates.
        """
        new_infected = (self.beta * self.S * self.I) / self.N
        new_recovered = self.gamma * self.I
        new_susceptible = self.xi * self.R

        self.S += new_susceptible - new_infected
        self.I += new_infected - new_recovered
        self.R += new_recovered - new_susceptible

    def simulate(self, steps: int) -> np.ndarray:
        """
        Simulate the model over a number of time steps.

        Parameters:
        steps (int): The number of time steps to simulate.

        Returns:
        np.ndarray: A history of S, I, R values over time.
        """

        history = np.zeros((steps, 3))
        for t in range(steps):
            history[t] = [self.S, self.I, self.R]
            self.step()
        return history

    def plot_results(self, history: np.ndarray) -> None:
        '''
        Plot the predicted trends
        '''
        plt.figure(figsize=(10, 6))
        plt.plot(history[:, 0], label='Susceptible')
        plt.plot(history[:, 1], label='Infected')
        plt.plot(history[:, 2], label='Recovered')
        plt.legend()
        plt.title('SIRS Model Simulation')
        plt.xlabel('Time (days)')
        plt.ylabel('Population')
        plt.grid(True)
        plt.show()


# Usage:
'''
Disease,Transmission Rate (beta),
Recovery Rate (gamma),
Loss of Immunity Rate (xi)
COVID-19,1.5 - 3.5,1/14 - 1/7,Unknown
Influenza,1.3 - 1.8,1/5 - 1/3,Varies
RSV,Unknown,Varies,Unknown

'''

if __name__ == "__main__":

    # Transmission Rate
    beta = 1.5
    # Recovery Rate
    gamma = 1/5
    # Immunity Loss Rate
    xi = 0.05
    N = 1000
    I0 = 5

    model = SIRSModel(beta, gamma, xi, N, I0)
    history = model.simulate(100)
    model.plot_results(history)
