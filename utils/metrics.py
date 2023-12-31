from typing import List, Any
import numpy as np
from collections import Counter
import logging
from scipy.stats import wasserstein_distance


class ThoughtDiversity:
    """
    Class to measure the diversity of thought in an AI agent composed of several agents.
    """

    def __init__(self, pack: Any) -> None:
        """
        Initializes the D_Metrics instance with an Agent_Pack instance.
        Provides a framework to test for diversity of thought.

        Parameters:
        agent_instance (Any): An instance of Agent_Pack.
        """

        self.pack = pack
        self.scores = []
        self.snd_scores = []
        self.jaccard_indexs = []
        self.current_mcs_samples = []
        self.shannon_entropy_scores = []
        self.true_diversity_scores = []
        self.wasserstein_metrics = []
        self.ginis = []

    def monte_carlo_sim(self, question="", rounds: int = 5) -> List[Any]:
        """
        Run a Monte Carlo simulation.

        Parameters:
        rounds (int): The number of rounds to run the simulation.
        agent (Any): The agent to be tested.
        test_params (List[Any]): A list of one_questions prompts to test.

        Returns:
        List[Any]: The results of the Monte Carlo simulation.
        """
        res = []

        for _ in range(rounds):
            round_res = self.pack.one_question(question)
            acot = round_res[self.pack.agent_names[0]]
            acor = round_res[self.pack.agent_names[1]]
            acor2 = round_res[self.pack.agent_names[2]]

            if round_res:
                res.append(str((acot, acor, acor2)))
                a_cot_vector = self.prob_vectors(acot)
                a_cor_vector = self.prob_vectors(acor)
                a_cor2_vector = self.prob_vectors(acor2)

                self.shannon_entropy_scores.append(
                    ('cot', self.shannon_entropy(a_cot_vector))
                )

                self.true_diversity_scores.append(
                    ('cot', self.true_diversity(a_cot_vector))
                )

                self.shannon_entropy_scores.append(
                    ('cot', self.shannon_entropy(a_cor2_vector))
                )

                self.true_diversity_scores.append(
                    ('cot', self.true_diversity(a_cor2_vector))
                )

                self.shannon_entropy_scores.append(
                    ('cor', self.shannon_entropy(a_cor_vector))
                )

                self.true_diversity_scores.append(
                    ('cor', self.true_diversity(a_cor_vector))
                )

                self.wasserstein_metrics.append((1, self.wasserstein(
                    a_cot_vector,  a_cor_vector)))
                self.wasserstein_metrics.append((2, self.wasserstein(
                    a_cot_vector,  a_cor2_vector)))
                self.wasserstein_metrics.append((3, self.wasserstein(
                    a_cor_vector,  a_cor2_vector)))

        res = [self.shannon_entropy_scores,
               self.true_diversity_scores, self.wasserstein_metrics]
        logging.info(res)
        return res

    def prob_vectors(self, vector: List[str]) -> List[int]:
        '''
        return prob_vector
        '''

        joined_strings = ' '.join(vector)
        # print('getting metrics H & D')
        # Split the single string into words
        words = joined_strings.split()
        # Count the occurrences of each word
        word_counts = Counter(words)

        counts = list(word_counts.values())
        m = sum(counts)/len(counts)
        prob_vector = [count/m for count in counts]
        return prob_vector

    def shannon_entropy(self, counts: List[int]) -> float:
        """
        Calculates Shannon Entropy (H) of a dataset.
        Formula: H = -sum(p_i * log(p_i))

        Parameters:
        counts (List[int]): A list of counts of occurrences of each species or category.

        Returns:
        float: Shannon Entropy of the dataset.
        """
        total_counts = sum(counts)
        proportions = [count / total_counts for count in counts]
        # log base e is used here
        entropy = -sum(p * np.log(p) if p > 0 else 0 for p in proportions)
        return entropy

    def true_diversity(self, counts: List[int]) -> float:
        """
        Calculates True Diversity (D) of a dataset using Shannon Entropy (H).
        Formula: D = exp(H)

        Parameters:
        counts (List[int]): A list of counts of occurrences of each species or category.

        Returns:
        float: True Diversity of the dataset.
        """
        entropy = self.shannon_entropy(counts)
        diversity = np.exp(entropy)
        return diversity

    def snd(self, questions, monte_Carlo_Simulation=False):
        '''
        System Neural Diversity
        (Bonttni: 2023)

        Measures the diversity of an ensamble agent
        through combonatrics of pair-wise agent distance
        Parameters:
        questions:List of strings

        Returns:
        snd results of questions 

        Options:
        monte_Carlo_Simulation: Bool Default: False
        '''

    def wasserstein(self, p1: List[int], p2: List[int]) -> float:
        '''
        returns the Wasserstein Metric between two numpy probaitly vectors 

        '''
        # Compute the 1st Wasserstein distance between the two distributions
        w_dist = wasserstein_distance(p1, p2)
        self.wasserstein_metrics.append(w_dist)
        print(f"Wasserstein Distance: {w_dist}")
# Example Usage:
# Assuming Agent_Pack is defined and has the necessary attributes
# agent_pack_instance = Agent_Pack()
# d_metrics_instance = D_Metrics(agent_pack_instance)
# counts = [10, 20, 30, 40]
# entropy = d_metrics_instance.shannon_entropy(counts)
# diversity = d_metrics_instance.true_diversity(counts)
# print(f'Shannon Entropy: {entropy}')
# print(f'True Diversity: {diversity}')
