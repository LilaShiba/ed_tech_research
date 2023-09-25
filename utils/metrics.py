from typing import List, Any
import numpy as np
from collections import Counter


class ThoughtDiversity:
    """
    Class to measure the diversity of thought in an AI agent composed of several agents.
    """

    def __init__(self, agent_instance: Any) -> None:
        """
        Initializes the D_Metrics instance with an Agent_Pack instance.
        Provides a framework to test for diversity of thought.

        Parameters:
        agent_instance (Any): An instance of Agent_Pack.
        """
        self.agent_a = agent_instance.agent_cot
        self.agent_b = agent_instance.agent_quant
        self.agent_c = agent_instance.agent_corpus

        self.agents = [self.agent_a, self.agent_b, self.agent_c]
        self.scores = []
        self.snd_scores = []
        self.jaccard_indexs = []
        self.current_mcs_samples = []
        self.shannon_entropy_scores = []
        self.true_diversity_scores = []

    def monte_carlo_sim(self, rounds: int = 10, agent: Any = None,
                        test_params: List[Any] = None) -> List[Any]:
        """
        Run a Monte Carlo simulation.

        Parameters:
        rounds (int): The number of rounds to run the simulation.
        agent (Any): The agent to be tested.
        test_params (List[Any]): A list of one_questions prompts to test.

        Returns:
        List[Any]: The results of the Monte Carlo simulation.
        """
        if not test_params:
            test_params = [self.agent_c.chat_bot.one_question(
                "what is a foundational model"), self.agent_a.chat_bot.one_question("what is a foundational model")]
        res = [self.current_mcs_samples.append(param)
               for r in range(rounds) for param in test_params]

        # Join all strings into a single string, separating them by space
        joined_strings = ' '.join(res)

        # Split the single string into words
        words = joined_strings.split()
        # Count the occurrences of each word
        word_counts = Counter(words)

        counts = list(word_counts.values())
        h = self.shannon_entropy(counts)
        self.shannon_entropy_scores.append(h)
        return self.shannon_entropy_scores

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

# Example Usage:
# Assuming Agent_Pack is defined and has the necessary attributes
# agent_pack_instance = Agent_Pack()
# d_metrics_instance = D_Metrics(agent_pack_instance)
# counts = [10, 20, 30, 40]
# entropy = d_metrics_instance.shannon_entropy(counts)
# diversity = d_metrics_instance.true_diversity(counts)
# print(f'Shannon Entropy: {entropy}')
# print(f'True Diversity: {diversity}')
