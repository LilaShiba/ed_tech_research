''''
Class to measure the diveristy of thought
in an AI agent composed of several Agents
'''


class D_Metrics():
    '''
    MCS: self.mcs() -> monte carlo sampling

    '''

    def __init__(self, agent_instance) -> None:
        '''
        Takes in Agent_Pack instance
        Provides framework to test for diversity
        of thought :)
        '''
        self.agent_a = agent_instance.agent_cot
        self.agent_b = agent_instance.agent_corpus
        self.agent_c = agent_instance.agent_corpus

        self.agents = [self.agent_a,
                       self.agent_b,
                       self.agent_c]
        self.scores = []
        self.snd_scores = []
        self.jaccard_indexs = []
        self.current_mcs_samples = []

    def mcs(self, rounds=1000, agent=None, test_params=[]):
        '''
        Run a monte carlo simulation :) 
        '''

        return [self.current_mcs_samples.append(agent.param)
                for _ in range(rounds) for param in test_params]
