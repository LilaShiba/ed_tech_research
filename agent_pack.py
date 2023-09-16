from agent import Agent
from typing import Any


class Pack:
    '''
    three agent instances are composed as one
    '''

    def __init__(self, corpus_path_array: list, main_document_path: str, agent_paths: list) -> None:
        '''
        Create Pack
        '''
        self.agent_cot: Agent = Agent(
            "agent_cot", main_document_path, True, 1)
        self.agent_quant: Agent = Agent(
            "agent_quant", main_document_path, False, 2)
        self.agent_corpus: Agent = Agent(
            "agent_corpus", main_document_path, False, 3)

        self.agents = zip([self.agent_cot, self.agent_quant,
                          self.agent_corpus], agent_paths)
        self.corpus_path_array = corpus_path_array

        self.load_agent_docs()

    def load_agent_docs(self):
        '''
        Loads embeddings for Agent_Name at DB_Path

        '''
        idx = 0
        for agent, db_path in self.agents:

            if db_path == 0:
                agent.new_course()
                if idx == 2:
                    self.load_docs()
            else:
                agent.path = db_path
                agent.load_course()
            idx += 1

    def load_docs(self):
        '''
        loop through array to load documents
        '''

        for doc in self.corpus_path_array:
            print('loading: ', doc)
            self.agent_corpus.chat_bot.add_fractual(doc)
        print('meow, I am created')


if __name__ == '__main__':
    main_doc_path = "documents/Norman-CognitiveEngineering.pdf"
    corpus_path = ['documents/ASD.pdf', 'documents/HilbertSpaceMulti.pdf',
                   'documents/LearnabilityandComplexityofQuantumSamples.pdf', 'documents/meowsmeowing.pdf',
                   'documents/The-order-of-time-Carlo-Rovelli.pdf']

    agent_db_paths = ['chroma_db/agent_cot',
                      'chroma_db/agent_quant', 'chroma_db/agent_corpus']

    test_agent = Pack(corpus_path, main_doc_path, agent_db_paths)
