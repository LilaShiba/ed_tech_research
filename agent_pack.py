from agent import Agent
from typing import Any


class Pack:
    '''
    three agent instances are composed as one
    '''

    def __init__(self, corpus_path_array: list, main_document_path: str) -> None:
        '''
        Create Pack
        '''
        self.agent_cot: Agent = Agent("agent_cot", main_document_path, True, 1)
        self.agent_quant: Agent = Agent(
            "agent_quant", main_document_path, False, 2)
        self.agent_corpus: Agent = Agent(
            "agent_corpus", main_document_path, False, 3)

        print('waking up agent_cot')
        self.agent_cot.new_course()
        print('waking up agent_corpus')
        self.agent_corpus.new_course()
        self.agent_corpus.path = "chroma_db/"+self.agent_corpus.name

        for doc in corpus_path_array:
            print('loading: ', doc)
            db_path = '/chroma_db/mem_bot'

            self.agent_corpus.chat_bot.add_fractual(doc)
        print('umm, the quantum agent just appeared ')


if __name__ == '__main__':
    main_path = "documents/VisualizingQuantumCircuitProbability.pdf"
    corpus_path = ['documents/ASD.pdf', 'documents/HilbertSpaceMulti.pdf',
                   'documents/LearnabilityandComplexityofQuantumSamples.pdf', 'documents/meowsmeowing.pdf',
                   'documents/The-order-of-time-Carlo-Rovelli.pdf']

    test_agent = Pack(corpus_path, main_path)
