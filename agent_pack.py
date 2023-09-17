from agent import Agent
from typing import Any
import logging


class Pack:
    '''
    three agent instances are composed as one
    '''

    def __init__(self, corpus_path_array: list, main_document_path: str, agent_paths: list) -> None:
        '''
        Create Pack
        '''
        self.corpus_path_array = corpus_path_array

        self.agent_cot: Agent = Agent(
            "agent_cot", main_document_path, True, 1)
        self.agent_quant: Agent = Agent(
            "agent_quant", main_document_path, False, 2)
        self.agent_corpus: Agent = Agent(
            "agent_corpus", corpus_path_array, False, 3)

        # Combine paths with agents
        self.agents = zip([self.agent_cot, self.agent_quant,
                          self.agent_corpus], agent_paths)
        self.main_document_path = main_doc_path
        # Create or load embeddings
        self.load_agent_docs()

    def load_agent_docs(self):
        '''
        Load or Create embeddings for Agent_Name at DB_Path

        '''
        for agent, db_path in self.agents:
            # New Instance
            if db_path == 0:
                agent.new_course()

            # Load VectorDB
            else:
                print('loading: ', self.main_document_path)
                agent.path = db_path
                agent.load_course()
        print('loading complete')

    def load_docs(self):
        '''
        loop through array to load documents
        '''

        for doc in self.corpus_path_array:
            print('loading: ', doc)
            self.agent_corpus.chat_bot.add_fractual(doc)
        print('meow, I am created')

    def add_docs(self, documents: list):
        '''
        update all agents with list of documents
        '''

        for doc in documents:
            print('adding: ', doc)
            self.agent_corpus.chat_bot.add_fractual(doc)
            self.agent_cot.chat_bot.add_fractual(doc)
            self.agent_quant.chat_bot.add_fractual(doc)
        print('upload successful :)')

    def chat(self):
        '''
        Speak with all agents at one time
        '''
        exit_flag = False
        print('starting chat')
        while not exit_flag:
            # Get question
            prompt = input("please ask a question to the pack")
            # Exit Path
            if prompt == 'exit':
                exit_flag = True

            res = {
                "agent_cot": self.agent_cot.chat_bot.one_question(prompt),
                "agent_quant": self.agent_corpus.chat_bot.one_question(prompt),
                "agent_corpus": self.agent_quant.chat_bot.one_question(prompt)
            }

        logging.info(res)
        print(res)


if __name__ == '__main__':
    main_doc_path = "documents/ASD.pdf"
    corpus_path = ['documents/ASD.pdf', 'documents/HilbertSpaceMulti.pdf'
                   ]

    agent_db_paths = ['chroma_db/agent_cot', 'chroma_db/agent_quant', 0]

    test_agent = Pack(corpus_path, main_doc_path, agent_db_paths)
    # test_agent.add_docs([test_agent.main_document_path])
    test_agent.chat()
