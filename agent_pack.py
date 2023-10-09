from agent import Agent
from utils.metrics import ThoughtDiversity
from typing import Any
import logging
from itertools import combinations


class Pack:
    '''
    three agent instances are composed as one
    '''

    def __init__(self, corpus_path_array: list, main_document_path: str, agent_paths: list, embedding_params: dict = None) -> None:
        '''
        Create Pack
        '''

        if not embedding_params:
            embedding_params = {
                1: ["facebook-dpr-ctx_encoder-multiset-base", 200, 25, 0.9],
                2: ["facebook-dpr-ctx_encoder-multiset-base", 150, 20, 0.7],
                3: ["facebook-dpr-ctx_encoder-multiset-base", 100, 15, 0.5]
            }
        self.agent_names = ["agent_corpus_cot_1",
                            "agent_corpus_cot_2", "agent_corpus_cot_3"]
        self.current_res = None
        self.current_jaccard_indices = None
        self.agent_cot: Agent = Agent(
            self.agent_names[0], main_document_path, True, 1, embedding_params[1])
        self.agent_quant: Agent = Agent(
            self.agent_names[1], main_document_path, True, 2, embedding_params[2])
        self.agent_corpus: Agent = Agent(
            self.agent_names[2], corpus_path_array, True, 3, embedding_params[3])
        self.agent_corpus.path = main_document_path
        self.embeddings = embedding_params

        # Combine paths with agents
        self.agents = zip([self.agent_cot, self.agent_quant,
                          self.agent_corpus], agent_paths)
        # Create or load embeddings
        self.load_agent_docs()

    def load_agent_docs(self):
        '''
        Load or Create embeddings for Agent_Name at DB_Path

        '''
        idx = 0
        for agent, db_path in self.agents:
            # New Instance
            if db_path == 0:
                agent.new_course()

            # Load VectorDB
            else:
                agent.path = db_path
                agent.load_course()
            idx += 1

    # def load_docs(self):
    #     '''
    #     loop through array to load documents
    #     '''

    #     for doc in self.corpus_path_array:
    #         print('loading: ', doc)
    #         self.agent_corpus.chat_bot.add_fractual(doc)
    #     print('meow, I am created')

    def add_docs(self, docs: list):
        '''
        update all agents with list/path of document(s)
        '''

        if isinstance(docs, list):
            print('meow')
            for doc in docs:
                self.agent_corpus.chat_bot.add_new_docs(doc)
                self.agent_cot.chat_bot.add_new_docs(doc)
                self.agent_quant.chat_bot.add_new_docs(doc)
        else:
            print('bork')
            self.agent_corpus.chat_bot.add_fractual(docs)
            self.agent_cot.chat_bot.add_fractual(docs)
            self.agent_quant.chat_bot.add_fractual(docs)
        print('upload successful :)')

    def one_question(self, prompt):
        '''
        one question for pack
        '''
        res = {
            self.agent_cot.name: self.agent_cot.chat_bot.one_question(prompt),
            self.agent_quant.name: self.agent_quant.chat_bot.one_question(prompt),
            self.agent_corpus.name: self.agent_corpus.chat_bot.one_question(
                prompt)
        }
        return res

    def chat(self):
        '''
        Speak with all agents at one time
        '''
        exit_flag = False

        while not exit_flag:
            # Get question
            prompt = input("please ask a question to the pack")
            # Exit Path
            if 'exit' in prompt:
                exit_flag = True

            res = {
                "agent_cot": self.agent_cot.chat_bot.one_question(prompt),
                "agent_quant": self.agent_corpus.chat_bot.one_question(prompt),
                "agent_corpus": self.agent_quant.chat_bot.one_question(prompt)
            }

            logging.info(res)
            print('Here are the collective answers: ')
            print(res)
            self.current_res = res
            self.jaccard_similarity(res)

    def jaccard_similarity(self, res=None):
        '''
        Return all jaccard indices for a given prompt
        '''
        self.current_jaccard_indices = []
        if not res:
            res = self.current_res
        # Step 1: Shingling
        str_a = res['agent_cot']
        str_b = res['agent_corpus']
        str_c = res["agent_quant"]

        # k = min(len(str_a), len(str_b), len(str_c)) - 1]
        k = self.agent_corpus.encoder.chunk_size
        print(f'k: {k} and the string is {str_a}')

        shingles_a = set([str_a[i:i+k] for i in range(len(str_a) - k + 1)])
        shingles_b = set([str_b[i:i+k] for i in range(len(str_b) - k + 1)])
        shingles_c = set([str_c[i:i+k] for i in range(len(str_c) - k + 1)])
        shingles = [shingles_a, shingles_b, shingles_c]
        combos = list(combinations(shingles, 2))

        for combo in combos:
            a, b = combo
            # Step 2: Intersection and Union
            intersection_a_b = a.intersection(b)
            union_a_b = a.union(b)
            # Step 3: Jaccard Index Calculation
            jaccard_index_a_b = len(intersection_a_b) / len(union_a_b)
            self.current_jaccard_indices.append(
                ((jaccard_index_a_b))
            )
        print(self.current_jaccard_indices)
        return self.current_jaccard_indices


if __name__ == '__main__':

    main_doc_path = "documents/kbai_book.pdf"
    corpus_path = ['documents/SND.pdf', 'documents/cot.pdf',
                   'documents/LtoA.pdf', 'documents/wider_deeper.pdf'
                   ]

    # agent_db_paths = ['chroma_db/agent_cot',
    #                   'chroma_db/agent_quant', 'chroma_db/agent_corpus']

    agent_db_paths = [0, 0, 0]
    embedding_params = {
        1: ["facebook-dpr-ctx_encoder-multiset-base", 200, 25, 0.9],
        2: ["facebook-dpr-ctx_encoder-multiset-base", 150, 20, 0.7],
        3: ["facebook-dpr-ctx_encoder-multiset-base", 100, 15, 0.5]
    }

    test_agent = Pack(corpus_path, main_doc_path,
                      agent_db_paths, embedding_params)
    test_agent.add_docs(['documents/SND.pdf'])
    # test_agent.chat()
    metrics = ThoughtDiversity(test_agent)
    print(metrics.monte_carlo_sim())
