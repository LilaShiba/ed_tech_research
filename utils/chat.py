""" Start Chat with resources """
import logging
import os

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.sentence_transformer import \
    SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma

logging.basicConfig(filename='output.log', level=logging.INFO)


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class ChatBot:
    """ Chat with resources """

    def __init__(self, agent_instance):
        """ Chat with resources """
        self.agent = agent_instance
        self.name = 'test'
        self.current_question = None
        self.qa_chain = None
        self.retriever = None
        self.vectordb = None
        self.question = None

        self.model = "facebook-dpr-ctx_encoder-multiset-base"

        self.embedding_function = SentenceTransformerEmbeddings(
            model_name=self.model)

        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo", temperature=0.5)

    def enter_chat(self, quest=None):
        """ Start Chat with resources """

        llm = self.llm
        qa_chain = RetrievalQA.from_chain_type(
            llm, retriever=self.agent.encoder.vectordb.as_retriever())

        if quest and self.agent.cot:
            response = qa_chain({"query": quest})
            print(f"{self.name}: {response}")
        else:
            response = qa_chain({"query": quest})
            print(f"{self.name}: {response}")

            # print(f"{self.name}: {response['answer']}")

        exit_flag = False
        while not exit_flag:
            quest = input(
                f"Please ask a question about {self.name} or type 'exit' to end: ")

            if quest.lower() == 'exit':
                exit_flag = True
                print("Goodbye!")
            else:
                response = qa_chain({"query": quest})
                print.pprint(f"{self.name}: {response}")
                logging.info(response['result'])

    def load_chat(self):
        """
        Chat with default agent settings
        """

        print('agent loaded')

        # Enter Chat Stream

        qa_chain = RetrievalQA.from_chain_type(
            self.llm, retriever=self.vectordb.as_retriever())

        if self.agent.cot:
            self.question = "step by step, and one by one explain"

        exit_flag = False
        while not exit_flag:
            quest = input(
                f"Please ask a question about {self.name} or type 'exit' to end: ")
            quest = self.question + quest

            if quest.lower() == 'exit':
                exit_flag = True
                print("Goodbye!")
            else:
                response = qa_chain({"query": quest})
                print(f"{self.name}: {response}")
                logging.info(response['result'])

    def set_agent(self):
        """
        loads vector embeddings for Agent parent class
        """
        self.vectordb = Chroma(
            persist_directory="chroma_db/order-of-time", embedding_function=self.embedding_function)
        self.retriever = self.vectordb.as_retriever()
        self.agent.encoder.vectordb = self.vectordb

    def add_fractual(self, docs):
        """
        add documents to corpus
        """
        self.vectordb.add_documents(docs)
