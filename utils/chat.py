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
        self.question = None

        self.model = "facebook-dpr-ctx_encoder-multiset-base"

        self.embedding_function = SentenceTransformerEmbeddings(
            model_name=self.model)

        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo", temperature=0.9)

    def load_chat(self):
        """
        Chat with default agent settings
        """

        print('agent loaded')

        # Enter Chat Stream
        self.question = ''
        qa_chain = RetrievalQA.from_chain_type(
            self.llm, retriever=self.agent.vectordb.as_retriever())

        if self.agent.cot_name == 1:
            self.question = "step by step, and one by one explain: "

        exit_flag = False
        while not exit_flag:
            quest = input(
                f"Please ask a question about {self.name} or type 'exit' to end: ")
            quest = self.question + quest

            if quest.lower() == 'exit' or "exit" in quest.lower():
                exit_flag = True
                print("Goodbye BB!")
            else:
                response = qa_chain({"query": quest})
                print(f"{self.name}: {response}")
                logging.info(response['result'])

    def set_agent(self):
        """
        loads vector embeddings for Agent parent class
        """

        self.agent.vectordb = Chroma(
            persist_directory="chroma_db/"+self.name, embedding_function=self.embedding_function)
        self.retriever = self.agent.vectordb.as_retriever()
        self.agent.vectordb.persist()

        if isinstance(self.agent.path, list):
            for doc in self.agent.path:
                print('doc:', doc)
                docs = self.agent.course.from_pdf(doc)
                self.agent.encoder.embed_vectors(docs)

                self.agent.vectordb.persist()
                print('updated vector db')
        print('init complete')

    def add_fractual(self, docs):
        """
        add documents to corpus

        """
        if not self.agent.vectordb:
            self.set_agent()

        self.agent.vectordb.persist()

    def one_question(self, question):
        '''
        For Pack Grand-Parent Class
        Chat with Agent one question at a time
        '''
        self.question = question
        qa_chain = RetrievalQA.from_chain_type(
            self.llm, retriever=self.agent.vectordb.as_retriever())

        if self.agent.cot_name == 1:
            self.question = "step by step, and one by one explain: " + self.question

        response = qa_chain({"query": self.question})
        print(f"{self.name}: {response}")
        logging.info(response['result'])
