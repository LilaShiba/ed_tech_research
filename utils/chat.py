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

    def add_fractual(self, docs):
        """
        add documents to corpus

        """
        print('loading', self.agent.name)

        docs = self.agent.course.from_pdf(self.agent.path)
        self.agent.encoder.create_chunks(docs)
        print("chunks created")
        self.agent.encoder.embed_chunks()
        print('Embedding created')

        if not self.agent.vectordb and isinstance(self.agent.corpus_path_array,list):
            print('corpus load start add_fractual: ')
            for doc in self.agent.corpus_path_array:
                print(doc)
                docs = self.agent.course.from_pdf(doc)
              
                print("chunks created")
                self.self.agent.encoder.embed_chunks()
                # save to disk
                self.agent.vectordb = Chroma.from_documents(
                    self.docs, self.embedding_function, persist_directory="./chroma_db/"+self.name)

        elif self.agent.vectordb:

            print('loading', self.agent.name)
            self.agent.vectordb.add_documents(docs)
            self.agent.vectordb.persist()
            print('update', self.agent.name)
        else:
            
                # save to disk
            self.agent.vectordb = Chroma.from_documents(
                self.docs, self.embedding_function, persist_directory="./chroma_db/"+self.name)

            print('loading', self.agent.name)
            self.agent.vectordb.add_documents(docs)
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
