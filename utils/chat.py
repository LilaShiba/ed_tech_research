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
            self.llm, retriever=self.agent.encoder.vectordb.as_retriever())

        if self.agent.cot_name == 1:
            self.question = "step by step, and one by one explain: "
        elif self.agent.cot_name == 2:
            self.question = "line by line, write python code that: "
        elif self.agent.cot_name == 3:
            self.question = "thought by thought, synthesize: "

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
                # print(f"{self.name}: {response}")
                logging.info(self.name, response)
                print(response)

    def set_agent(self):
        """
        loads vector embeddings for Agent parent class
        """
        self.agent.encoder.vectordb = Chroma(
            persist_directory="chroma_db/"+self.name, embedding_function=self.embedding_function)
        self.retriever = self.agent.encoder.vectordb.as_retriever()

    def create_vectordb(self, docs):
        '''
        creates vectordb for parent agent
        '''
        print('docs, ', docs)
        doc = self.agent.course.from_pdf(docs)
        self.agent.encoder.subprocess_create_embeddings(doc)
        print('process complete')

    def add_fractual(self, docs):
        """
        add documents to corpus

        """
        print('loading:', self.agent.name)

        # Process array of docs
        if isinstance(self.agent.knowledge_document_path, list) and not self.agent.vectordb:
            print('corpus load start add_fractual: ')
            for doc in self.agent.knowledge_document_path:
                print(doc)
                docs = self.agent.course.from_pdf(doc)
                self.agent.encoder.create_chunks(docs)
                print("chunks created")
                self.agent.encoder.embed_chunks()
                print("embedding complete")
                # save to disk
        # No DB
        elif not self.agent.vectordb:
            # Load DB
            if 'chroma_db' in docs:
                print('no agent vector db. Creating now')
                self.agent.encoder.vectordb = Chroma.from_documents(
                    self.agent.encoder.docs, self.agent.encoder.embedding_function, persist_directory="./chroma_db/"+self.name)

        else:
            print(docs)
            print('loading', self.agent.name)
            doc = self.agent.course.from_pdf(docs)
            self.agent.encoder.create_chunks(doc)
            print("chunks created")
            self.agent.encoder.embed_chunks()
            print("embedding complete")
            self.agent.encoder.vectordb.add_documents(doc)
        # print('loading done for:', docs, ' in: ', self.agent.name)
        self.agent.encoder.vectordb.persist()

    def one_question(self, question):
        '''
        For Pack Grand-Parent Class
        Chat with Agent one question at a time
        '''
        self.question = question

        qa_chain = RetrievalQA.from_chain_type(
            self.llm, retriever=self.agent.encoder.vectordb.as_retriever())

        if self.agent.cot_name == 1:
            self.question = "step by step, and one by one explain: " + self.question

        response = qa_chain({"query": self.question})
        # print(f"{self.name}: {response}")
        # logging.info(response['result'])
        return response['result']

    def add_new_docs(self, docs):
        '''
        add new docs to db
        '''
        print(docs)
        print('loading', self.agent.name)
        doc = self.agent.course.from_pdf(docs)
        self.agent.encoder.create_chunks(doc)
        print("chunks created")
        self.agent.encoder.embed_chunks()
        print("embedding complete")
        self.agent.encoder.vectordb.add_documents(doc)
        # print('loading done for:', docs, ' in: ', self.agent.name)
        self.agent.encoder.vectordb.persist()
