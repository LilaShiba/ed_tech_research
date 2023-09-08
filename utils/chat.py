""" Start Chat with resources """
import os
from langchain.embeddings.sentence_transformer import \
    SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import chromadb


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
        self.chat_vectordb = None

        self.model = "facebook-dpr-ctx_encoder-multiset-base"

        self.embedding_function = SentenceTransformerEmbeddings(
            model_name=self.model)

        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo", temperature=0.5)

    def question(self, quest):
        """ Chat with resources """

        self.current_question = quest

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
                print(f"{self.name}: {response}")

    def load_chat(self):
        """
        Chat with default agent settings
        """

        print('agent loaded')

        # Enter Chat Stream

        exit_flag = False
        while not exit_flag:
            quest = input(
                f"Please ask a question about {self.name} or type 'exit' to end: ")

            if quest.lower() == 'exit':
                exit_flag = True
                print("Goodbye!")
            else:
                response = self.agent.encoder.db.similarity_search(quest)

                # response = self.qa_chain({"query": quest})
                print(f"{self.name}: {response[0].page_content}")

    def set_agent(self):
        """
        loads vector embeddings for Agent parent class
        """

        # self.chat_vectordb = Chroma(persist_directory=self.agent.path,
        #                             embedding_function=embedding_function)
        # self.retriever = self.chat_vectordb.as_retriever()

        # self.qa_chain = RetrievalQA.from_chain_type(llm=self.llm,
        #                                             retriever=self.retriever)

        persistent_client = chromadb.PersistentClient()
        collection = persistent_client.get_or_create_collection(
            "collection_name")
        collection.add(ids=["1", "2", "3"], documents=["a", "b", "c"])

        langchain_chroma = Chroma(
            client=persistent_client,
            collection_name="collection_name",
            embedding_function=self.embedding_function,
        )

        print("There are", langchain_chroma._collection.count(), "in the collection")
