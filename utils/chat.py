""" Start Chat with resources """


from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings.sentence_transformer import \
    SentenceTransformerEmbeddings


class ChatBot:
    """ Chat with resources """

    def __init__(self, agent_instance):
        """ Chat with resources """
        self.agent = agent_instance
        self.name = agent_instance.name
        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo", temperature=0.5)
        self.current_question = None

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

    def load_chat(self, path_to_db: str):
        """
        loads vector embeddings for Agent parent class
        """
        model = "facebook-dpr-ctx_encoder-multiset-base"

        embedding_function = SentenceTransformerEmbeddings(
            model_name=model)

        print('loading agent...')

        self.agent.encoder.vectordb = Chroma(persist_directory=path_to_db,
                                             embedding_function=embedding_function)
        print('agent loaded')

        # Enter Chat Stream
        llm = self.llm
        qa_chain = RetrievalQA.from_chain_type(
            llm, retriever=self.agent.encoder.vectordb.as_retriever())

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
