"""
Top Lvl Agent Module

"""

from utils.chat import ChatBot
from utils.document_loader import NewCourse
from utils.dual_encoder import Encoder


class Agent:
    """Top level for AI Agent. Composed of
       Encoder, DB, & NewCourse instance 
    """

    def __init__(self, name: str, path: str, cot: bool, cot_type: int, embedding_params: list):
        """
        Initializes the Agent with a name and path.

        Parameters:
            - name: Name of the agent.
            - path: Document path.
        """
        self.name = name
        self.path = path
        self.cot = cot
        self.cot_name = cot_type
        self.path = path
        self.knowledge_document_path = path
        self.vectordb = None
        self.agent_instance = None
        self.current_docs = None
        self.docs = None
        self.embedding_params = embedding_params
        # Subprocesses
        print('creating course for ' + self.name)
        self.course = NewCourse(name, path, embedding_params)
        print('creating encoder for ' + self.name)
        self.encoder = Encoder(self.course)
        print('creating chat_bot for ' + self.name)
        self.chat_bot = ChatBot(self)

        print(f'the path being used for {self.name} is {path}')

    def new_course(self):
        """
        Creates the Docs and Embeddings with a name and path.

        Parameters:
            - name: Name of the agent.
            - path: Document path.
        """
        self.course.from_pdf(self.path)
        self.vectordb = self.encoder.subprocess_create_embeddings(
            self.course.docs)
        print('instance created')

    def start_chat(self):
        """
        Chat with a resource

        Parameters:
            - name: Name of the agent.
            - path: Document path.
        """
        self.chat_bot.load_chat()

    def save(self):
        """
        Save the current instance of Agent to ChromaDB
        DBPath = docs/chroma/self.name

        Parameters:
            - name: Name of the agent.
            - path: Document path.
            - encoder: document instance
            - course: course instance
        """
        self.encoder.subprocess_persist(self.course.knowledge_document_path)
        print(f'instance saved at docs/chroma/{self.name}')

    def load_course(self):
        """
        load vector embeddings from Chroma

        """
        print(f'waking up agent {self.name}')
        self.chat_bot.set_agent()

    def load_mem(self):
        """
       load Agent Memory
       Provided self.path is to the DB
       """

    def add_memory(self, path, path_to_db):
        '''
         Add documents to vector db
         path: document path
         path_to_db: path to chroma 
        '''
        print(f'adding {path}')
        pages = self.course.from_pdf(path)
        docs = self.encoder.create_chunks(pages)

        self.vectordb.add_documents(docs)
        self.vectordb.persist()
        # self.encoder.vectordb.add_documents(embeddings)
        print("memory updated")

        with open('output.log', 'r') as file:
            # Read the content of the file
            pages = self.course.from_txt('output.log')
            docs = self.encoder.create_chunks(pages)
            self.chat_bot.add_fractual(docs)

        # Clear the contents of the .log file
        with open('output.log', 'w') as file:
            pass


if __name__ == "__main__":
    # Create Course Demo
    # testAgent = Agent("agent_snd", 'documents/SND.pdf', True, 1)
    # testAgent.new_course()
    # testAgent.start_chat()

    # Enable Chains of Thought
    # Memory & COT example
    testAgent = Agent(
        "agent_snd", "chroma_db/agent_snd", True, 1, embedding_params=["facebook-dpr-ctx_encoder-multiset-base", 200, 25, 0.7])
    testAgent.cot = True
    testAgent.load_course()
    # testAgent.add_memory("documents/HilbertSpaceMulti.pdf", path_to_db)
    testAgent.start_chat()

    # mem_bot = Agent(
    #     "mem_bot", 'documents/HilbertSpaceMulti.pdf', True, 1)
    # mem_bot.new_course()
    # db_path = '/chroma_db/mem_bot'
    # mem_bot.path = db_path
    # mem_bot.load_course()
    # mem_bot.add_memory(
    #     'documents/VisualizingQuantumCircuitProbability.pdf', db_path)
    # mem_bot.start_chat()
