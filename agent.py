"""
Top Lvl Agent Module

"""
from utils.document_loader import NewCourse
from utils.dual_encoder import Encoder
from utils.chat import ChatBot


class Agent:
    """Top level for AI Agent. Composed of
       Encoder instance & NewCourse instance 
    """

    def __init__(self, name: str, path: str, cot: bool):
        """
        Initializes the Agent with a name and path.

        Parameters:
            - name: Name of the agent.
            - path: Document path.
        """
        self.name = name
        self.path = path
        self.cot = cot
        self.course = NewCourse(name, path)
        self.encoder = Encoder(self.course)
        self.chat_bot = ChatBot(self)
        self.path = path
        self.course.knowledge_document_path = path
        self.agent_instance = None
        self.current_docs = None
        print(f'the knowledge document being used is {path}')

    def new_course(self):
        """
        Creates the Docs and Embeddings with a name and path.

        Parameters:
            - name: Name of the agent.
            - path: Document path.
        """
        self.course.from_pdf(self.path)
        self.encoder.subprocess_create_embeddings(
            self.course.docs)
        print('instance created')

    def start_chat(self, quest=None):
        """
        Chat with a resource

        Parameters:
            - name: Name of the agent.
            - path: Document path.
        """

        self.chat_bot.enter_chat(quest)

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

    def load_course(self, db_path: str):
        """
        load vector embeddings from Chroma

        """
        self.chat_bot.load_chat(db_path)

    def load_mem(self):
        """
       load Agent Memory
       Provided self.path is to the DB
       """


if __name__ == "__main__":
    # Create Course Demo
    # testAgent = Agent(
    #     "Agent_Time", 'documents/The-order-of-time-Carlo-Rovelli.pdf', True)
    # testAgent.new_course()
    # testAgent.start_chat("What is this document about?")

    # Load Course Embeddings Demo
    EMBEDDING_PATH = 'docs/chroma/a65cd2ee-3115-4cd1-82ea-0e1592af494d'
    testAgent = Agent(
        "Agent_Time", None, True)
    testAgent.load_course(EMBEDDING_PATH)
