from utils.document_loader import NewCourse
from utils.dual_encoder import Encoder
from utils.chat import ChatBot


class Agent:
    """Top Level for AI Agent."""

    def __init__(self, name: str, path: str):
        """
        Initializes the Agent with a name and path.

        Parameters:
            - name: Name of the agent.
            - path: Document path.
        """
        self.name = name
        self.path = path
        self.course = NewCourse(name, path)
        self.encoder = Encoder(self.course)
        self.conversation = ChatBot(self.name)
        self.path = path
        self.course.knowledge_document_path = path
        self.agent_instance = None
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

    def chat(self):
        """
        Chat with a resource

        Parameters:
            - name: Name of the agent.
            - path: Document path.
        """
        pass


if __name__ == "__main__":

    testAgent = Agent("KBAI_2023", 'documents/kbai_2023.pdf')
    testAgent.new_course()
