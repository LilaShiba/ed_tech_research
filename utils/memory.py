"""
Memory For Agent Module

"""
from langchain.vectorstores import Chroma


class Memory:
    """Memory for AI Agent. Composed of
       Encoder instance & Chat_bot instance
       Stored in Chroma DB
       Logged and reviewed here to knowldge corpus
    """

    def __init__(self, agent):
        '''
        Takes in Agent Instance
        Connects to Memories 
        '''
        self.agent = super().__init__()
