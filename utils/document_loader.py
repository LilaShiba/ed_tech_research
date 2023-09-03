import os
import sys
import openai
from dotenv import find_dotenv, load_dotenv
from langchain.document_loaders import PyPDFLoader, TextLoader, UnstructuredHTMLLoader
from langchain.document_loaders.csv_loader import CSVLoader

sys.path.append('../..')
load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']


class NewCourse:
    """Manages the creation and handling of courses."""

    def __init__(self, name: str, path: str):
        """
        Initializes the NewCourse object.

        Parameters:
            - name: Name of the course.
            - path: Document path for the course.
        """
        self.name = name
        self.links = None
        self.docs = None
        self.chunks = None
        self.knowledge_document_path = path
        self.vectordb = None
        self.k_results = None
        self.embedding_function = None

    def from_pdf(self, knowledge_document_path):
        """
        Creates new course from pdf

        Parameters:
        PDF path

        Returns:
        self.url
        self.docs 
        self.chunks
        self.embeddings

        """
        self.knowledge_document_path = knowledge_document_path
        loader = PyPDFLoader(knowledge_document_path)
        self.docs = loader.load()
        print('docs created')

    def from_txt(self, knowledge_document_path):
        """
        Creates new course from txt

        Parameters:
        txt file path

        Returns:
        self.url
        self.docs 
        self.chunks
        self.embeddings

        """
        self.knowledge_document_path = knowledge_document_path
        loader = TextLoader(knowledge_document_path)
        self.docs = loader.load()
        print('docs created')

    def from_csv(self, knowledge_document_path):
        """
        Creates new course from csv

        Parameters:
        csv file path

        Returns:
        self.url
        self.docs 
        self.chunks
        self.embeddings

        """
        self.knowledge_document_path = knowledge_document_path
        loader = CSVLoader(file_path=knowledge_document_path)
        self.docs = loader.load()
        print('docs created')

    def from_html(self, knowledge_document_path):
        """
        Creates new course from html

        Parameters:
        csv file path

        Returns:
        self.url
        self.docs 
        self.chunks
        self.embeddings

        """
        self.knowledge_document_path = knowledge_document_path
        loader = UnstructuredHTMLLoader(file_path=knowledge_document_path)
        self.docs = loader.load()
        print('docs created')
