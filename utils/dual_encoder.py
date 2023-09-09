"""
Dual Encoder Module
Creates Vector Embeddings

"""
import sys

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.sentence_transformer import \
    SentenceTransformerEmbeddings
from langchain.text_splitter import TokenTextSplitter
from langchain.vectorstores import Chroma

sys.path.append('../..')


class Encoder:
    """
    Handles encoding of documents for a given course.
    """

    def __init__(self, course_instance):
        """
        Initializes the Encoder with a given course instance.

        Parameters:
            - course_instance: Instance of NewCourse.
        """
        self.course_instance = course_instance
        self.links = course_instance.links
        self.docs = course_instance.docs
        self.chunks = course_instance.chunks
        self.vectordb = course_instance.vectordb
        self.k_results = course_instance.k_results
        self.embedding_check = False
        self.name = course_instance.name
        self.data_base = None

        self.model = "facebook-dpr-ctx_encoder-multiset-base"

        self.embedding_function = SentenceTransformerEmbeddings(
            model_name=self.model)

    def create_chunks(self, docs, chunk=200, overlap=15):
        """
        Creates new chunks from documents
        """
        text_splitter = TokenTextSplitter(
            chunk_size=chunk, chunk_overlap=overlap)
        self.chunks = text_splitter.split_documents(docs)
        return self.chunks

    def embed_chunks(self, persist_directory='docs/chroma/'):
        """
        Creates new embeddings from chunks

        Parameters:
        persist_directory path
        self.docs
        self.chunks

        Returns:
        self.vectordb
        """
        embedding = OpenAIEmbeddings()
        self.vectordb = Chroma.from_documents(
            documents=self.chunks,
            embedding=embedding,
            persist_directory=persist_directory
        )

    def subprocess_create_embeddings(self, docs):
        """
        Creates new course from processed docs

        Parameters:
        self.docs

        Returns:
        self.chunks
        self.embeddings
        self.vectordb

        """
        self.docs = docs
        self.create_chunks(docs)
        print("chunks created")
        self.embed_chunks()
        print("embedding created")
        # save to disk
        self.data_base = Chroma.from_documents(
            self.docs, self.embedding_function, persist_directory="./chroma_db/"+self.name)

    def subprocess_persist(self, path, model="facebook-dpr-ctx_encoder-multiset-base"):
        """
        Creates new course docs, chunks, and embeddings 
        from processed docs

        Parameters:
        self.docs

        Returns:
        self.chunks
        self.embeddings
        self.vectordb

        """
        embedding_function = SentenceTransformerEmbeddings(
            model_name=model)

        self.create_chunks(self.docs)

        # save to disk
        vectordb = Chroma.from_documents(
            self.chunks, embedding_function, persist_directory=path)
        vectordb.persist()
        self.vectordb = vectordb
        self.embedding_function = embedding_function

    def encoded_query(self, q_a, k_docs=15):
        """
       Encodes query then searches

        Parameters:
        self.docs

        Returns:
        k docs
        doc len

        """
        if self.course_instance.cot:
            cot_q = "step by step and one by one explain" + q_a
            docs = self.vectordb.similarity_search_with_score(
                query=cot_q, distance_metric="cos", k=k_docs)
        else:
            docs = self.vectordb.similarity_search_with_score(
                query=q_a, distance_metric="cos", k=k_docs)
        return docs, len(docs)

    def from_db(self, path_to_db, model="facebook-dpr-ctx_encoder-multiset-base"):
        """
        loads vector embeddings for Agent
        Start of memory 
        TODO: Add to pipeline
        """
        self.path_to_db = path_to_db
        model = "facebook-dpr-ctx_encoder-multiset-base"

        embedding_function = SentenceTransformerEmbeddings(
            model_name=model)

        print('loading agent...')

        self.vectordb = Chroma(persist_directory=path_to_db,
                               embedding_function=embedding_function)
        print('agent loaded')

    def add_documents(self, path):
        '''
        Add documents to vector db
        '''

        self.vectordb.add_documents(documents=path)

    def add_embeddings(self, docs, path_to_db):
        """
        Add documents
        """
        embedding = OpenAIEmbeddings()
        vectordb = Chroma.from_documents(
            documents=docs,
            embedding=embedding,
            persist_directory=path_to_db
        )
        return vectordb
