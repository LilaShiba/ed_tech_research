"""
Dual Encoder Module

"""
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import TokenTextSplitter
import sys
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
        self.embedding_function = course_instance.embedding_function

    # ... Other methods ...

    def create_chunks(self, docs, chunk=20, overlap=5):
        """
        Creates new chunks from documents
        """
        text_splitter = TokenTextSplitter(
            chunk_size=chunk, chunk_overlap=overlap)
        self.chunks = text_splitter.split_documents(docs)

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

    def encoded_query(self, query, k_docs=5):
        """
       Encodes query then searches

        Parameters:
        self.docs

        Returns:
        k answers

        """
        res = self.vectordb.similarity_search_with_score(
            query=query, distance_metric="cos", k=k_docs)
        return res

    def from_db(self, path_to_db, model="facebook-dpr-ctx_encoder-multiset-base"):
        """
        Creates exisiting course obj, chunks,
        from chroma db

        Parameters:
        path_to_db
        (Opt) model: default dpr

        Returns:
        self.vectordb
        wrapper around vector_db
        """
        embedding_function = SentenceTransformerEmbeddings(
            model_name=model)

        self.vectordb = Chroma(persist_directory=path_to_db,
                               embedding_function=embedding_function)
