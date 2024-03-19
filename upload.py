# dongyoun@iastate.edu, dek@IASTATE.EDU
#Copyleft Dongyoun Kim, Daeun Kim
"""
# Written by Dongyoun Kim, and Daeun Kim.
    Tasks:
    1. Upload/add pdf to  RAG platform
    2. Index the PDF files (chunk the content of the pdf files, embed them, and store them in a vector DB)
    
    Output:
    
"""

import argparse
import os
import re

#from llama_index.embeddings.openai import OpenAIEmbedding -> is subject to rate limit
# Llama indexs frameworks
## load the file
from llama_index.core import Settings, SimpleDirectoryReader
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
# Initialize VectorStore
from pinecone import Pinecone, PodSpec

import config


class Upload:
    """
        ## PipeCone
        DATA_URL: https://rag579-9051ec9.svc.gcp-starter.pinecone.io
        
        # Future works
        1. Upload the file from Funix.io by using drag and drop
            (ref: line 224 https://github.com/forrestbao/vectara-python-cli/blob/main/src/vectara/__init__.py)
        Currently, Upload the file to PipeCone with command line.
    """
    
    def __init__(self, args):
        """
        Summary
        """
        #Key setting - Hide the API keys on Github
        #os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY
        
        #Enter the API keys
        os.environ["PINECONE_API_KEY"] = config.PINECONE_API_KEY
        os.environ["PINECONE_ENV"] = config.PINECONE_ENV
        
        self.args = args
        self.file_name = self.args.file_name #str
        
        if self.file_name != None:
            self.file_path = os.path.join('./documents',self.file_name)
            print(self.file_path)
        
        ## Folder loader
        else:
            self.file_path = self.args.folder
        
        self.ch_size = self.args.chunck_size #int [100, 200]
        self.ch_over = self.args.chunck_overlap #float
        self.ch_over = self.ch_over * self.ch_size #int
        self.name_space = self.args.name_space
        
        #Activate the vector DB
        self.init_pipecone()
        
    def read_data(self, file_path):
        """
            1. Check the path whether it is file or folder
            2. Read the documents to upsert
        """
        ## 1. The data convert it into chunks using the load_data method
        #a. file case
        if os.path.isfile(file_path):
            self.data_file = [file_path]
            self.docs = SimpleDirectoryReader(input_files = self.data_file).load_data()
        
        #b. folder case
        else:
            self. docs = SimpleDirectoryReader(self.file_path).load_data()
        
        print("Total pages read:", len(self.docs))
        #return documents
        return self.docs
    
    # Define a function to preprocess text
    def preprocess_text(self,text):
        # Replace consecutive spaces, newlines and tabs
        text = re.sub(r'\s+', ' ', text)
        return text
    
    def chunck(self, docs):
        #Chunk -> nodes
        splitter = SentenceSplitter(chunk_size=self.ch_size, chunk_overlap=self.ch_over)
        nodes = splitter.get_nodes_from_documents(docs)
        print("Chunck size :", self.ch_size, "Chunck overlap :", self.ch_over)
        return nodes
        
    def embedding(self, nodes):
        #Embedding output dimension = 384
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        embed_model= Settings.embed_model
        embeddings_list = []
        
        for node in nodes:
            print("...Progressing the indexing data:", node.metadata['file_path'] + '...Page number: ' + node.metadata['page_label'])
            res = embed_model.get_text_embedding(node.text)
            embeddings_list.append(res)
        return embeddings_list
        
    def upsert_data(self):
        #Load
        docs = self.read_data(self.file_path)
        #chunks
        nodes= self.chunck(docs)
        
        #embedding
        embedding = self.embedding(nodes)
        ## upsert
        self.pc_index.upsert(
            vectors=[(node.metadata['file_name'][:2]+node.metadata['page_label'], emb) for node, emb in zip(nodes,embedding)],
                    namespace=self.name_space)
        self.show_vectordb()
        
    def init_pipecone(self):
        # Define the index name
        self.pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        index_name = "rag579"
        if index_name not in self.pc.list_indexes().names():
            #Embedding output dimension = 384
            self.pc.create_index(name='rag579', dimension=384, metric="cosine", spec=PodSpec(environment="gcp-starter"))
            # connect to index
            self.pc_index = self.pc.Index(index_name)
            
        self.pc_index = self.pc.Index(index_name)
        self.vector_store = PineconeVectorStore(pinecone_index=self.pc_index, namespace= self.name_space)
        #Print init status
        print("VectorDB init")
        self.show_vectordb()
    
    def show_vectordb(self):
        print("Vector DB (Pinecone)- status: ", self.pc_index.describe_index_stats())
        
        
def get_args(argv=None):
    parser = argparse.ArgumentParser(description= 'Process the pdf file for uploading the file to Pinecone (Vector DB)')
    parser.add_argument('--file_name', type=str, default= None, help='A path of input file')
    parser.add_argument('--folder', type=str, default= './documents/', help='A folder path for input files')
    parser.add_argument('--name_space', type=str, default = None, help= 'Enter the assgining the namespace on Pinecone')
    parser.add_argument('--chunck_size', type=int, default=200, help='Enter the chunck size over 100 range')
    parser.add_argument('--chunck_overlap', type=float, default=0.25, help='The portion of the overlap chunks: 25% = 0.25 range[0,1]')
    parser.parse_args()
    #parser.print_help()
    
    return parser.parse_args(argv)

if __name__== "__main__":
    inputs = get_args()
    data_load = Upload(inputs)
    # Upload
    data_load.upsert_data()