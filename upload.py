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
from llama_index.core import (Document, Settings, SimpleDirectoryReader,
                              VectorStoreIndex)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
# Initialize VectorStore
from pinecone import Pinecone, PodSpec

import config

#class Upload:
"""
    ## PipeCone
    DATA_URL: https://rag579-9051ec9.svc.gcp-starter.pinecone.io
    
    # Future works
    1. Upload the file from Funix.io by using drag and drop
        (ref: line 224 https://github.com/forrestbao/vectara-python-cli/blob/main/src/vectara/__init__.py)
    Currently, Upload the file to PipeCone with command line.
"""
os.environ["PINECONE_API_KEY"] = config.PINECONE_API_KEY
os.environ["PINECONE_ENV"] = config.PINECONE_ENV
        
def read_data(file_path):
    """
        1. Check the path whether it is file or folder
        2. Read the documents to upsert
    """
    ## 1. The data convert it into chunks using the load_data method
    #a. file case
    if file_path != None:
        file_path = os.path.join('./documents',file_path)
            
    if os.path.isfile(file_path):
        data_file = [file_path]
        docs = SimpleDirectoryReader(input_files = data_file).load_data()
    #b. folder case
    else:
        docs = SimpleDirectoryReader(file_path).load_data()
    
    print("Total pages read:", len(docs))
    #return documents
    return docs
    
# Define a function to preprocess text
def preprocess_text(text):
    # Replace consecutive spaces, newlines and tabs
    text = re.sub(r'\s+', ' ', text)
    return text
    
def chunk(docs, chunk_size = 200, chunk_overlap=0.25):
    #Chunk -> nodes
    #Check the node types
    ## document type
    chunk_overlap = int(chunk_size*chunk_overlap)
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if isinstance(docs[0],str):
        documents = [Document(text=t, metadata={"file_name":"funix gui","file_path": "funix", "page_label": "{}".format(_idx)}) for _idx,t in enumerate(docs)]
        nodes = splitter.get_nodes_from_documents(documents)
    else:
        nodes = splitter.get_nodes_from_documents(docs)
        print("Chunck size :", chunk_size, "Chunck overlap :", chunk_overlap)
    return nodes
        
def embedding(nodes):
    #Embedding output dimension = 384
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    embed_model= Settings.embed_model
    embeddings_list = []
    
    #Check the node types
    if isinstance(nodes[0],str):
        for node in nodes:
            res = embed_model.get_text_embedding(node)
            embeddings_list.append(res)
    else:
        for node in nodes:
            print("...Progressing the indexing data:", node.metadata['file_path'] + '...Page number: ' + node.metadata['page_label']+"\n")
            res = embed_model.get_text_embedding(node.text)
            embeddings_list.append(res)
        return embeddings_list
        
def upsert_data(_idx, docs, ch_size = 200, ch_overlap =0.25):
    #Load
    #docs = read_data(self.file_path)
    #chunks
    nodes= chunk(docs, ch_size, ch_overlap)
    #embedding
    embeded_text = embedding(nodes)
    ## upsert
    _idx.upsert(
        vectors=[(node.metadata['file_name'][:2]+node.metadata['page_label'], emb, {'text': node.text}) for node, emb in zip(nodes,embeded_text)])
    #show_vectordb()
    
def init_pipecone():
    # Define the index name
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index_name = "rag579"
    if index_name not in pc.list_indexes().names():
        #Embedding output dimension = 384
        pc.create_index(name='rag579', dimension=384, metric="cosine", spec=PodSpec(environment="gcp-starter"))
        # connect to index
        pc_index = pc.Index(index_name)
        
    pc_index = pc.Index(index_name)
    vector_store = PineconeVectorStore(pinecone_index=pc_index)
    #Print init status
    print("VectorDB init")
    return pc_index, vector_store
    
def show_vectordb(_idx):
    print("Vector DB (Pinecone)- status: ", _idx.describe_index_stats())

def deletes(_idx):
    #current delete all
    _idx.delete(delete_all=True)
        
if __name__== "__main__":
    # parsing
    parser = argparse.ArgumentParser(description= 'Process the pdf file for uploading the file to Pinecone (Vector DB)')
    parser.add_argument('--file_name', type=str, default= None, help='A path of input file')
    parser.add_argument('--chunk_size', type=int, default=200, help='Enter the chunk size over 100 range')
    parser.add_argument('--chunk_overlap', type=float, default=0.25, help='The portion of the overlap chunks: 25% = 0.25 range[0,1]')
    parser.add_argument('--delete', type=bool, default=False)
    parser.add_argument('--folder', type=str, default= './documents/', help='A folder path for input files')
    args = parser.parse_args()
    
    #Initiate the vector space
    index, vector_space = init_pipecone()

    #Check the task
    if args.file_name is None and args.delete == False:
        deletes(index)
    elif args.file_name:
        #File read
        data = read_data(args.file_name)
        upsert_data(index, data, args.chunk_size, args.chunk_overlap)
    else:
        print("Choose the tasks - upload pdf or delete pdf")