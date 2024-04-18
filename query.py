import argparse
import os

from ctransformers import AutoModelForCausalLM
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.vector_stores.pinecone import PineconeVectorStore
# Initialize VectorStore
from pinecone import Pinecone, PodSpec

import config

# construct vector store query



#Enter the API keys for accessing Pipecone
os.environ["PINECONE_API_KEY"] = config.PINECONE_API_KEY
os.environ["PINECONE_ENV"] = config.PINECONE_ENV

class Query():
    def __init__(self, question, topk ):
        self.query = question
        self.top_k = topk
        self.pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        index_name = "rag579"
        if index_name not in self.pc.list_indexes().names():
            #Embedding output dimension = 384
            self.pc.create_index(name='rag579', dimension=384, metric="cosine", spec=PodSpec(environment="gcp-starter"))
            # connect to index
            self.pc_index = self.pc.Index(index_name)
            
        self.pc_index = self.pc.Index(index_name)
        self.vector_store = PineconeVectorStore(pinecone_index=self.pc_index)
        #Print init status
        #print("VectorDB init")
        #self.show_vectordb()
        self.send_query()
    
    def send_query(self):
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        embed_model= Settings.embed_model
        """
            language model
        """
        model_url= "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q4_0.gguf"
        llm = LlamaCPP(
            model_url=model_url,
            # optionally, you can set the path to a pre-downloaded model instead of model_url
            #model_path="./llm/llama-2-7b-chat.Q2_K.gguf",
            temperature=0.1,
            max_new_tokens=384,
            context_window=3000,
            generate_kwargs={},
            verbose=False,)
        #llm = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-7b-Chat-GGUF", model_file="llama-2-7b-chat.Q4_K_M.gguf", model_type="llama", gpu_layers=0)
        #Embedding the query
        query_embedding = embed_model.get_text_embedding(self.query)
        
        #Query result from Pinecone
        query_result = self.pc_index.query(vector = query_embedding, top_k=self.top_k, include_values=True, include_metadata=True)
        #print(query_result)
        
        ## Generate new nodes for Vector Index
        ### Due to the version
        _nodes= []
        print("Query: ", self.query)
        print("Retrieval")
        print("top-k :", self.top_k)
        for i, _t in enumerate(query_result['matches']):
            print("-------------",i ,"-------------")
            _node =TextNode(text=_t['metadata']['text'])
            _nodes.append(_node)
            print("Similarity: ", _t['score'])
            print("Context:", _node.text)
            print("---------------------------")
            
                
        # create vector store index
        _index = VectorStoreIndex(_nodes)
        #Re-rank
        query_engine = _index.as_query_engine(similarity_top_k=self.top_k, llm=llm)
        response = query_engine.query(self.query)
        print("Answer: ")
        print(str(response))
        
    def embedding(self, text):
        #Embedding output dimension = 384
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        embed_model= Settings.embed_model
        embeddings_list = []
        
        for node in text:
            #print("...Progressing the indexing data:", node.metadata['file_path'] + '...Page number: ' + node.metadata['page_label'])
            res = embed_model.get_text_embedding(text)
            embeddings_list.append(res)
        return embeddings_list

def get_args(argv=None):
    parser = argparse.ArgumentParser(description= 'Process the pdf file for uploading the file to Pinecone (Vector DB)')
    parser.add_argument('--question', type=str, default= None, required=True, help='A query')
    parser.add_argument('--top_k', type=int, default=5, help='top_k')
    parser.parse_args()
    
    return parser.parse_args(argv)

if __name__== "__main__":
    inputs = get_args()
    data_load = Query(inputs.question, inputs.top_k)