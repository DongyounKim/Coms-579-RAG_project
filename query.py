import argparse
import os

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP

import config
import upload

#Enter the API keys for accessing Pipecone
os.environ["PINECONE_API_KEY"] = config.PINECONE_API_KEY
os.environ["PINECONE_ENV"] = config.PINECONE_ENV
    
def send_query(_idx,query,top_k =5):
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
    query_embedding = embed_model.get_text_embedding(query)
    
    #Query result from Pinecone
    query_result = _idx.query(vector = query_embedding, top_k=top_k, include_values=True, include_metadata=True)
    #print(query_result)
    
    ## Generate new nodes for Vector Index
    ### Due to the version
    _nodes= []
    print("Query: ", query ,"\n")
    print("Retrieval \n")
    print("top-k :", top_k,"\n")
    for i, _t in enumerate(query_result['matches']):
        print("-------------",i ,"------------- \n")
        _node =TextNode(text=_t['metadata']['text'])
        _nodes.append(_node)
        print("Similarity: ", _t['score'],"\n")
        print("Context:", _node)
        print("--------------------------- \n")
    # create vector store index
    _index = VectorStoreIndex(_nodes)
    #Re-rank
    query_engine = _index.as_query_engine(similarity_top_k=top_k, llm=llm)
    response = query_engine.query(query)
    return response

if __name__== "__main__":
    parser = argparse.ArgumentParser(description= 'Process the pdf file for uploading the file to Pinecone (Vector DB)')
    parser.add_argument('--question', type=str, default= None, required=True, help='A query')
    parser.add_argument('--top_k', type=int, default=5, help='top_k')
    args = parser.parse_args()
    
    #Initiate the vector space
    index, vector_space = upload.init_pipecone()
    response = send_query(index,args.question, args.top_k)
    print("Answer: ")
    print(str(response))