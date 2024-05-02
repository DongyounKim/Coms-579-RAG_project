import argparse
import os
import typing  # Python native
from typing import Dict, List  # Python native

import fitz
from funix import funix
from funix.hint import Markdown
from funix.widget.builtin import BytesFile

import config
import query
import upload

#import ipywidgets  # popular UI library



# def upload_pdf(PDFBytes: BytesFile,
#             chunk_size: int = 500,
#     chunk_overlap: int = 0.2,
#     top_k: int =5,
#     prompt_template: str = "The following is part of a paper {article}. Please answer the question: ...",
#     system_message: str = "",) -> Markdown:
#     with fitz.open(stream = PDFBytes, filetype = "pdf") as doc: # open document
#         text_list = [page.get_text() for page in doc]
#     #print(text_list)

#     upload.upsert_data(index, text_list, chunk_size, chunk_overlap)
    
#     response = query.send_query(index, prompt_template, top_k)
#     print (response)

#     #response_messages.append(response)

#     return str(response)

os.environ["PINECONE_API_KEY"] = config.PINECONE_API_KEY
os.environ["PINECONE_ENV"] = config.PINECONE_ENV
index, vector_space= upload.init_pipecone()

@funix(
    print_to_web=True
)
def upload_pdf(PDFBytes: BytesFile,
        chunk_size: int = 500,
chunk_overlap: int = 0.2,)->str:
    with fitz.open(stream = PDFBytes, filetype = "pdf") as doc: # open document
        text_list = [page.get_text() for page in doc]
    #print(text_list)
    upload.upsert_data(index, text_list, chunk_size, chunk_overlap)
    return "Done"

@funix(
    print_to_web=True
)
def request_query(top_k: int =5,
prompt_template: str = "what is the large language model?",)-> Markdown:
    response = query.send_query(index, prompt_template, top_k)
    #print (response)
    #response_messages.append(response)
    print("\n ** ANWSER **\n")
    return str(response)