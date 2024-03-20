# NLP_project (Spring 2024)
This is a repo for the NLP class project (Retrieval-augmented generation).
 
The main instruction following: https://github.com/forrestbao/nlp-class/blob/master/projects.md
## Team members
Dongyoun Kim, Daeun Kim

## Objectives
Build the Retrieval-augmented generation (RAG)
### Tech stack
1. Pipeline: [LlamaIndex] (https://www.llamaindex.ai/)
2. VectorDB: [PineCone]  (https://www.pinecone.io/)
3. UI: Funix.io

## Project Structure

#### Data repository
```
.
|-- documents/ # documentation files (Every input have to store it in doucments)
|-- demos/ # videos demos
| upload.py
| requirement.txt

```
- **documents**: This folder contains input files (pdf)
- **demos**: This folder contains the demo short video
- **upload.py**: This code reads the file or files in the default folder('documents'), and indexes and stores the file/files in a vector DB ('Pipecone').
- **requirement.txt**: A list of packages or libraries needed to work on a project that can all be installed with the file.


## Getting Started

### Sub-tasks1
**Task**: March 20: PDF upload and indexing via command line done. Include README file and a short video demo to show the usage and completion of your command line tool. e.g., `python upload.py --pdf_file=example.pdf` to add one PDF file each time. 

#### Steps of the sub-task
1. Read the file : Load and read pdf file + pre-processing for replacing consecutive spaces, newlines and tabs in the file.

2. Chunk: This project chunks the pdf file based on the sentence - chunk size and chunk overlap

3. Embedding: This project use a simple embedding model (bge-small-en-v1.5) from hugging face. Because the OpenAI API (GPT) is subject to rate limitation error.

4. Upload(Upsert) the data

#### Arguments
```
    parser = argparse.ArgumentParser(description= 'Process the pdf file for uploading the file to Pinecone (Vector DB)')
    parser.add_argument('--file_name', type=str, default= None, help='A path of input file')
    parser.add_argument('--folder', type=str, default= './documents/', help='A folder path for input files')
    parser.add_argument('--name_space', type=str, default = None, help= 'Enter the assgining the namespace on Pinecone')
    parser.add_argument('--chunck_size', type=int, default=200, help='Enter the chunck size over 100 range')
    parser.add_argument('--chunck_overlap', type=float, default=0.25, help='The portion of the overlap chunks: 25% = 0.25 range[0,1]')
    parser.parse_args()
```

* '--file_name': *[Optional]* Enter the PDF file name. The file have to store in 'documents/'.  If do not enter it, the code will **Reads the files in the documents folder**

    ```
    >>> python upload.py --file_name sample.pdf
    ```  

* '--folder': *[Optional]* Enther the folder path containing pdf files. The default is './documents'

    ```
    >>> python upload.py --folder ./docs
    ```

* '--chunk_size': *[Optional]* Enter the chunk size as integer. The default is 200

    ```
    >>> python upload.py --chunk_size 100
    ```

* '--chunk_overlap': *[Optional]* Enter the ratio of chunk overlap ranging [0,1]. The default is 0.25
 
    ```
    >>> python upload.py --chunck_overlap 0.25
    ```

* '--name_space': *[Optional]* Enter the name space for 'Pipecone' DB when storing the data on the index of the DB.

    ```
    >>> python upload.py --name_space test
    ```

## Getting Started
To implement this project,

### KEY setting
```
#line 47 -  
#Enter the API keys for accessing Pipecone
os.environ["PINECONE_API_KEY"] = 'xxxxxx..' #Enter YOUR KEY
os.environ["PINECONE_ENV"] = "gcp-starter" #Enter YOUR environment in Pipecone(DB)
``` 

### Sub-tasks1
#### 1. Single file upload
The file have to store in 'documents' folder.

```
>>> python upload.py --file_name sample.pdf
```

#### 2. Folder upload
Upload the default the folder path

```
>>> python upload.py
```
#### 3. Optional arguments
 
```
>>> python upload.py --name_space test_case --chunck_size 100 --chunk_overlap 0.20 # overlap = 100*0.2 = 20.
```


## Reference