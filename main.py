# importing required modules
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import LineTokenizer
import glob
import re
import os
from typing import Dict, Any, AnyStr
import uuid
import pandas as pd
import docx2txt
import textract
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import string
from docx.api import Document
import numpy as np
from fastapi import FastAPI, HTTPException
from transformers import BertModel, BertTokenizer
import csv
from typing import List
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
from nltk.corpus import stopwords
from itertools import chain
nltk.download("stopwords")
nltk.download('punkt')

#pandas max columns and rows
pd.set_option('display.max_colwidth', None)
pd.set_option("display.max_rows", None)





import os
import pandas as pd

file1_name = "Source rules or OCD rulebook.csv"
with open(file1_name, 'r', encoding='utf-8') as f:
    file1_content = f.read()
    
file2_name = "Target Rules or EA rulebook.csv"
with open(file2_name, 'r', encoding='utf-8') as f:
    file2_content = f.read()

pdf_data_name = "pdf_data_short.csv"
with open(pdf_data_name, 'r', encoding='utf-8') as f:
    pdf_data_content = f.read()
    
pdf_data2_name = "pdf_data2_short.csv"
with open(pdf_data2_name, 'r', encoding='utf-8') as f:
    pdf_data2_content = f.read()
    
pdf_data3_name = "pdf_data3_short.csv"
with open(pdf_data3_name, 'r', encoding='utf-8') as f:
    pdf_data3_content = f.read()

main_df = pd.DataFrame({
    'Document name': [os.path.basename(file1_name), os.path.basename(file2_name), ' Management of Safety-Related Rail Vehicle Defects.csv', 'ERTMS/ETCS Baseline 3 Onboard Subsystem Requirements.csv', 'BSI Standard_Railway Acpplications - The Specification and Demonstration of RAMS.csv'],
    'Document text': [file1_content, file2_content, pdf_data_content, pdf_data2_content, pdf_data3_content]
})





stop = stopwords.words('english')

custom_stopwords = ["ï","»","¿","â€“","â€™","a)","â","¿","–","–","b)","c)","d)","e)",":","(",")","â€˜","-",'must','used','using'
                   'near']
# punctation=[":","(",")"]
stop.extend(custom_stopwords)



main_df['Clean_documents_rules'] = main_df['Document text'].str.lower()



def token_sent(text):   
    sent_tokens=LineTokenizer(blanklines='keep').tokenize(text)
    return sent_tokens



main_df['Sentence_Tokenize_rules']=main_df['Clean_documents_rules'].apply(token_sent) 
main_df['Word_Tokenize_rules']=main_df['Clean_documents_rules'].apply(word_tokenize) 

token_doc_name = dict(zip(main_df['Document name'], main_df['Sentence_Tokenize_rules']))



#Create an embedding for all the sentences in the documents
model = SentenceTransformer('sentence-transformers/stsb-distilbert-base')


# finding the cosine similarity
def cosine_sim(embeddings1,embeddings2):
    """Cosine similarity metric function to calculate the distance between the two vectors."""
    cossim=( np.dot(embeddings1,embeddings2) )/ (np.linalg.norm(embeddings1)*np.linalg.norm(embeddings2))
    if np.isnan(np.sum(cossim)):
        return 0
    return cossim


docs_sent_tokens=list(chain.from_iterable(main_df['Sentence_Tokenize_rules']))



app = FastAPI()

@app.get("/vectorize")
async def vectorize(search_sentence: str):
    # Tokenize the sentence
    search_sentence_embeddings = model.encode(search_sentence)

    # set the threshold value to get the similarity result accordingly
    threshold=0.5

    results = []
    # embedding all the documents and find the similarity between search text and all the tokenize sentences
    for doc_name, docs_sent_tokens in token_doc_name.items():
        for docs_sent_token in docs_sent_tokens:
            sentence_embeddings = model.encode(docs_sent_token)
            sim_score = cosine_sim(search_sentence_embeddings, sentence_embeddings)
            if sim_score > threshold:
                results.append({
                    'Document Name': doc_name,
                    'Matching Sentence': docs_sent_token,
                    'Similarity Score': sim_score,
                })
    
    # sorting in descending order based on the similarity score
    results.sort(key=lambda x: x['Similarity Score'], reverse=True)

    # change the value of n to see more results
    top_n = 20
    results_top_n = results[:top_n]
    
    df = pd.DataFrame(results_top_n, columns=['Document Name', 'Matching Sentence', 'Similarity Score'])

    return df.to_dict(orient='records')



