import cohere

from weaviate.util import generate_uuid5

from langchain.llms import Cohere
from langchain.chat_models import ChatCohere
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.cohere import CohereEmbeddings


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Weaviate
from langchain.vectorstores import Chroma, FAISS
import weaviate

from langchain.docstore.document import Document

from langchain.agents import initialize_agent
from langchain.agents import AgentType

from langchain.memory import ConversationBufferMemory

from langchain.retrievers import ContextualCompressionRetriever, CohereRagRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from langchain.chains.summarize import load_summarize_chain

from unidecode import unidecode

import PyPDF2
import time
import os
import json
import googlemaps
from datetime import datetime

# google map api: https://github.com/googlemaps/google-maps-services-python --------------------------------


# Utilities ------------------------------
def get_text_from_pdf(fileobj):
    #create reader variable that will read the pdffileobj
    reader = PyPDF2.PdfReader(fileobj)
    
    #This will store the number of pages of this pdf file
    num_pages = len(reader.pages)
    
    combined_text = ''
    for i in range(num_pages):
        # create a variable that will select the selected number of pages
        pageobj = reader.pages[i]
        text = unidecode(pageobj.extract_text())    # remove unnecessary unicode characters
        combined_text += text
        combined_text += '\n'

    return combined_text

# Response  ------------------------------
def get_summary(text):
    '''Return summary using co.summarize endpoint and a two-stage map-reduce approach'''
    
    #split text recursively
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=50000, chunk_overlap=1000)
    splits = text_splitter.split_text(text)

    co = cohere.Client(os.environ["COHERE_API_KEY"]) # This is your trial API key

    section_summaries = []
    for t in splits[:-1]:
        response = co.summarize( 
            text=t,
            length='long',
            format='auto',
            model='command',
            additional_command='focusing on the section summary and details',
            temperature=0,
        )
        section_summaries.append(response)
        section_summaries.append(response.summary)

    combined_section_summaries = '\n\n New Section Summary \n\n'.join(section_summaries)

    with open('section_summaries', 'w') as f:
        f.write(combined_section_summaries)

    response = co.summarize( 
            text=combined_section_summaries,
            length='long',
            format='auto',
            model='command',
            additional_command='combined the section summaries with focus on client, project scope/description, project location and expected timeline',
            temperature=0,
        )
    
    return response.summary

def get_location(summary):
    '''Return location to feed into Google Maps API'''

    co = cohere.Client(os.environ["COHERE_API_KEY"]) # This is your trial API key

    prompt = f'''
        Extract the address, preferably the nearest intersection, from the following project summary. 
        The location should be readable by Google Maps API to obtain its coordinates.
        Only output the address without any other descriptions.

        Project Summary:
        {summary}
    '''

    response = co.generate(
        model='command',
        prompt=prompt,
        max_tokens=330,
        temperature=0,
        k=0,
        stop_sequences=[],
        return_likelihoods='NONE'
    )
        
    return response.generations[0].text

def get_coordinates(intersection):
    '''Get coordinates from Google Maps API'''

    print(intersection)

    gmaps = googlemaps.Client(key=os.environ["GOOGLE_API_KEY"])

    # Geocoding an address
    geocode_result = gmaps.geocode(intersection)

    try:
        location = geocode_result[0]['geometry']['location']
        return [c for c in location.values()]
    
    except:
        return [43.65349167474285, -79.38440687827095]

def chat_from_database(prompt: str, chat_history: list=[], summary: str='') -> str:
    ''' Return response based on the given input '''

    print(chat_history)
    print()

    co = cohere.Client(os.environ['COHERE_API_KEY']) # This is your trial API key
    response = co.chat( 
        chat_history=chat_history,
        preamble_override='You are an engineering proposal expert with experience in the infrastructure industry',
        message=prompt,
        prompt_truncation='OFF',
        connectors=[{"id": "weaviate-cfa-proposal-xyt464"}],
        return_chat_history=True
    )

    # print(response.chat_history)

    return response.text