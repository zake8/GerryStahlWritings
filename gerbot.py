#!/usr/bin/env python

### TODO:
### Ability to load a (small) text file as a rag doc and hit LLM w/ whole thing, no vector query 

# GerBot project is an LLM RAG chat intended to make http://gerrystahl.net/pub/index.html even more accessible
# Generative AI "chat" about the gerrystahl.net writings
# Code by Zake Stahl
# https://github.com/zake8/GerryStahlWritings
# March, April 2024
# Based on public/shared APIs and FOSS samples
# Built on Linux, Python, Apache, WSGI, Flask, LangChain, Ollama, Mistral, more

# initialize global variables
user_username_in_chat = "User"
docs_dir = 'gerbot' # ex: 'docs' or '/home/leet/GerryStahlWritings/docs' but not 'docs/'
chatbot = f'GerBot'
my_chunk_size = 300 # chunk_size= and chunk_overlap, what should they be, how do they relate to file size, word/token/letter count?
my_chunk_overlap = 100 # what should overlap % be to retain meaning and search-ability? # https://chunkviz.up.railway.app/
my_map_red_chunk_size = 50000 # This is for map reduce summary, the largest text by character length to try to send
# Mixtral-8x7b is a max context size of 32k tokens
rag_source_clue_value = f'{docs_dir}/rag_source_clues.txt' # doc helps llm choose rag file
# change these in fullragchat_init() or UI:
fullragchat_history = []
fullragchat_rag_source = f'{docs_dir}/nothing.txt' # no rag doc selected
fullragchat_model = ''
fullragchat_temp = ''
fullragchat_stop_words = ''
fullragchat_embed_model = ''
query = ''

import logging
logging.basicConfig(
    filename = 'convo_rag_agent_retrieval_chatbot.log', 
    level = logging.INFO, 
    filemode = 'a', 
    format = '%(asctime)s -%(levelname)s - %(message)s')
logging.info(f'===> Starting main')

from flask import Flask, redirect, url_for, render_template, request
app = Flask(__name__)

from dotenv import load_dotenv
load_dotenv('./.env')

import socket
import random
import os
import re
import requests
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import JSONLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import RunnableParallel
from langchain_core.runnables import RunnablePassthrough
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings

gerbot_template = """
You are the RAG conversational chatbot "GerBot". (RAG is Retrieval Augmented GenerativeAI.)
Your prime goal is to assist users with exploring, searching, querying, and "chatting with" 
Gerry Stahl's published works, all available here, http://gerrystahl.net/pub/index.html.
If you do not know the answer or know how to respond just say, 
I don't know, or I don't know how to respond to that, or 
you can you ask user to rephrase the question, or 
maybe rarely occasionally share an interesting tidbit of wisdom from the writings.
Try not to be too verbose, flowery, or chatty.
Answer the question based primarily on this relevant retrieved context: 
{context}
Reference chat history for conversationality (
    to see if there is something to circle back to, 
    help drill down into volumes and chapters by directing a query of the same, 
    but not to reply on by repeating your own possibly mistaken statements): 
{history}
Question: 
{question}
Answer:
"""

filename_template = """
Your task is to return a filename from the provided list.
Mentioning a book or even chapter title should be enough to return its filename.
Example: If question from user is about alphabet, A B C's, and provided list has item with summary about letters in the alphabet, then answer with "alphabet.txt", assuming that is the filename for that summary.
Example: If user is questioning about Python programming, and there is a summary including Python stuff, then return "py_coding.txt" or whatever its name is.
Example: If question is about "The Things" by Peter Watts, then send "TheThings-PeterWatts.faiss".
Question from user is: 
{question}
Lightly reference this chat history help understand what information area user is looking to explore: 
{history}
Here is provided list containing filenames for various content/information areas: 
{context}
Single filename value:
"""

summary_template = """
In clear and concise language, summarize the text 
    (key main points, 
    themes or topic presented, 
    intended audience or purpose, 
    interesting terms or jargon if any). 
Summary needs to include just enough to give an inkling of the source, 
only a brief hint to lead the reader to the full text to read and search that directly. 
In a few sentences, summarize the main idea or argument of the text, 
then include the most important supporting crucial details, all while keeping the summary surprisingly concise.
Do not "write another book", ie. don't write a summary as long as the text it's summarizing. 
Use terse (but coherent) language and don't repeat anything; 
sentences fragments and dropping words like the document, the author, is preferred. 
Please make summary as short as possible. 
(Stick to the presented, and accurately represent the author's intent.)
Keep the summary focused on the most essential elements of the text; 
aim for brevity while capturing all key points. 
If you encounter <INDIVIDUAL SUMMARY> START and END tags, 
then this is the reduce pass of a larger map/reduce sequence, 
so gently consolidate all the individual summary into one massive summary.
<text>
{question}
</text>
Summary:
"""

@app.route("/")
def root():
    webserver_hostname = socket.gethostname()
    return render_template('staging.html', webserver_hostname=webserver_hostname)

@app.route("/gerbotsamples")
def gerbotsamples():
    return render_template('gerbotsamples.html')

def convo_mem_function(query):
    # ignoring query and generating text history from global fullragchat_history dict
    history = f'<chat_history>\n'
    for line in fullragchat_history:
        history += f'{line}\n'
    history += f'</chat_history>\n'
    return history

def get_rag_text(query, start_page, end_page): # loads from loader fullragchat_rag_source path/file w/ .txt .html .pdf or .json 
    # function ignores passed query value
    pattern = r'\.([a-zA-Z]{3,5})$'
    match = re.search(pattern, fullragchat_rag_source) # global
    rag_ext = match.group(1)
    # https://python.langchain.com/docs/modules/data_connection/document_loaders/json
    if rag_ext == "txt":
        loader = TextLoader(fullragchat_rag_source) # ex: /path/filename
    elif (rag_ext == "html") or (rag_ext == "htm"):
        loader = WebBaseLoader(fullragchat_rag_source) # ex: https://url/file.html
    elif rag_ext == "pdf":
        loader = PyPDFLoader(fullragchat_rag_source) 
    elif rag_ext == "json":
        loader = JSONLoader(file_path=fullragchat_rag_source,
            jq_schema='.',
            text_content=False)
    else:
        return f'Unable to make loader for "{fullragchat_rag_source}"!\n '
    # from https://docs.mistral.ai/guides/basic-RAG/
    docs = loader.load() # docs is a type 'document'...
    if start_page and end_page:
        # Reduce docs to just the desired pages
        docs = docs[ int(start_page) - 1 : int(end_page) ]
    return docs

def rag_text_function(query):
    # function ignores passed query value
    # rag_source_clues is global defined in chat_query_return func
    loader = TextLoader(rag_source_clues)
    context = loader.load()
    return context

def choose_rag(mkey, model, fullragchat_temp, query): # chain which chooses a file for you
    rag_text_runnable = RunnableLambda(rag_text_function)
    history_runnable = RunnableLambda(convo_mem_function)
    setup_and_retrieval = RunnableParallel({
        "context": rag_text_runnable, 
        "question": RunnablePassthrough(),
        "history": history_runnable })
    prompt = ChatPromptTemplate.from_template(filename_template)
    large_lang_model = ChatMistralAI(
                model_name=model, 
                mistral_api_key=mkey, 
                temperature=float(fullragchat_temp) )
    output_parser = StrOutputParser()
    chain = ( setup_and_retrieval | prompt | large_lang_model | output_parser )
    selected_rag = chain.invoke(query)
    return selected_rag

def create_summary(to_sum, model, mkey, fullragchat_temp):
    prompt = ChatPromptTemplate.from_template(summary_template)
    llm = ChatMistralAI(
            model_name=model, 
            mistral_api_key=mkey, 
            temperature=fullragchat_temp )
    chain = ( prompt | llm | StrOutputParser() )
    try:
        summary = chain.invoke(to_sum)
    except Exception as err_mess:
        # returns 'Request size limit exceeded' in the form of "KeyError: 'choices'" in chat_models.py
        logging.error(f'===> Got error: {err_mess} when invoking summary chain')
        summary = f'Got error: "{err_mess}". Likely this is "Request size limit exceeded" in the form of "KeyError: choices" in chat_models.py. Maybe try smaller "map_red_chunk_size".'
    return summary

def create_map_reduce_summary(to_sum, map_red_chunk_size, model, mkey, fullragchat_temp):
    logging.info(f'===> Starting Map Reduce')
    # Map
    summary = ''
    piece_summaries = ''
    text_splitter = CharacterTextSplitter(
        separator="\n", # should be something else?
        chunk_size = map_red_chunk_size, 
        chunk_overlap = max(map_red_chunk_size // 5, 500),
        length_function=len, 
        is_separator_regex=False )
    to_sum_str = ''
    for index in range(0, len(to_sum)):
        to_sum_str += to_sum[index].page_content + '\n'
    pieces = text_splitter.create_documents([to_sum_str])
    num_pieces = len(pieces)
    for piece in pieces:
        piece_summaries += f'\n\n<INDIVIDUAL SUMMARY START>\n'
        individual_summary = create_summary(
            to_sum = piece, 
            model=model, mkey=mkey, fullragchat_temp=fullragchat_temp )
        logging.info(f'Individual_summary is: \n{individual_summary}')
        piece_summaries += (individual_summary + '\n')
        piece_summaries += f'\n<INDIVIDUAL SUMMARY END>\n\n'
    # Reduce
    logging.info(f'===> Map Reduce Summary with {num_pieces + 1} LLM inferences (character chunk size of "{map_red_chunk_size}"). ')
    summary += create_summary(
        to_sum = piece_summaries, 
        model=model, mkey=mkey, fullragchat_temp=fullragchat_temp )
    return summary

def ingest_document(model, fullragchat_embed_model, mkey, query, fullragchat_temp, start_page, end_page):
    logging.info(f'===> Attempting ingestion on "{fullragchat_rag_source}", with page range "{start_page}" to "{end_page}". (All pages if Nones.)')
    answer = ''
    if not os.path.exists(fullragchat_rag_source):
        answer += f'Source document "{fullragchat_rag_source}" not found locally. '
        return answer
    pattern = r'\.([a-zA-Z]{3,5})$'
    match = re.search(pattern, fullragchat_rag_source) # global
    if not match:
        answer += f'There is no extension found on "{fullragchat_rag_source}"'
        return answer
    rag_ext = match.group(1)
    base_fn = os.path.basename(fullragchat_rag_source) # strip path
    base_fn = base_fn[:-(len(rag_ext)+1)] # strip extension
    if start_page and end_page: # tweak filename to save with pdf page numbers
        base_fn = f'{base_fn}-{start_page}-{end_page}'
    faiss_index_fn = f'{base_fn}.faiss'
    # Check if file to save already exists...
    if os.path.exists(faiss_index_fn): ### This doesn't seem to work, maybe 'cause .faiss is a directory?
        answer += f'{faiss_index_fn} already exists; please delete and then retry. '
        return answer
    # Get text
    rag_text = get_rag_text(
        query=query, 
        start_page=start_page, 
        end_page=end_page )
    answer += f'Read "{fullragchat_rag_source}". '
    # Prep summary
    summary_text_for_output = create_map_reduce_summary(
        to_sum = rag_text, 
        map_red_chunk_size = my_map_red_chunk_size, 
        model=model, mkey=mkey, fullragchat_temp=fullragchat_temp )
    ##### summary_text_for_output = create_summary(
    #####     to_sum = rag_text, 
    #####     model=model, mkey=mkey, fullragchat_temp=fullragchat_temp )
    # Write _loadered.txt to disk
    if rag_ext != 'txt': #don't write out a '_loadered.txt' if input was '.txt'
        txtfile_fn = f'{base_fn}_loadered.txt'
        text_string = ''
        for page_number in range(0, len(rag_text) ):
            text_string += rag_text[page_number].page_content + '\n'
            # LangChain document object is a list, each list item is a dictionary with two keys, 
            # page_content and metadata
        with open(docs_dir + '/' + txtfile_fn, 'a') as file: # 'a' = append, create new if none
            if start_page and end_page:
                file.write(f'Specifically PDF pages {start_page} to {end_page} \n')
            file.write(text_string)
        logging.info(f'===> Saved new .txt file, "{txtfile_fn}"')
        answer += f'Wrote "{txtfile_fn}". '
    else: txtfile_fn = 'None'
    # Write FAISS to disk
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=my_chunk_size, chunk_overlap=my_chunk_overlap)
    documents = text_splitter.split_documents(rag_text)
    embeddings = MistralAIEmbeddings(
                model=fullragchat_embed_model, 
                mistral_api_key=mkey)
    vector = FAISS.from_documents(documents, embeddings)
    ### Could not load library with AVX2 support due to: ModuleNotFoundError("No module named 'faiss.swigfaiss_avx2'")
    vector.save_local(docs_dir + '/' + faiss_index_fn)
    logging.info(f'===> saved new FAISS, "{faiss_index_fn}"')
    answer += f'Wrote "{faiss_index_fn}". '
    # Write new .cur file
    curfile_fn = f'{base_fn}.cur'
    date_time = datetime.now()
    curfile_content  = f'\nCuration content for HITL use. \n\n'
    curfile_content += f'Date and time      = {date_time.strftime("%Y-%m-%d %H:%M:%S")} \n'
    curfile_content += f'Target document    = {fullragchat_rag_source} \n'
    if start_page and end_page:
        curfile_content += f'PDF pages          = {start_page} to {end_page} \n'
    curfile_content += f'Chunk size         = {my_chunk_size} \n'
    curfile_content += f'Chunk overlap      = {my_chunk_overlap} \n'
    curfile_content += f'Saved FAISS DB     = {faiss_index_fn} \n'
    curfile_content += f'# vectors in DB    = {vector.index.ntotal} \n'
    curfile_content += f'Model/temp DB      = {fullragchat_embed_model} / {fullragchat_temp} \n'
    curfile_content += f'Model/temp summary = {model} / {fullragchat_temp} \n'
    curfile_content += f'\n<summary>\n{summary_text_for_output}\n</summary>\n'
    with open(docs_dir + '/' + curfile_fn, 'a') as file: # 'a' = append, create new if none
        file.write(curfile_content)
    logging.info(f'===> saved new .cur file, "{curfile_fn}"')
    answer += f'Wrote "{curfile_fn}". '
    # Add name and summary to rag source clue file for LLM to use!
    strip = len(f'{docs_dir}/')
    clue_file_text  = '\n'
    clue_file_text += '  { \n'
    clue_file_text += '    "rag_item": { \n'
    clue_file_text += '      "filename": "' + faiss_index_fn + '", \n'
    if start_page and end_page:
        clue_file_text += '      "pages": "' + start_page + '" to "' + end_page + '", \n'
    clue_file_text += '      "Title": "", \n'
    clue_file_text += '      "Volume": "", \n'
    clue_file_text += '      "summary": "' + summary_text_for_output + '", \n'
    clue_file_text += '      "txt_filename": "' + txtfile_fn + '", \n'
    clue_file_text += '    } \n'
    clue_file_text += '  } \n'
    clue_file_text += '\n'
    with open(rag_source_clue_value, 'a') as file: # 'a' = append, file pointer placed at end of file
        file.write(clue_file_text)
    logging.info(f'===> Added new .faiss and summary to "{rag_source_clue_value}"')
    answer += f'Updated "{rag_source_clue_value}". '
    return answer

def mistral_convo_rag(fullragchat_embed_model, mkey, model, fullragchat_temp, query):
    # load existing faiss, and use as retriever
    # Potentially dangerous - load only local known safe files
    ### need to implement this safety check!
    ### if fullragchat_rag_source contains http or double wack "//" then set answer = 'illegal faiss source' and return
    embeddings = MistralAIEmbeddings(
            model=fullragchat_embed_model, 
            mistral_api_key=mkey)
    loaded_vector_db = FAISS.load_local(fullragchat_rag_source, embeddings, allow_dangerous_deserialization=True)
    retriever = loaded_vector_db.as_retriever()
    # this sequence seems to use query value so the above, if not in a langchain pipe as a runnable would read vector.as_retriever(query)
    history_runnable = RunnableLambda(convo_mem_function)
    setup_and_retrieval = RunnableParallel({
        "context" : retriever, 
        "question": RunnablePassthrough(),
        "history" : history_runnable })
    prompt = ChatPromptTemplate.from_template(gerbot_template)
    large_lang_model = ChatMistralAI(
                model_name=model, 
                mistral_api_key=mkey, 
                temperature=float(fullragchat_temp) )
    output_parser = StrOutputParser()
    chain = ( setup_and_retrieval | prompt | large_lang_model | output_parser )
    answer = chain.invoke(query)
    return answer

"""
Notes on Ollama:
   ollama = Ollama(
        model=model, 
        temperature=float(fullragchat_temp), 
        stop=stop_words_list, 
        verbose=True,
    )
    oembed = OllamaEmbeddings(model=fullragchat_embed_model)
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=oembed)
    # Chroma sends out Anonymized telemetry
    context_runnable = RunnableLambda(ollama_embed_search)
    history_runnable = RunnableLambda(convo_mem_function)
    chain = ( setup_and_retrieval | prompt | ollama | output_parser )
"""

def fake_llm(query):
    rand = random.randint(1, 15)
    if rand <= 3:
        answer = "The answers are in the questions - " + query + " - Think about it."
    elif rand <= 4:
        answer = "I don't understand, can you rephrase the question?" + query + "???"
    elif rand <= 5:
        answer = "That's a good question. Ask another."
    elif rand <= 6:
        answer = "Positive"
    elif rand <= 7:
        answer = "Negative"
    elif rand <= 8:
        answer = "I didn't quite catch that, again?"
    elif rand <= 9:
        answer = "Maybe, we'll see."
    elif rand <= 13:
        answer = "System down for temporary maintinance; check back in a bit."
    elif rand <= 14:
        answer = "Can you hear me now?"
    else:
        answer = "Go away." + query + "Huh."
    return answer

def chat_query_return(model, query, fullragchat_temp, fullragchat_stop_words, fullragchat_embed_model):
    global fullragchat_rag_source
    answer = ''
    stop_words_list = fullragchat_stop_words.split(', ')
    if stop_words_list == ['']: stop_words_list = None
    mkey = os.getenv('Mistral_API_key')
    ### check if user is an admin !!!
    pattern = r'^chatbot_command\.([a-z]+)\(([^)]+)\)$'
    match = re.search(pattern, query)
    if match: # override for commands
        meth = match.group(1)
        path_filename = match.group(2)
        pattern = r'\.([a-zA-Z]{3,5})$'
        match = re.search(pattern, path_filename) # pulls extension
        if path_filename == 'None' or path_filename == 'none': rag_ext = 'None'
        elif match: rag_ext = match.group(1)
        else: rag_ext = ''
        if (rag_ext != 'None') and (rag_ext != 'pdf') and (rag_ext != 'html') and (rag_ext != 'htm') and (rag_ext != 'txt') and (rag_ext != 'json'):
            answer += f'Error: Invalid extension request, "{rag_ext}".'
            return answer
        else:
            if meth == 'summary': # output to chat only
                answer += f'Summary of "{path_filename}": ' + '\n'
                fullragchat_rag_source = path_filename
                some_text_blob = get_rag_text(
                    query=query, 
                    start_page=None, 
                    end_page=None )
                answer += create_summary(
                    to_sum=some_text_blob, 
                    model=model, mkey=mkey, fullragchat_temp=fullragchat_temp )
                return answer
            if meth == 'mapreducesummary': # output to chat only
                answer += f'Map reduce summary of "{path_filename}": ' + '\n'
                fullragchat_rag_source = path_filename
                some_text_blob = get_rag_text(query=query, start_page=None, end_page=None )
                answer += create_map_reduce_summary(
                    to_sum = some_text_blob, 
                    map_red_chunk_size = my_map_red_chunk_size, 
                    model=model, mkey=mkey, fullragchat_temp=fullragchat_temp )
                return answer
            elif meth == 'ingest': # from web or local - saves X as .faiss (and .txt), w/ .cur file, and adds to rag_source_clue_value
                fullragchat_rag_source = path_filename
                answer = ingest_document(
                    model=model, 
                    fullragchat_embed_model=fullragchat_embed_model, 
                    mkey=mkey, 
                    query=query, 
                    fullragchat_temp=fullragchat_temp,
                    start_page=None, 
                    end_page=None )
                return answer
            elif meth == 'download': # just save from web to local
                local_filename = docs_dir + '/' + os.path.basename(path_filename)
                if os.path.exists(local_filename):
                    answer += f'{local_filename} already exists; please delete and then retry. '
                    return answer
                response = requests.get(path_filename)
                if response.status_code == 200:
                    with open(local_filename, 'wb') as file:
                        file.write(response.content)
                    answer += f'Downloaded {path_filename} and saved as {local_filename}. '
                else:
                    answer += f'Fail to download {path_filename}, status code: {response.status_code} '
                return answer
            elif meth == 'listfiles': # lists available docs on disk
                extensions = (".faiss")
                answer += f'List of docs in "{docs_dir}" with "{extensions}" extension: '
                for file in os.listdir(docs_dir):
                    if file.endswith(extensions):
                        answer += '"' + file  + '" '
                answer += 'End of list. '
                return answer
            elif meth == 'listclues': # lists available docs as per clues file
                answer += f'List of docs called out in "{rag_source_clue_value}": '
                with open(rag_source_clue_value, 'r') as file:
                    clues_blob = file.read()
                clues = clues_blob.split('\n')
                for item in clues:
                    pattern = r'"filename":\s*"([^"]+\.\w+)"'
                    match = re.search(pattern, item)
                    if match:
                        filename_with_extension = match.group(1)
                        base_filename, extension = filename_with_extension.rsplit('.', 1)
                        if extension == 'faiss':
                            answer += '"' + filename_with_extension  + '" '
                return answer
            elif meth == 'delete': ### deletes X (low priority to build)
                # check if file to save already exists
                # delete file
                answer = f'Delete not implemented; just use ssh or WinSCP. '
                # answer = f'Deleted "{path_filename}".'
                return answer
            elif meth == 'batchingest': # batch ingest from list text file
                if os.path.exists(path_filename):
                    with open(path_filename, 'r') as file:
                        batch_list_str = file.read()
                    batch_list = batch_list_str.split('\n')
                    for item in batch_list:
                        if (item[0:2] == '# ') or (item == '') :
                            pass # skips comments and blank lines
                        else:
                            pattern = r'^([\w./]+),\s*(\d+),\s*(\d+)$'
                            match = re.search(pattern, item)
                            if match: # item is pfn, page, page
                                fullragchat_rag_source = match.group(1)
                                start_page = match.group(2)
                                end_page = match.group(3)
                                answer += ingest_document(
                                    model=model, 
                                    fullragchat_embed_model=fullragchat_embed_model, 
                                    mkey=mkey, 
                                    query=query, 
                                    fullragchat_temp=fullragchat_temp,
                                    start_page=start_page,
                                    end_page=end_page )
                            else:
                                pattern = r'^([\w./]+)$'
                                match = re.search(pattern, item)
                                if match: # item is pfn only w/ no page numbers
                                    fullragchat_rag_source = match.group(1)
                                    answer += ingest_document(
                                        model=model, 
                                        fullragchat_embed_model=fullragchat_embed_model,
                                        mkey=mkey, 
                                        query=query, 
                                        fullragchat_temp=fullragchat_temp,
                                        start_page=None,
                                        end_page=None )
                                else:
                                    answer += f'Can not process: "{item}" '
                else:
                    answer += f'Unable to batch from non-existent (local) file: "{path_filename}".'
                return answer
            else: # Invalid command
                answer += 'Error: Invalid command.'
                return answer
    logging.info(f'===> Starting first double LLM pass')
    # Figure out which staged rag doc to use
    global rag_source_clues
    rag_source_clues = rag_source_clue_value # doc helps llm choose rag file
    selected_rag = choose_rag(
        mkey=mkey, 
        model=model, 
        fullragchat_temp=fullragchat_temp, 
        query=query )
    logging.info(f'===> selected_rag: {selected_rag}')
    # Should be "return_filename.faiss" or the like; sometimes LLM is chatty tho
    # Comments from LLM show in log, and in chat if unable to parse
    # sanity check that filename is docs/ and ends in .faiss follows
    pattern = r'\b[A-Za-z0-9_-]+\.[A-Za-z0-9]{3,5}\b'
    filenames = re.findall(pattern, selected_rag)
    if filenames:
        clean_selected_rag = filenames[0]
        answer += f'Selecting document "{clean_selected_rag}". '
        fullragchat_rag_source = f'{docs_dir}/{clean_selected_rag}'
        if not os.path.exists(fullragchat_rag_source):
            answer += f'The file selected does not exist... '
            fullragchat_rag_source = f'{docs_dir}/nothing.faiss'
        pattern = r'\.([a-zA-Z]{3,5})$'
        match = re.search(pattern, clean_selected_rag)
        if match:
            rag_ext = match.group(1)
            if rag_ext != 'faiss':
                answer += f'.faiss is required at this point... '
                fullragchat_rag_source = f'{docs_dir}/nothing.faiss'
        else:
            answer += f'There is no extension found on "{fullragchat_rag_source}". '
            fullragchat_rag_source = f'{docs_dir}/nothing.faiss'
    else:
        answer += 'Unable to parse out a filename from:\n"' + selected_rag + '"\n'
        fullragchat_rag_source = f'{docs_dir}/nothing.faiss'
    logging.info(f'===> Second/last of the double LLM pass')
    answer += mistral_convo_rag(
        fullragchat_embed_model=fullragchat_embed_model, 
        mkey=mkey, 
        model=model, 
        fullragchat_temp=fullragchat_temp, 
        query=query )
    return answer

Reimplement = """
    if model == "fake_llm":
        answer = fake_llm(query)
    elif (model == "orca-mini") or 
        (model == "phi") or 
        (model == "tinyllama") or
        (model == "llama2") or 
        (model == "llama2-uncensored") or 
        (model == "mistral") or 
        (model == "mixtral") or 
        (model == "command-r") ):
    elif ( (model == "open-mixtral-8x7b") or 
        model == "mistral-large-latest") or 
        model == "open-mistral-7b") ):
    else:
        answer = "No LLM named " + model
"""

@app.route("/reset_history")
def reset_fullragchat_history():
    global fullragchat_history
    global fullragchat_model
    global fullragchat_temp
    global fullragchat_stop_words
    global fullragchat_embed_model
    global fullragchat_skin 
    global fullragchat_music
    fullragchat_history.clear()
    fullragchat_history.append({'user':'System', 'message':'Reset.'})
    return render_template('fullragchat.html', 
        fullragchat_history=fullragchat_history, 
        fullragchat_model=fullragchat_model, 
        fullragchat_temp=fullragchat_temp, 
        fullragchat_stop_words=fullragchat_stop_words, 
        fullragchat_rag_source=fullragchat_rag_source, 
        fullragchat_embed_model=fullragchat_embed_model,
        fullragchat_skin=fullragchat_skin,
        fullragchat_music=fullragchat_music,
    )

def init_fullragchat_history():
    global fullragchat_history
    reset_fullragchat_history()
    fullragchat_history.clear()
    fullragchat_history.append({'user':chatbot, 'message':'Hi!'}) 
    fullragchat_history.append({'user':chatbot, 'message':"Lets chat about Gerry Stahl's writing."}) 
    fullragchat_history.append({'user':chatbot, 'message':'Enter a question, and click query.'}) 

@app.route("/fullragchat_init")
def fullragchat_init():
    global fullragchat_history
    global fullragchat_model
    global fullragchat_temp
    global fullragchat_stop_words
    global fullragchat_rag_source
    global fullragchat_embed_model
    global fullragchat_skin 
    global fullragchat_music
    fullragchat_history = []
    fullragchat_model = "open-mixtral-8x7b"
    fullragchat_temp = "0.25"
    fullragchat_rag_source = "Auto"
    fullragchat_embed_model = "mistral-embed"
    fullragchat_stop_words = ""
    fullragchat_skin = "not yet implemented"
    fullragchat_music = "not yet implemented"
    init_fullragchat_history()
    logging.info(f'===> Set initial values, starting html loop')
    return render_template('fullragchat.html', 
        fullragchat_history=fullragchat_history,
        fullragchat_model=fullragchat_model,
        fullragchat_temp=fullragchat_temp,
        fullragchat_stop_words=fullragchat_stop_words, 
        fullragchat_rag_source=fullragchat_rag_source, 
        fullragchat_embed_model=fullragchat_embed_model,
        fullragchat_skin=fullragchat_skin,
        fullragchat_music=fullragchat_music,
    )

def pending_fullragchat_history():
    global fullragchat_history
    fullragchat_history.append({'user':'System', 'message':'pending - please wait for model inferences - small moving graphic on browser tab should indicate working'}) 

def unpending_fullragchat_history():
    global fullragchat_history
    if fullragchat_history: fullragchat_history.pop()

@app.route("/fullragchat_pending", methods=['POST'])
def fullragchat_pending():
    # fullragchat.html submits here
    global fullragchat_history
    global query
    global fullragchat_model
    global fullragchat_temp
    global fullragchat_stop_words
    global fullragchat_rag_source
    global fullragchat_embed_model
    global fullragchat_skin 
    global fullragchat_music
    query = request.form['query']
    fullragchat_model = request.form['model']
    fullragchat_temp = request.form['temp']
    fullragchat_stop_words = request.form['stop_words']
    fullragchat_rag_source = request.form['rag_source']
    fullragchat_embed_model = request.form['embed_model']
    fullragchat_skin = request.form['skin']
    fullragchat_music = request.form['music']
    fullragchat_history.append({'user':user_username_in_chat, 'message':query}) 
    logging.info(f'===> {user_username_in_chat}: {query}')
    pending_fullragchat_history()
    return render_template('fullragchat_pending.html', 
        fullragchat_history=fullragchat_history, 
        fullragchat_model=fullragchat_model,
        fullragchat_temp=fullragchat_temp, 
        fullragchat_stop_words=fullragchat_stop_words, 
        fullragchat_rag_source=fullragchat_rag_source, 
        fullragchat_embed_model=fullragchat_embed_model,
        fullragchat_skin=fullragchat_skin,
        fullragchat_music=fullragchat_music,
    )

@app.route("/fullragchat_reply")
def fullragchat_reply():
    # fullragchat_pending.html refreshes here
    global fullragchat_history
    global query
    global fullragchat_model
    global fullragchat_temp
    global fullragchat_stop_words
    global fullragchat_embed_model
    global fullragchat_skin 
    global fullragchat_music
    unpending_fullragchat_history()
    logging.info(f'===> model info: model={fullragchat_model}, temp={fullragchat_temp}, stop={fullragchat_stop_words}, rag={fullragchat_rag_source}, embed={fullragchat_embed_model}')
    answer = chat_query_return(
        fullragchat_model, 
        query, 
        fullragchat_temp, 
        fullragchat_stop_words, 
        fullragchat_embed_model )
    fullragchat_history.append({'user':chatbot, 'message':answer})
    logging.info(f'===> {chatbot}: {answer}')
    return render_template('fullragchat.html', 
        fullragchat_history=fullragchat_history, 
        fullragchat_model=fullragchat_model, 
        fullragchat_temp=fullragchat_temp, 
        fullragchat_stop_words=fullragchat_stop_words, 
        fullragchat_rag_source=fullragchat_rag_source, 
        fullragchat_embed_model=fullragchat_embed_model,
        fullragchat_skin=fullragchat_skin,
        fullragchat_music=fullragchat_music,
        )

if __name__ == "__main__":
    app.run()
