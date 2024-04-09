#!/usr/bin/env python

### TODO:
### break pdf txt into chapters w/ book and chapter summaries
### Q: what are max token in sizes per model? A: Mixtral-8x7b = 32k token context 
### botname_cmd.summary(path_filename, template_additions)
### botname_cmd.injest(path_filename, template_additions)
##### function cleanup...
##### misc cleanup

# GerBot project is an LLM RAG chat intended to make http://gerrystahl.net/pub/index.html even more accessible
# Generative AI "chat" about the gerrystahl.net writings
# Code by Zake Stahl
# https://github.com/zake8/GerryStahlWritings
# March, April 2024
# Based on public/shared APIs and FOSS samples
# Built on Linux, Python, Apache, WSGI, Flask, LangChain, Ollama, Mistral, more

from flask import Flask, redirect, url_for, render_template, request
app = Flask(__name__)

from dotenv import load_dotenv
load_dotenv('./.env')

##### # ConversationBufferWindowMemory setup
##### from langchain.memory import ConversationBufferWindowMemory
##### memory = ConversationBufferWindowMemory(k=14)

import logging
logging.basicConfig(
    filename = 'convo_rag_agent_retreival_chatbot.log', 
    level = logging.INFO, 
    filemode = 'a', 
    format = '%(asctime)s -%(levelname)s - %(message)s')
import socket
import random
import os
import re
##### import json
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader
##### from langchain.chains import RetrievalQA
##### from langchain.chains import create_retrieval_chain
##### from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import JSONLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
##### from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import RunnableParallel
from langchain_core.runnables import RunnablePassthrough
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings
from mistralai.client import MistralClient ### is this used?
from mistralai.models.chat_completion import ChatMessage ### is this used?

# +++++++++++++++++++++++++++
# initialize global variables
chatbot = (f'GerBot')
user_username_in_chat = "User"
rag_source_clue_value = 'docs/rag_source_clues.txt' # doc helps llm choose rag file
my_chunk_size = 250
my_chunk_overlap = 37
# chunk_size= and chunk_overlap, what should they be, how do they relate to file size, word/token/letter count?
# what should overlap % be to retain meaning and searchability?
fullragchat_history = []
fullragchat_rag_source = "docs/nothing.txt"
# +++++++++++++++++++++++++++
logging.info(f'===> Starting main')

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

def get_rag_text(query): # loads from loader fullragchat_rag_source path/file w/ .txt .html .pdf or .json 
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
        answer = "Unable to make loader for " + fullragchat_rag_source
        return answer
    # from https://docs.mistral.ai/guides/basic-RAG/
    docs = loader.load() # docs is a list...
    return docs

def rag_text_function(query):
    # function ignores passed query value
    # rag_source_clues is global defined in chat_query_return func
    loader = TextLoader(rag_source_clues)
    context = loader.load()
    return context

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

summary_template = """
In clear and concise language, summarize (key points, themes presented, interesting terms or jargon (if any) ) the text. 
<text>
{question}
</text>
Summary:
"""

def create_summary(to_sum, model, mkey, fullragchat_temp):
    prompt = ChatPromptTemplate.from_template(summary_template)
    llm = ChatMistralAI(
            model_name=model, 
            mistral_api_key=mkey, 
            temperature=fullragchat_temp )
    chain = ( prompt | llm | StrOutputParser() )
    summary = chain.invoke(to_sum)
    return summary

gerbot_template = """
You are the RAG conversational chatbot "GerBot". (RAG is Retrieval Augmented GenerativeAI.)
Your prime goal is to assist users with exploring, searching, querying, and "chatting with" 
Gerry Stahl's published works, all available here, http://gerrystahl.net/pub/index.html.
If you do not know the answer or know how to respond just say, 
I don't know, or I don't know how to respond to that, or 
you can you ask user to rephrase the question, or 
maybe occationally share an interesting tidbit of wisdom from the writtings.
Try not to be too verbose, flowery, or chatty.
Answer the question based primarily on this relevant retrieved context: 
{context}
Reference chat history for conversationality (
    to see if there is something to circle back to, 
    help drill down into volumes and chapters by directing a query of the same, 
    but not to reply on by repeating your own possibly mistaken statments): 
{history}
Question: 
{question}
Answer:
"""

def injest_document():
    pattern = r'\.([a-zA-Z]{3,5})$'
    match = re.search(pattern, fullragchat_rag_source) # global
    if not match:
        answer = f'There is no extension found on "{fullragchat_rag_source}"'
        return answer
    rag_ext = match.group(1)
    embeddings = MistralAIEmbeddings(
                model=fullragchat_embed_model, 
                mistral_api_key=mkey)
    rag_text = get_rag_text(query)
    summary_text_for_cur = create_summary(
        to_sum=rag_text, 
        model=model, 
        mkey=mkey, 
        fullragchat_temp=fullragchat_temp)
    base_fn = fullragchat_rag_source[:-(len(rag_ext)+1)]
    ### write _loadered.txt to disk
    txtfile_fn = ''
    ### txtfile_fn = f'{base_fn}_loadered.txt'
    ### rag_text_srt = ' '.join(rag_text) # must be str, not list...
    ### with open(txtfile_fn, 'a') as file: # 'a' = append, create new if none
    ###     file.write(rag_text_srt)
    ### logging.info(f'===> saved new .txt file, "{txtfile_fn}"')
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=my_chunk_size, chunk_overlap=my_chunk_overlap)
    documents = text_splitter.split_documents(rag_text)
    faiss_index_fn = f'{base_fn}.faiss'
    vector = FAISS.from_documents(documents, embeddings)
    # Could not load library with AVX2 support due to: ModuleNotFoundError("No module named 'faiss.swigfaiss_avx2'")
    vector.save_local(faiss_index_fn)
    logging.info(f'===> saved new FAISS, "{faiss_index_fn}"')
    # write new .cur file
    curfile_fn = f'{base_fn}.cur'
    date_time = datetime.now()
    curfile_content  = f'\nCuration content for HITL use. \n\n'
    curfile_content += f'Date and time      = {date_time.strftime("%Y-%m-%d %H:%M:%S")} \n'
    curfile_content += f'Target document    = {fullragchat_rag_source} \n'
    curfile_content += f'Saved FAISS DB     = {faiss_index_fn} \n'
    curfile_content += f'# vectors in DB    = {vector.index.ntotal} \n'
    curfile_content += f'Model/temp DB      = {fullragchat_embed_model} / {fullragchat_temp} \n'
    curfile_content += f'Model/temp summary = {model} / {fullragchat_temp} \n'
    curfile_content += f'\n<summary>\n{summary_text_for_cur}\n</summary>\n'
    with open(curfile_fn, 'a') as file: # 'a' = append, create new if none
        file.write(curfile_content)
    logging.info(f'===> saved new .cur file, "{curfile_fn}"')
    # add name and summary to rag source clue file for LLM to use!
    faiss_index_fn = faiss_index_fn[5:] # strip off leading 'docs/' so as not to double it up later
    clue_file_text  = '\n'
    clue_file_text += '  { \n'
    clue_file_text += '    "rag_item": { \n'
    clue_file_text += '      "filename": "' + faiss_index_fn + '", \n'
    clue_file_text += '      "title": "", \n'
    clue_file_text += '      "volume": "", \n'
    clue_file_text += '      "chapter": "", \n'
    clue_file_text += '      "summary": "' + summary_text_for_cur + '", \n'
    clue_file_text += '      "txt_filename": "' + txtfile_fn + '", \n'
    clue_file_text += '    } \n'
    clue_file_text += '  } \n'
    clue_file_text += '\n'
    with open(rag_source_clue_value, 'a') as file: # 'a' = append, file pointer placed at end of file
        file.write(clue_file_text)
    logging.info(f'===> Added new .faiss and summary to "{rag_source_clue_value}"')
    ##### retriever = vector.as_retriever()
    return None

def mistral_convo_rag(fullragchat_embed_model, mkey, model, fullragchat_temp, query):
    # load existing faiss, and use as retriever
    # Potentially dangerious - load only local known safe files
    ### need to implement this safety check!
    ### if fullragchat_rag_source contains http or double wack "//" then set answer = 'illegal faiss source' and return
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
    if query.startswith(f'{chatbot}_command.': # check for overrides
        ### check if user is an admin
        meth = '' ###
        path_filename = '' ###
        if meth == 'summary': # Takes X and returns summary
            ### sanity check that filename is docs/ and ends in pdf html txt 
            answer = f'Summary of "{path_filename}": ' + '\n'
            fullragchat_rag_source = path_filename
            some_text_blob = get_rag_text(query)
            answer += create_summary(
                to_sum=some_text_blob, 
                model=model, 
                mkey=mkey, 
                fullragchat_temp=fullragchat_temp )
            return answer
        elif meth == 'injest': # Saves X as .txt and .faiss w/ .cur file and adds to rag_source_clue_value
            ### sanity check that filename is docs/ and ends in pdf html txt 
            fullragchat_rag_source = path_filename
            injest_document()
            answer = f'Injested "{path_filename}".'
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
    pattern = r'\b[A-Za-z0-9_-]+\.[A-Za-z0-9]{3,5}\b'
    filenames = re.findall(pattern, selected_rag)
    if filenames:
        clean_selected_rag = filenames[0]
        answer += f'Selecting document "{clean_selected_rag}". '
        fullragchat_rag_source = f'docs/{clean_selected_rag}'
    else:
        answer += 'Unable to parse out a filename from:\n"' + selected_rag + '"\n'
        fullragchat_rag_source = f'docs/nothing.txt'
    if not os.path.exists(fullragchat_rag_source):
        answer += f'The file selected does not exist...'
        fullragchat_rag_source = f'docs/nothing.txt'
    logging.info(f'===> Second/last of the double LLM pass')
    ### sanity check that filename is docs/ and ends in .txt or .faiss
    answer += mistral_convo_rag(
        fullragchat_embed_model=fullragchat_embed_model, 
        mkey=mkey, 
        model=model, 
        fullragchat_temp=fullragchat_temp, 
        query=query )
    return answer

"""
Reimplement:
    pattern = r'\.([a-zA-Z]{3,5})$'
    match = re.search(pattern, fullragchat_rag_source) # global
    if not match:
        answer = f'There is no extension found on "{fullragchat_rag_source}"'
        return answer
    rag_ext = match.group(1)
    if (rag_ext == 'txt') or (rag_ext == 'pdf') or (rag_ext == 'html') or (rag_ext == 'htm') or (rag_ext == 'json'): # doc to injest
    elif rag_ext =='faiss': 
    
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
    fullragchat_history.append({'user':'-reset--', 'message':'reset'})
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
    fullragchat_history.append({'user':'-reset', 'message':'pending - please wait for model inferences - small moving graphic on browser tab should indicate working'}) 

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
        fullragchat_embed_model,
    )
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
