#!/usr/bin/env python

# TODO:
# break pdf txt into chapters w/ book and chapter summaries
# Q: what are max token in sizes per model? A: Mixtral-8x7b = 32k token context 
##### cleanup

### GerBot project is an LLM RAG chat intended to make http://gerrystahl.net/pub/index.html even more accessible
### Generative AI "chat" about the gerrystahl.net writings
### Code by Zake Stahl
### https://github.com/zake8/GerryStahlWritings
### March, April 2024
### Based on public/shared APIs and FOSS samples
### Built on Linux, Python, Apache, WSGI, Flask, LangChain, Ollama, Mistral, more

from flask import Flask, redirect, url_for, render_template, request
app = Flask(__name__)

from dotenv import load_dotenv
load_dotenv('./.env')

# ConversationBufferWindowMemory setup
from langchain.memory import ConversationBufferWindowMemory
memory = ConversationBufferWindowMemory(k=14)

import logging
import socket
import random
import os
import re
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import ConversationChain
from langchain.chains import RetrievalQA
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import JSONLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import RunnableParallel
from langchain_core.runnables import RunnablePassthrough
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

# +++++++++++++++++++++++++++
# initialize global variables
chatbot = (f'GerBot')
user_username_in_chat = "User"
fullragchat_rag_source = "Auto"
rag_source_clue_value = 'docs/rag_source_clues.txt' # doc helps llm choose rag file
# rag_source_clue_value = 'docs/rag_summary_link_to_rags.txt' # doc helps llm choose rag file
my_chunk_size=250
my_chunk_overlap=37
# chunk_size= and chunk_overlap, what should they be, how do they relate to file size, word/token/letter count?
# what should overlap % be to retain meaning and searchability?
# +++++++++++++++++++++++++++

logging.basicConfig(
    filename = chatbot + '.log', 
    level = logging.INFO, 
    filemode = 'a', 
    format = '%(asctime)s -%(levelname)s - %(message)s')

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

##### remove this function
def mistral_convochat(model, mkey, fullragchat_temp, query):
    large_lang_model = ChatMistralAI(
        model_name=model, 
        mistral_api_key=mkey, 
        temperature=float(fullragchat_temp) )
    global memory
    chain = ConversationChain(llm=large_lang_model, memory=memory) # ConversationBufferWindowMemory leveraged by ConversationChain
    answer = chain.predict(input=query)
    return answer

##### remove this function
def ollama_convochat(model, fullragchat_temp, stop_words_list, query):
    ollama = Ollama(
        model=model, 
        temperature=float(fullragchat_temp), 
        stop=stop_words_list, 
        verbose=True )
    global memory
    chain = ConversationChain(llm=ollama, memory=memory) # ConversationBufferWindowMemory leveraged by ConversationChain
    answer = chain.predict(input=query)
    return answer

##### remove this function
def mistral_qachat(model, mkey, fullragchat_temp, query):
    # simple chat from https://docs.mistral.ai/platform/client/
    client = MistralClient(api_key=mkey)
    messages = [ ChatMessage(role=user_username_in_chat, content=query) ]
    chat_response = client.chat(
            model=model,
            messages=messages,
            temperature=float(fullragchat_temp) )
    answer = chat_response.choices[0].message.content
    return answer

##### remove this function
def ollama_qachat(model, fullragchat_temp, stop_words_list, query):
    ### instanciate model
    ollama = Ollama(
        model=model, 
        temperature=float(fullragchat_temp), 
        stop=stop_words_list, 
        verbose=True )
    ### invoke model
    answer = ollama(query)
    # answer = ollama.invoke(query) # is this preferred? how does the above know what to do?
    return answer

def get_rag_text(query):
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
    docs = loader.load()
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=my_chunk_size, chunk_overlap=my_chunk_overlap)
    rag_text = text_splitter.split_documents(docs)
    return rag_text

def rag_text_function(query):
    # function ignores passed query value
    # rag_source_clues is global defined in chat_query_return func
    loader = TextLoader(rag_source_clues)
    context = loader.load()
    return context

filename_template = """
Your task is to return, in json format, a single filename, from the provided list.
Mentioning a book or even chapter title should be enough to return its filename.
Please be sure to return "filename": "return_filename.faiss".
Example: If question from user is about alphabet, A B C's, and provided list has item with summary about letters in the alphabet, then answer with "filename": "alphabet.txt", assuming that is the filename for that summary.
Example: If user is questioning about Python programming, and there is a summary including Python stuff, then return "filename": "py_coding.txt" or whatever its name is.
Example: If question is about "The Things" by Peter Watts, then send the string "filename": "TheThings-PeterWatts.txt".
Question from user is: 
{question}
Lightly reference this chat history help understand what information area user is looking to explore: 
{history}
Here is provided list containing filenames for various content/information areas: 
{context}
Single path_filename value:
"""

def choose_rag(mkey, model, fullragchat_temp, query):
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
maybe occationally share an interesting tidbit of wisdom from the 
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

def mistral_convo_rag(fullragchat_embed_model, mkey, model, fullragchat_temp, query):
    pattern = r'\.([a-zA-Z]{3,5})$'
    match = re.search(pattern, fullragchat_rag_source) # global
    if not match:
        answer = f'There is no extension found on "{fullragchat_rag_source}"'
        return answer
    rag_ext = match.group(1)
    logging.info(f'===> rag_ext: "{rag_ext}"')
    embeddings = MistralAIEmbeddings(
                model=fullragchat_embed_model, 
                mistral_api_key=mkey)
    if (rag_ext == 'txt') or (rag_ext == 'pdf') or (rag_ext == 'html') or (rag_ext == 'htm') or (rag_ext == 'json'): # doc to injest
        # Injest new document, save faiss, and use as retriever
        documents = get_rag_text(query)
        faiss_index_fn = f'{fullragchat_rag_source}.faiss'
        # logging.info(f'++++++ documents ++++++++++\n{documents}\n')
        vector = FAISS.from_documents(documents, embeddings)
        vector.save_local(faiss_index_fn)
        logging.info(f'===> saved new FAISS, "{faiss_index_fn}"')
        ### write text to disk
        txtfile_fn = fullragchat_rag_source + '.txt'
        with open(txtfile_fn, 'a') as file: # 'a' = append, create new if none
            file.write(documents)
        logging.info(f'===> saved new .txt file, "{txtfile_fn}"')
        ### write new .cur file
        curfile_fn = fullragchat_rag_source + '.cur'
        date_time = datetime.now()
        summary_text_for_cur = create_summary(
            to_sum=documents, # documents is a list?, whereas this needs a string 
            model=model, 
            mkey=mkey, 
            fullragchat_temp=fullragchat_temp)
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
        ### add name and summary to rag source clue file for LLM to use!
        faiss_index_fn = faiss_index_fn[5:] # strip off leading 'docs/' so as not to double it up later
        clue_file_text  = f'\n'
        clue_file_text += f'  { \n'
        clue_file_text += f'    "rag_item": { \n'
        clue_file_text += f'      "filename": "{faiss_index_fn}", \n'
        clue_file_text += f'      "title": "", \n'
        clue_file_text += f'      "volume": "", \n'
        clue_file_text += f'      "chapter": "", \n'
        clue_file_text += f'      "summary": "{summary_text_for_cur}", \n'
        clue_file_text += f'      "txt_filename": "", \n'
        clue_file_text += f'    } \n'
        clue_file_text += f'  } \n'
        clue_file_text += f'\n'
        ##### cleanup
        # clue_file_text  = f'\n\n'
        # clue_file_text += f'*****{faiss_index_fn}*****\n'
        # clue_file_text += f'<summary text for *****{faiss_index_fn}*****>\n'
        # clue_file_text += f'{summary_text_for_cur}\n'
        # clue_file_text += f'</summary text for *****{faiss_index_fn}*****>\n'
        # clue_file_text += f'\n\n'
        with open(rag_source_clue_value, 'a') as file: # 'a' = append, file pointer placed at end of file
            file.write(clue_file_text)
        logging.info(f'===> Added new .faiss and summary to "{rag_source_clue_value}"')
        retriever = vector.as_retriever()
    elif rag_ext =='faiss':
        # Load existing faiss, and use as retriever
        # Potentially dangerious - load only local known safe files
        ### need to implement this safety check!
        # if fullragchat_rag_source contains http or double wack "//" then set answer = 'illegal faiss source' and return
        loaded_vector_db = FAISS.load_local(fullragchat_rag_source, embeddings, allow_dangerous_deserialization=True)
        retriever = loaded_vector_db.as_retriever()
    else:
        answer = f'Invalid extension on "{fullragchat_rag_source}"...'
        return answer
    # testing_retriever = retriever.invoke(query)
    # logging.info(f'++++++ retriever ++++++++++\n{testing_retriever}\n')
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

##### remove this function
def mistral_rag(fullragchat_embed_model, mkey, model, fullragchat_temp, query):
    documents = get_rag_text(query)
    # Define the embedding model
    embeddings = MistralAIEmbeddings(
                model=fullragchat_embed_model, 
                mistral_api_key=mkey
    )
    # Create the vector store 
    vector = FAISS.from_documents(documents, embeddings)
    # Define a retriever interface
    retriever = vector.as_retriever()
    # Define LLM
    large_lang_model = ChatMistralAI(
                model_name=model, 
                mistral_api_key=mkey, 
                temperature=float(fullragchat_temp), 
    )
    # Define prompt template
    prompt = ChatPromptTemplate.from_template("""
        Answer the following question based only on the provided context:
        <context>
        {context}
        </context>
        Question: {input}
        Answer:
    """)
    # Create a retrieval chain to answer questions
    document_chain = create_stuff_documents_chain(large_lang_model, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    input_query = {}
    input_query['input'] = query
    # invoke chain
    response_dic = retrieval_chain.invoke(input_query)
    answer = response_dic['answer'] # parse return from LLM with input, context, and, answer into just answer
    return answer

def ollama_embed_search(query):
    global fullragchat_embed_model
    oembed = OllamaEmbeddings(model=fullragchat_embed_model)
    all_splits = get_rag_text(query)
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=oembed)
    docs = vectorstore.similarity_search(query)
    # this sequence uses the query to return a few text strings of similar semantics
    # also Chroma sends out Anonymized telemetry https://docs.trychroma.com/telemetry
    # replace with FAISS code!
    vector_store_hits = len(docs)
    context_text = f'<rag_context>\n'
    for line in docs:
        context_text += f'{line}\n'
    context_text += f'</rag_context>\n'
    return context_text

def ollama_convo_rag(model, fullragchat_temp, stop_words_list, fullragchat_embed_model, query):
    context_runnable = RunnableLambda(ollama_embed_search)
    history_runnable = RunnableLambda(convo_mem_function)
    setup_and_retrieval = RunnableParallel({
        "context":  context_runnable, 
        "question": RunnablePassthrough(),
        "history":  history_runnable})
    prompt = ChatPromptTemplate.from_template(gerbot_template)
    ollama = Ollama(
        model=model, 
        temperature=float(fullragchat_temp), 
        stop=stop_words_list, 
        verbose=True,
    )
    output_parser = StrOutputParser()
    chain = ( setup_and_retrieval | prompt | ollama | output_parser )
    answer = chain.invoke(query)
    return answer

##### remove this function
def ollama_rag(model, fullragchat_temp, stop_words_list, fullragchat_embed_model, query):
    ### instanciate model
    ollama = Ollama(
        model=model, 
        temperature=float(fullragchat_temp), 
        stop=stop_words_list, 
        verbose=True,
    )
    all_splits = get_rag_text(query)
    oembed = OllamaEmbeddings(model=fullragchat_embed_model)
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=oembed)
    docs = vectorstore.similarity_search(query)
    vector_store_hits = len(docs)
    ### create chain w/ model
    qachain=RetrievalQA.from_chain_type(ollama, retriever=vectorstore.as_retriever())
    ### invoke chain
    results = qachain.invoke({"query": query})
    answer = results['result'] 
    return answer

##### remove this function
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
    stop_words_list = fullragchat_stop_words.split(', ')
    if stop_words_list == ['']: stop_words_list = None
    if model == "fake_llm":
        answer = fake_llm(query)
    elif ( (model == "orca-mini") or 
            (model == "phi") or 
            (model == "tinyllama") or
            (model == "llama2") or 
            (model == "llama2-uncensored") or 
            (model == "mistral") or 
            (model == "mixtral") or 
            (model == "command-r") ):
        if fullragchat_rag_source:
            if fullragchat_loop_context == 'True':
                answer = ollama_convo_rag(
                    model=model, 
                    fullragchat_temp=fullragchat_temp, 
                    stop_words_list=stop_words_list, 
                    fullragchat_embed_model=fullragchat_embed_model, 
                    query=query
                )
            else:
                answer = ollama_rag(
                    model=model, 
                    fullragchat_temp=fullragchat_temp, 
                    stop_words_list=stop_words_list, 
                    fullragchat_embed_model=fullragchat_embed_model, 
                    query=query )
        else:
            if fullragchat_loop_context == 'True':
                answer = ollama_convochat(
                    model=model, 
                    fullragchat_temp=fullragchat_temp, 
                    stop_words_list=stop_words_list, 
                    query=query, 
                )
            else:
                answer = ollama_qachat(
                    model=model, 
                    fullragchat_temp=fullragchat_temp, 
                    stop_words_list=stop_words_list, 
                    query=query, 
                )
    elif ( (model == "open-mixtral-8x7b") or 
            (model == "mistral-large-latest") or 
            (model == "open-mistral-7b") ):
        mkey = os.getenv('Mistral_API_key')
        if fullragchat_rag_source:
            if fullragchat_loop_context == 'True':
                if fullragchat_rag_source != 'Auto': # source specified in UI
                    if query == 'command: summarize': # override - just do direct summary of full doc
                        to_sum = get_rag_text(query)
                        answer = f'Summary of {fullragchat_rag_source}: \n'
                        answer += create_summary(
                            to_sum=to_sum, 
                            model=model, 
                            mkey=mkey, 
                            fullragchat_temp=fullragchat_temp)
                    else: # mistral_convo_rag - "older" single LLM pass
                        answer = mistral_convo_rag(
                            fullragchat_embed_model=fullragchat_embed_model, 
                            mkey=mkey, 
                            model=model, 
                            fullragchat_temp=fullragchat_temp, 
                            query=query )
                else: # source set to 'Auto' - "newer" double LLM pass
                    # figure out which staged rag doc to use
                    global rag_source_clues
                    rag_source_clues = rag_source_clue_value # doc helps llm choose rag file
                    selected_rag = choose_rag(
                        mkey=mkey, 
                        model=model, 
                        fullragchat_temp=fullragchat_temp, 
                        query=query )
                    logging.info(f'===> choose_rag returned: {selected_rag}')
                    # cleanup as LLM tends to be chatting and not listen to just the filename please...
                    # Should be json like {"filename": "return_filename.faiss"}
                    # note: drops any comments from LLM...
                    match = re.search(r'"filename":\s+"([^"]+)"', selected_rag)
                    clean_selected_rag = match.group(1)
                    ##### cleanup
                    # pattern = r'\*{5}(.+?)\*{5}' # Hope LLM put *****filename.txt***** in there somewhere...
                    # match = re.search(pattern, selected_rag)
                    # clean_selected_rag = match.group(1)
                    answer = f'Retrieved document "{clean_selected_rag}". \n'
                    clean_selected_rag = f'docs/{clean_selected_rag}'
                    logging.info(f'===> clean_selected_rag: {clean_selected_rag}')
                    fullragchat_rag_source = clean_selected_rag
                    answer += mistral_convo_rag(
                        fullragchat_embed_model=fullragchat_embed_model, 
                        mkey=mkey, 
                        model=model, 
                        fullragchat_temp=fullragchat_temp, 
                        query=query )
                    # now set global back to 'Auto' for UI and next round
                    fullragchat_rag_source = 'Auto'
            else:
                answer = mistral_rag(
                    fullragchat_embed_model=fullragchat_embed_model, 
                    mkey=mkey, 
                    model=model, 
                    fullragchat_temp=fullragchat_temp, 
                    query=query )
        else:
            if fullragchat_loop_context == 'True':
                answer = mistral_convochat(
                    mkey=mkey, 
                    query=query, 
                    model=model, 
                    fullragchat_temp=fullragchat_temp )
            else:
                answer = mistral_qachat(
                    mkey=mkey, 
                    query=query, 
                    model=model, 
                    fullragchat_temp=fullragchat_temp )
    else:
        answer = "No LLM named " + model
    return answer

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
    global fullragchat_loop_context
    ### set initial values
    fullragchat_history = []
    fullragchat_model = "open-mixtral-8x7b"
    fullragchat_temp = "0.25"
    fullragchat_rag_source = "Auto"
    fullragchat_embed_model = "mistral-embed"
    fullragchat_loop_context = "True"
    fullragchat_stop_words = ""
    fullragchat_skin = "not yet implemented"
    fullragchat_music = "not yet implemented"
    init_fullragchat_history()
    return render_template('fullragchat.html', 
        fullragchat_history=fullragchat_history,
        fullragchat_model=fullragchat_model,
        fullragchat_temp=fullragchat_temp,
        fullragchat_stop_words=fullragchat_stop_words, 
        fullragchat_rag_source=fullragchat_rag_source, 
        fullragchat_embed_model=fullragchat_embed_model,
        fullragchat_skin=fullragchat_skin,
        fullragchat_music=fullragchat_music,
        fullragchat_loop_context=fullragchat_loop_context, 
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
    global fullragchat_loop_context
    query = request.form['query']
    fullragchat_model = request.form['model']
    fullragchat_temp = request.form['temp']
    fullragchat_stop_words = request.form['stop_words']
    fullragchat_rag_source = request.form['rag_source']
    fullragchat_embed_model = request.form['embed_model']
    fullragchat_skin = request.form['skin']
    fullragchat_music = request.form['music']
    fullragchat_loop_context = request.form['loop_context']
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
        fullragchat_loop_context=fullragchat_loop_context, 
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
    global fullragchat_loop_context
    global memory
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
    memory.save_context({"input": query}, {"output": answer}) # ConversationBufferWindowMemory save of query and answer
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
        fullragchat_loop_context=fullragchat_loop_context, 
        )

if __name__ == "__main__":
    app.run()
