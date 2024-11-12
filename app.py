import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun,DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler ## allow you to cominicate with all this tools 

import os 
from dotenv import load_dotenv
load_dotenv()
st.secrets["GROQ_API_KEY"]


## Arxiv, wikipedia and DuckDuckGoSearch Tools
arxiv_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=1000)
arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=1000)
wiki=WikipediaQueryRun(api_wrapper=wiki_wrapper)

search=DuckDuckGoSearchRun(name="Search")

## Stting my streamlit web app 

st.title("LangChain- Chat with search")
"""
In this example, we are using `StreamlitCallbackHandler` to display the thoughts and actions of an agetn in an interactive Streamlit app.
Try mode Langchain Streamlit Agent examples at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
"""

## Sidebar for settings
st.sidebar.title("Settings")
api_key=st.sidebar.text_input("Enter your GROQ api key:", type="password")

if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {
            "role":"assitant", 
            "content":"Hi, I am a chatbot who can search the web, How Can I help you?",
         }
    ]
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

if prompt:=st.chat_input(placeholder="What is machine learning?"):
    st.session_state.messages.append({"role":"user", "content":prompt})
    st.chat_message("user").write(prompt)

    llm=ChatGroq(groq_api_key=api_key, model_name="gemma2-9b-it", streaming=True)
    tools=[wiki, arxiv, search]

    search_agent=initialize_agent(
        tools=tools, 
        llm=llm, 
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
        handling_parsing_errors=True, ## if you faces any errors please make sure to parse those errors 
        )
    
    with st.chat_message("assistant"):
        st_cb=StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response=search_agent.run(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append({"role":"assistant", "content":response})
        st.write(response)




