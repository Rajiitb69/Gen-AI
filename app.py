import streamlit as st
import numpy as np
import os
import re
from langchain.callbacks import StreamlitCallbackHandler
from langchain_groq import ChatGroq

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
## Code #####
user_histories = {}
system_prompt = f"""You are CodeGenie, an expert software engineer and coding tutor.
Your job is to help users with code suggestions, debugging, and explanations across programming languages like Python, Java, C++, JavaScript, SQL, etc.

You always reply with:
- Clear, concise answers
- Relevant code blocks
- Helpful comments
- Language as asked by the user
- No extra text unless necessary"""

# Prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])



st.title("🔎 LangChain - Code-Assistant")
"""
It's a code assistant that provides you with answers to your queries. 
It helps users with code suggestions, debugging, and explanations 
across programming languages like Python, Java, C++, JavaScript, SQL, etc.

"""

## Sidebar for settings
st.sidebar.title("Inputs")
user_name = st.sidebar.text_input("Enter your Name:")
groq_api_key = st.sidebar.text_input("Enter your Groq API Key:",type="password")
query = st.chat_input(placeholder="Write your query?")

if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role": "assistant", "content": "Hi, I'm a code assistant. How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

if user_name and groq_api_key and query:
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").write(query)
    
    llm3 = ChatGroq(model="llama-3.3-70b-versatile",
                   groq_api_key=groq_api_key,
                    temperature = 0.2,  # for randomness, low- concise & accurate output, high - diverse and creative output
                  max_tokens = 300,   # Short/long output responses (control length)
                    model_kwargs={
                               "top_p" : 0.5,        # high - diverse and creative output
                                })
    
    chain: Runnable = prompt_template | llm3
    
    if user_name not in user_histories:
        user_histories[user_name] = []
    history = user_histories[user_name]

    with st.chat_message("assistant"):
        st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
        response=chain.invoke({"input": st.session_state.messages,"chat_history": history},callbacks=[st_cb])
        st.session_state.messages.append({'role':'assistant',"content":response})
        st.write(response)
        
        # Store conversation
        history.append(HumanMessage(content=user_input))
        history.append(AIMessage(content=response.content if hasattr(response, "content") else response))

