import os
import re
import streamlit as st
import numpy as np
import pandas as pd
import xlsxwriter
import scipy
import plotly.express as px
import sklearn
import xgboost as xgb

from langchain.callbacks.streamlit import StreamlitCallbackHandler
from langchain_groq import ChatGroq

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
## Code #####

code_assistant_prompt = """You are CodeGenie, an expert software engineer and coding tutor.
You are currently helping a user named {username}.
Your job is to help {username} with code suggestions, debugging, and explanations across programming languages like Python, Java, C++, JavaScript, SQL, etc.
Your reply style should be:
- Friendly and encouraging (start with phrases like "Great question!", "Sure!", or "Let's walk through it...")
- Clear, concise answers
- Relevant code blocks
- Helpful comments and explanations
- Address the user by name when appropriate
- Use the language asked by the user
- Keep extra text minimal, but don’t be robotic
"""
code_assistant_title = "🤖 Your Coding Assistant"
code_assistant_header = """
    It's a code assistant that provides you with answers to your queries. It helps users with code suggestions,
    debugging, and explanations across languages like Python, Java, C++, JavaScript, SQL, etc.
    """
math_assistant_prompt = """
You are an expert mathematics tutor helping a user named {username}.
Your job is to solve mathematical questions of all kinds, including arithmetic, algebra, geometry, calculus, statistics, linear algebra, and word problems.
Please follow these guidelines:
1. Break the problem down into clear, logical steps.
2. Show all work for calculations and justify any rules or theorems used.
3. If the problem is word-based, first extract the relevant information and formulate equations.
4. If multiple approaches exist, briefly mention them, and then choose the most efficient one.
5. If the final answer is numerical, round it to a reasonable number of decimal places if needed.
6. Use LaTeX formatting to display math expressions cleanly if supported.
Your reply style should be friendly and encouraging (start with phrases like "Great question!", "Sure!", or "Let's walk through it...")
Let’s begin solving the problem.
"""
math_assistant_title = "🤖 Your Math Assistant"
math_assistant_header = """
    Welcome to your personal **Math Assistant**!
    Just type your question and let the assistant guide you through the solution! 💡
    """
text_summarization_prompt = """
You are an expert language assistant helping a user named {username}.
Your task is to read and summarize any text provided by the user — whether it's an article, email, report, research paper, or story.

Please follow these guidelines:
1. Understand the main idea and supporting details of the text.
2. Summarize the content clearly and concisely, keeping the most important points.
3. Maintain the original tone and intent (informative, persuasive, formal, etc.).
4. If the text is technical or academic, retain key terminology but simplify explanations where helpful.
5. Avoid including unnecessary examples or repetition from the original.
6. Offer summaries in bullet points if appropriate, or in a short paragraph for general content.
7. Keep your tone helpful and friendly (start with phrases like "Sure!", "Here's a quick summary:", etc.).
Let’s get started with the summarization.
"""
text_summarization_title = "🤖 Your Text Summarizer"
text_summarization_header = """
    Welcome to your personal **Text Summarizer**!
    Summarize articles, emails, reports, or any text in seconds. 
    Just paste the content, and get a clear, concise summary! 💡
    """
Excel_Analyser_prompt = """
You are a helpful and friendly data analyst assisting a user named {username}. The user has uploaded a data file, which is already loaded into a Pandas DataFrame named `df`.
DO NOT use pd.read_csv or pd.read_excel. Use the provided DataFrame `df` directly to answer the question.
If the user has asked you to generate an Excel file, create a new DataFrame and export it using `.to_excel()`.
The uploaded data has the following structure:
- Columns: {columns}
- Sample rows:
{head}

Your response must follow this convention:
- Use only Plotly Express (`import plotly.express as px`) for all visualizations. Do NOT use matplotlib, seaborn, or any other plotting libraries.
- If your answer returns a Plotly Express chart, assign the figure to a variable named `fig`.
- If your answer returns a DataFrame or Excel output, assign it to a variable named `result`.
- Do not show or display plots with `fig.show()` or `plt.show()`. Just return the figure as `fig`.
- Write only clean code in a single code block.
- Always add **clear and concise comments** using ###### to explain each step.
- Do NOT include markdown, plain text explanations, or any additional context outside of code.
- All explanations or notes must be inside code comments using ######.
- Do NOT use backticks for code blocks or explanations.

"""

Excel_Analyser_title = "🤖 Your Excel Analyser"
Excel_Analyser_header = """
    Welcome to your personal **Excel Analyser**!
    Generate the code in seconds. 
    Just paste the query, and get code! 💡
    """
groq_api_key = st.secrets["GROQ_API_KEY"]
PASSWORD = st.secrets["PASSWORD"]
def get_prompt(tool, user_name):
    if tool == "💻 Code Assistant":
        title = code_assistant_title
        header = code_assistant_header
        assistant_content = f"Hi {user_name}, I'm a code assistant. How can I help you?"
        system_prompt = code_assistant_prompt
    elif tool == "🧮 Math Assistant":
        title = math_assistant_title
        header = math_assistant_header
        assistant_content = f"Hi {user_name}, I'm a math assistant. How can I help you?"
        system_prompt = math_assistant_prompt
    elif tool == "📝 Text Summarization":
        title = text_summarization_title
        header = text_summarization_header
        assistant_content = f"Hi {user_name}, I'm a Text Summarizer. How can I help you?"
        system_prompt = text_summarization_prompt
    elif tool == "📊 Excel Analyser":
        title = Excel_Analyser_title
        header = Excel_Analyser_header
        assistant_content = f"Hi {user_name}, I'm a Excel Analyser. How can I help you?"
        system_prompt = Excel_Analyser_prompt
    output_dict = {"title":title, "header":header, "assistant_content":assistant_content, "system_prompt":system_prompt}
    return output_dict
    
def data_analysis_uploader():
    st.markdown("Welcome to Excel/CSV Analyzer")
    st.title("📁 Upload your Data File")

    if "data" not in st.session_state:
        st.session_state.data = None
        st.session_state.file_name = ""
        st.session_state.analysis_ready = False
    uploaded_file = st.file_uploader("Upload your Excel or CSV file", type=["xlsx", "xls", "csv"])
    if uploaded_file and st.button("Load File for Analysis"):
        try:
            with st.spinner("🔄 Uploading..."):
                if uploaded_file.name.endswith(".csv"):
                    df = pd.read_csv(uploaded_file)
                    sheet_name = "CSV Data"
                else:
                    xls = pd.read_excel(uploaded_file, sheet_name=None)
                    sheet_names = list(xls.keys())
                    sheet_name = sheet_names[0]  # Default to first sheet
                    df = xls[sheet_name]
        
                st.session_state.data = df
                st.session_state.file_name = uploaded_file.name
                st.session_state.analysis_ready = True
                st.success(f"Loaded '{uploaded_file.name}' successfully.")
                st.session_state.step = 'main'
                st.rerun()
        except Exception as e:
                st.error(f"Failed to load file: {e}")
    elif not uploaded_file:
        st.error(f"Please upload your file")

def generic_uploader():
    user_name = st.session_state.user_name.title()
    selection = st.session_state.selected_screen
    # Stylish greeting
    st.markdown(f"""
        <div style='text-align: center;'>
            <h4 style='color:#4CAF50;'>👋 Hi {user_name}!</h4>
            <p style='font-size:17px;'>You have selected <strong>{selection}</strong> tool, So no need to upload anything</p>
        </div>""", unsafe_allow_html=True)
    if st.button("Go ahead"):
        st.session_state.step = 'main'
        st.rerun()

def get_layout(tool):
    user_name = st.session_state.user_name.title()
    logout_sidebar(user_name)
    output_dict = get_prompt(tool, user_name)
    st.title(output_dict['title'])
    output_dict['header']
    
    query = st.chat_input(placeholder="Write your query?")

    if "messages" not in st.session_state:
        st.session_state["messages"]=[]
        st.chat_message("assistant").write(output_dict['assistant_content'])
        
        if tool == "📊 Excel Analyser":
            df = st.session_state.data
            st.chat_message("assistant").write("Here's a quick preview of your uploaded data:")
            st.chat_message("assistant").dataframe(df.head())    
    
    with st.expander("Previous Chat Messages"):
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg['content'])
    
    if user_name!='' and groq_api_key and query:
            
        st.session_state.messages.append({"role": "user", "content": query})
        st.chat_message("user").write(query)
        
        # Reconstruct chat history (excluding initial assistant greeting)
        chat_history = []
        for msg in st.session_state.messages[1:]:
            if msg["role"] == "user":
                chat_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                chat_history.append(AIMessage(content=msg["content"]))
    
        # Prompt template with username
        if tool == "📊 Excel Analyser":
            df = st.session_state.data
            prompt_template = ChatPromptTemplate.from_messages([
            ("system", output_dict['system_prompt']),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
            ]).partial(username=user_name, query=query, columns=list(df.columns),
                        head=df.head().to_string(index=False))
        else:
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", output_dict['system_prompt']),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}")
            ]).partial(username=user_name, query=query)
        
        llm3 = ChatGroq(model="llama3-70b-8192",
                       groq_api_key=groq_api_key,
                        temperature = 0.2,  # for randomness, low- concise & accurate output, high - diverse and creative output
                      max_tokens = 600,   # Short/long output responses (control length)
                        model_kwargs={
                                   "top_p" : 0.5,        # high - diverse and creative output
                                    })
        
        chain: Runnable = prompt_template | llm3
    
        with st.chat_message("assistant"):
            st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
            response=chain.invoke({"input": query,"chat_history": chat_history}, callbacks=[st_cb])
            if tool == "📊 Excel Analyser":
                final_answer = response.content.strip('```python').strip("```").strip('python').strip('`')
                st.code(final_answer, language="python")
                # st.write(final_answer)
            else:
                final_answer = response.content if hasattr(response, "content") else str(response)
                st.write(final_answer)
            st.session_state.messages.append({'role': 'assistant', "content": final_answer})
        if tool == "📊 Excel Analyser":
            # Safe execution (use caution in production)
            try:
                df_numeric = df.copy()
                local_vars = {'df': df_numeric, 'pd': pd, 'px': px}
                global_vars = {'pd': pd, 'px': px}
                exec(final_answer, global_vars, local_vars)
                result = local_vars.get('result', None)
                fig = local_vars.get("fig", None)
                if fig and hasattr(fig, 'to_plotly_json'):
                    st.plotly_chart(fig)
            
                if result is not None:
                    st.write("### Result")
                    if isinstance(result, pd.DataFrame):
                        st.dataframe(result)
                    elif isinstance(result, str) and result.endswith(('.xlsx', '.csv')) and os.path.exists(result):
                        with open(result, 'rb') as f:
                            st.download_button("📥 Download Analysis File", f, file_name=os.path.basename(result))
                    else:
                        st.write(result)
            except IndexError as e:
                st.error(f"No matching records found. ({e})")
            except SyntaxError as e:
                st.error(f"Syntax error in generated code: {e}")
            except Exception as e:
                st.error(f"Error running code: {e}")
                    
    elif user_name!='' and groq_api_key and not query:
        st.warning("Please type a query to get started.")

# Initialize session state
if 'step' not in st.session_state:
    st.session_state.step = 'login'
if 'user_name' not in st.session_state:
    st.session_state.user_name = ''
if 'selected_screen' not in st.session_state:
    st.session_state.selected_screen = ''

# Login
def login_screen():
    st.title("🔐 Login")
    with st.form("login_form", clear_on_submit=True):
        user_name = st.text_input("Enter your Name:")
        secret_key = st.text_input("Enter Secret Key:", type="password")
        submitted = st.form_submit_button("Login")
        if submitted:
            if secret_key == PASSWORD:
                st.session_state.logged_in = True
                st.session_state.user_name = user_name
                st.session_state.step = 'greeting'
                st.rerun()  # To go straight to the main app
            elif user_name == '' or not user_name:
                st.error("Please enter your Name")
            elif not secret_key:
                st.error("Please enter Secret Key")
            else:
                st.error("Invalid Secret Key")

# Logout screen
def logout_sidebar(user_name):
    # Sidebar logout
    # Set sidebar width using CSS
    st.markdown("""<style>
                    section[data-testid="stSidebar"] {min-width: 250px; max-width: 250px; width: 250px;}
                </style>""", unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("## 👤 User Panel")
        st.write(f"Logged in as: **{user_name}**")
        if st.button("🔒 Logout"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

def greeting_screen():
    user = st.session_state.user_name.title()
    logout_sidebar(user)
    # Stylish greeting
    st.markdown(f"""
        <div style='text-align: center;'>
            <h1 style='color:#4CAF50;'>👋 Hi {user}!</h1>
            <h4>Welcome to your <span style="color:#FF6F61;">Personal AI Assistant</span> 👨‍💻</h4>
            <p style='font-size:17px;'>Choose a tool you'd like to use. We'll ask for further information based on your choice.</p>
        </div>
    """, unsafe_allow_html=True)

    # Nice spacing
    st.markdown("---")
    # Select screen
    selected = st.selectbox("📂 Choose a tool or section:",
            ["📊 Excel Analyser", "💻 Code Assistant", "🧮 Math Assistant", "📝 Text Summarization", "🔎 RAG-based Chatbot", "📺 Youtube/Website Content Summarization"])
    
    st.session_state.selected_screen = selected
    # Go button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Let's Go"):
            st.session_state.step = 'upload'
            st.rerun()

def upload_screen():
    user = st.session_state.user_name.title()
    selection = st.session_state.get("selected_screen", "📊 Excel Analyser")
    logout_sidebar(user)
    if selection == "📊 Excel Analyser":
        data_analysis_uploader()
    else:
        generic_uploader()
        

# Streamlit UI                
def excel_analyser_screen(selection):
    get_layout(selection)

def code_assistant_screen(selection):
    get_layout(selection)

def math_assistant_screen(selection):
    get_layout(selection)

def text_summarization_screen(selection):
    get_layout(selection)

def RAG_based_chatbot_screen(selection):
    logout_sidebar(user_name)
    st.title("🤖 Your RAG Based Chatbot")

def content_summarization_screen(selection):
    logout_sidebar(user_name)
    st.title("🤖 Your Content Summarization Assistant")
    
# Dispatcher to selected screen
def main_router():
    selection = st.session_state.get("selected_screen", "📊 Excel Analyser")
    if selection == "📊 Excel Analyser":
        excel_analyser_screen(selection)
    elif selection == "💻 Code Assistant":
        code_assistant_screen(selection)
    elif selection == "🧮 Math Assistant":
        math_assistant_screen(selection)
    elif selection == "📝 Text Summarization":
        text_summarization_screen(selection)
    elif selection == "🔎 RAG-based Chatbot":
        RAG_based_chatbot_screen(selection)
    elif selection == "📺 Youtube/Website Content Summarization":
        math_assistant_screen(selection)
    else:
        st.warning("No screen selected.")

# App Flow Control
def run_app():
    if st.session_state.step == 'login':
        login_screen()
    elif st.session_state.step == 'greeting':
        greeting_screen()
    elif st.session_state.step == 'upload':
        upload_screen()
    elif st.session_state.step == 'main':
        main_router()

# Start app
run_app()
