import streamlit as st
from langchain.agents import ConversationalChatAgent, AgentExecutor
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_ollama import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.tools import DuckDuckGoSearchRun
import json
import requests
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title= 'Chat')

col1, col4 = st.columns([4, 1])
with col1:
    st.title('Chat para Agente de Conversação e Busca Semântica com LLM e LangChain')

msgs = StreamlitChatMessageHistory()

memory = ConversationBufferMemory(chat_memory= msgs,
                                  return_messages= True,
                                  memory_key= 'chat_history',
                                  output_key= 'output')

if len(msgs.messages) == 0:
    msgs.clear()
    msgs.add_ai_message('Como eu posso ajudar você?')
    st.session_state.steps = {}

avatars = {'human': 'user', 'ai': 'assistant'}

for idx, msg in enumerate(msgs.messages):
    with st.chat_message(avatars[msg.type]):
        for step in st.session_state.steps.get(str(idx), []):
            if step[0].tool == '_Exception':
                continue

            with st.expander(f'✅ **{step[0].tool}**: {step[0].tool_input}'):
                st.write(step[0].log)
                st.write(f'**{step[1]}**')
            
        st.write(msg.content)

if prompt := st.chat_input(placeholder = 'Digite ima pergunta paara começar!'):
    st.chat_message('user').write(prompt)
    llm = ChatOllama(model= 'llama3.1:8b', streaming= True)
    mecanismo_busca = [DuckDuckGoSearchRun(name = 'Search')]
    chat_agent = ConversationalChatAgent.from_llm_and_tools(llm = llm, tools = mecanismo_busca)
    executor = AgentExecutor.from_agent_and_tools(agent = chat_agent,
                                                  tools = mecanismo_busca,
                                                  memory = memory,
                                                  return_intermediate_steps = True,
                                                  handle_parsing_errors = True)
    
    with st.chat_message('assistant'):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts= False)
        response = executor(prompt, callbacks= [st_cb])
        st.write(response['output'])

        st.session_state.steps[str(len(msgs.messages) - 1)] = response['intermediate_steps']