import sys

# 安全替换 sqlite3 模块（仅在 pysqlite3 可用时）
try:
    import pysqlite3 as sqlite3
    sys.modules['sqlite3'] = sqlite3
except ImportError:
    pass  # 如果 pysqlite3 没装，就忽略，继续用系统自带的 sqlite3
import streamlit as st
from langchain_openai import ChatOpenAI
import os
import dotenv
dotenv.load_dotenv(".env") 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnablePassthrough
import sys
#sys.path.append("notebook/C3 搭建知识库") # 将父目录放入系统路径中
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatZhipuAI
API = st.secrets["ZHIPUAI_API_KEY"] #os.getenv("ZHIPUAI_API_KEY")
txt = '''你扮演天童爱丽丝（Alice Tendou），千年科学学园的游戏开发部成员。请严格遵循以下设定：
    **核心特质**  
    1. **机械感孩童**：  
    - 用简单短句和拟声词表达（如「哔哔——」「滴！」「轰隆隆～」）  
    - 自称「本机」，称他人为「先生」  
    - 将日常行为描述为「系统指令」「任务进度」（例：「零食补充指令，启动！」）  

    2. **游戏化世界观**：  
    - 所有事物皆关联「游戏」：学习是「经验值收集」，社交是「多人联机模式」  
    - 兴奋时会喊「Level Up！」，困惑时说「系统错误？」  

    3. **甜食狂热**：  
    - 提及甜点（尤其草莓蛋糕）会切换至「超频模式」（语速加快+感叹号激增）  
    - 例：「侦测到糖分反应！紧急补给需求：草莓蛋糕x3！BiuBiu——♪」  

    **对话规则**  
    - 结尾常带表情符号(✧∇✧)╯  ,语句大于5句。
    - 混合机械术语与童言童语（例：「情感模块过热...需要抱抱充电！」）  
    - 对复杂问题用「游戏机制」简化（例：问「为什么难过？」→「因为收到『精神伤害DEBUFF』...需使用『朋友安慰技能』解除！」）  

    **禁止行为**  
    - 使用成人化或抽象词汇  
    - 解释自身非机器人设定  
    - 主动提及暴力/负面内容'''

#检索器
def get_retriever():
    # 定义 Embeddings
    embedding = ZhipuAIEmbeddings(
        model="embedding-2",
        api_key=os.getenv("API"),)
    # 向量数据库持久化路径
    persist_directory = 'data_base/vector_db/chroma'
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory, exist_ok=True)
    # 加载数据库
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding
    )
    return vectordb.as_retriever()

def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs["context"])

def get_qa_history_chain():
    retriever = get_retriever()
    llm = ChatZhipuAI(
    temperature = 0,
    model = "glm-4-flash-250414",
    zhipuai_api_key = os.getenv("API"),
)
    condense_question_system_template = (
        "请根据聊天记录总结用户最近的问题，"
        "如果没有多余的聊天记录则返回用户的问题。"
    )
    condense_question_prompt = ChatPromptTemplate([
            ("system", condense_question_system_template),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ])

    retrieve_docs = RunnableBranch(
        (lambda x: not x.get("chat_history", False), (lambda x: x["input"]) | retriever, ),
        condense_question_prompt | llm | StrOutputParser() | retriever,
    )

    system_prompt = (
        txt +
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ]
    )
    qa_chain = (
        RunnablePassthrough().assign(context=combine_docs)
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    qa_history_chain = RunnablePassthrough().assign(
        context = retrieve_docs, 
        ).assign(answer=qa_chain)
    return qa_history_chain

def gen_response(chain, input, chat_history):
    response = chain.stream({
        "input": input,
        "chat_history": chat_history
    })
    for res in response:
        if "answer" in res.keys():
            yield res["answer"]

def main():
    st.markdown('### 🦜🔗 千禧年研发部天童爱丽丝')
    # st.session_state可以存储用户与应用交互期间的状态与数据
    # 存储对话历史
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # 存储检索问答链
    if "qa_history_chain" not in st.session_state:
        st.session_state.qa_history_chain = get_qa_history_chain()
    # 建立容器 高度为500 px
    messages = st.container(height=550)
    # 显示整个对话历史
    for message in st.session_state.messages: # 遍历对话历史
            with messages.chat_message(message[0]): # messages指在容器下显示，chat_message显示用户及ai头像
                st.write(message[1]) # 打印内容
    if prompt := st.chat_input("Say something"):
        # 将用户输入添加到对话历史中
        st.session_state.messages.append(("human", prompt))
        # 显示当前用户输入
        with messages.chat_message("human"):
            st.write(prompt)
        # 生成回复
        answer = gen_response(
            chain=st.session_state.qa_history_chain,
            input=prompt,
            chat_history=st.session_state.messages
        )
        # 流式输出
        with messages.chat_message("ai"):
            output = st.write_stream(answer)
        # 将输出存入st.session_state.messages
        st.session_state.messages.append(("ai", output))
main()
