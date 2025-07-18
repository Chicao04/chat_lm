from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.llms import OpenAI
from langchain.utilities import WikipediaAPIWrapper

# Khởi tạo LLM (GPT-4 hoặc GPT-3.5)
llm = OpenAI(model_name="gpt-4", temperature=0)

# Tạo Tools (Wikipedia Search)
wiki_tool = Tool(
    name="WikipediaSearch",
    func=WikipediaAPIWrapper().run,
    description="Tìm kiếm thông tin trên Wikipedia theo yêu cầu người dùng."
)

# Khởi tạo Agent với Tool
agent = initialize_agent(
    tools=[wiki_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Chạy thử Agent
response = agent.run("LangChain là gì?")
print(response)
