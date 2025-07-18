from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

# Tạo mô hình chat
chat_model = ChatOllama(model="llama2")

# Khai báo schema
color_schema = ResponseSchema(
    name="items",
    description="A list of 5 colors"
)

# Tạo parser từ schema
parser = StructuredOutputParser.from_response_schemas([color_schema])

# Prompt template
template = "Generate a list of 5 {text}.\n{format_instructions}"
prompt = ChatPromptTemplate.from_template(template)
prompt = prompt.partial(format_instructions=parser.get_format_instructions())

# Tạo chain
chain = prompt | chat_model | parser

# Gọi mô hình
result = chain.invoke({"text": "colors"})

# In kết quả
print(result)
