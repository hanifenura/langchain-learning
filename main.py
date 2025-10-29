from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

def main() -> None:

    prompt_template = PromptTemplate(
        input_variables = ["input"],
        template="Translate the following English text to French: {input}",
    )

    llm_ollama = ChatOllama(model="deepseek-r1:1.5b",temperature=0.7)    
    
    output_parser_str = StrOutputParser()

    chain_with_str_output = prompt_template | llm_ollama | output_parser_str

    response= chain_with_str_output.invoke(
        {"input":"Hello, how are you?"}
    )
    
    print("Response with string output:",response)


if __name__ == "__main__":
    main()
