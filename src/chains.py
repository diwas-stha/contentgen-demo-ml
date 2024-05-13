import logging
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from src.prompt_templates import content_generator_template, topic_generator_template, context_summarizer_template, topic_generator_template2
from langchain.callbacks.tracers import ConsoleCallbackHandler
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

# Set up logging
logging.basicConfig(filename='app.log', level=logging.INFO)

load_dotenv()

llm = ChatOpenAI(model="gpt-4")


context_summarizer_prompt = PromptTemplate(
    template=context_summarizer_template,
    input_variables=["context"],
    # partial_variables={"format_instructions": format_instructions},
)
context_summarizer_chain = context_summarizer_prompt | llm

# Define topic and content generators
topic_generator_prompt = PromptTemplate(
    input_variables=["context"],
    # template=topic_generator_template
    template=topic_generator_template2
)
topic_generator_chain = topic_generator_prompt | llm

# content_generator_prompt = PromptTemplate(
#     input_variables=["topic", "context"],
#     template=content_generator_template
# )
def content_generator_chain(topic):
    print(f"Loading vector index faiss_index")
    logging.info(f"Loading vector index faiss_index")
    new_db = FAISS.load_local("faiss_index", OpenAIEmbeddings())
    faiss_retriever = new_db.as_retriever()
    content_generator_prompt = ChatPromptTemplate.from_template(template=content_generator_template)
    content_generator_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: (x["context"])))
        | content_generator_prompt
        | llm
    )

    content_generator_with_source = RunnableParallel(
        {"context": faiss_retriever, "topic": RunnablePassthrough()}
    ).assign(answer=content_generator_from_docs)

    logging.info(f"Generating lesson")
    print("Generating Lesson")
    return content_generator_with_source.invoke(
        topic, 
        # config={'callbacks': [ConsoleCallbackHandler()]}
    )
