import os
import shutil
from typing import List
from langchain_community.document_loaders import AsyncHtmlLoader, PyPDFDirectoryLoader, PyPDFLoader, UnstructuredFileLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from src.prompt_templates import content_generator_template, topic_generator_template, context_summarizer_template, topic_generator_template2
from langchain.callbacks.tracers import ConsoleCallbackHandler
# from langchain.chains.summarize import load_summarize_chain
# from map_reduce import map_reduce_chain
from src.summarization import brv
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from src.chains import topic_generator_chain, content_generator_chain
import logging
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(filename='app.log', level=logging.INFO)

load_dotenv()

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]

)

class GenerateTopicsRequest(BaseModel):
    url: str

class GenerateContentRequest(BaseModel):
    topic: str


llm = ChatOpenAI(model="gpt-4")
embeddings = OpenAIEmbeddings()

def parse(text: str) -> List[str]:
        """Parse the output of an LLM call."""
        return text.strip().split(", ")

# context_summarizer_prompt = PromptTemplate(
#     template=context_summarizer_template,
#     input_variables=["context"],
#     # partial_variables={"format_instructions": format_instructions},
# )
# context_summarizer_chain = context_summarizer_prompt | llm

# # Define topic and content generators
# topic_generator_prompt = PromptTemplate(
#     input_variables=["context"],
#     # template=topic_generator_template
#     template=topic_generator_template2
# )
# topic_generator_chain = topic_generator_prompt | llm

# # content_generator_prompt = PromptTemplate(
# #     input_variables=["topic", "context"],
# #     template=content_generator_template
# # )
# new_db = FAISS.load_local("faiss_index", OpenAIEmbeddings())
# faiss_retriever = new_db.as_retriever()
# content_generator_prompt = ChatPromptTemplate.from_template(template=content_generator_template)
# content_generator_from_docs = (
#     RunnablePassthrough.assign(context=(lambda x: (x["context"])))
#     | content_generator_prompt
#     | llm
# )

# content_generator_with_source = RunnableParallel(
#     {"context": faiss_retriever, "topic": RunnablePassthrough()}
# ).assign(answer=content_generator_from_docs)

# content_generator_chain = content_generator_prompt | llm

# Logic to split text into chunks
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=2500, chunk_overlap=500
)

# summary_chain = load_summarize_chain(llm=llm, chain_type='map_reduce',
#                                      verbose=True # Set verbose=True if you want to see the prompts being used
#                                     )


@app.post("/generateTopicsFromURL")
async def generate_topics(request: GenerateTopicsRequest) -> List[str]:
    try:
        logging.info(f"Received request body: {request}")
        print(request)
        url = request.url
        if url:
            loader = AsyncHtmlLoader(url)
            docs = loader.load()
            
            # Logic to transform HTML to text
            html2text = Html2TextTransformer()
            docs_transformed = html2text.transform_documents(docs)

            text = ""

            for page in docs_transformed:
                text += page.page_content
                
            text = text.replace('\t', ' ')
            
            if llm.get_num_tokens(text) > 4000:
                combined_summary = brv(text=text)
            else:
                combined_summary = docs_transformed

            # chunks = text_splitter.split_documents(docs_transformed)
            
            # Logic to generate combined summary
            # combined_summary = ""
            # for index, chunk in enumerate(chunks):
            #     summary = context_summarizer_chain.invoke({'context': chunk.page_content})
            #     combined_summary += summary.content + "\n"

            # combined_summary = map_reduce_chain.run(chunks)
            
            # Logic to generate topics
            logging.info("Generating the topics")
            print("Generating topics")
            topics = topic_generator_chain.invoke({
                'context': combined_summary}, 
                # config={'callbacks': [ConsoleCallbackHandler()]}
            )
            
            # Logic to parse generated topics into a list
            topics_list = parse(text=topics.content)
            
            logging.info("Topic generation completed")
            print(f"Topic generation completed: \n{topics_list}" )
            return topics_list
        else:
            raise HTTPException(status_code=400, detail="URL is required.")

    except Exception as e:
        logging.info(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/generateTopicsFromFile")
async def generate_topics_from_file(file: UploadFile = File(...)) -> List[str]:
    directory = "files"
    if not os.path.exists(directory):
        os.makedirs(directory)
    try:
        logging.info(f"Received file: {file.filename}")
        print(f"Received file: {file.filename}")


        file_path = os.path.join(directory, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        logging.info(f"File saved on {file_path}")
        print(f"File saved: {file_path}")


        loader = PyPDFLoader(file_path)
        docs = loader.load()
        logging.info(f"file loaded with directory loader")
        print(f"file loaded with directory loader")

        text = ""

        for page in docs:
            text += page.page_content
            
        text = text.replace('\t', ' ')
        
        if llm.get_num_tokens(text) > 4000:
            combined_summary = brv(text=text)
        else:
            combined_summary = docs
            faiss = FAISS.from_documents(docs, embeddings)
            faiss.save_local("faiss_index")
        # combined_summary = brv(pages=docs)
        
        # chunks = text_splitter.split_documents(docs)
        # logging.info(f"Chunking of documents completed")
        # print(f"Chunking of documents completed with {len(chunks)} chunks.")
        
        # Logic to generate combined summary
        # combined_summary = ""
        # for index, chunk in enumerate(chunks):
        #     summary = context_summarizer_chain.invoke({'context': chunk.page_content})
        #     combined_summary += summary.content + "\n"
        # combined_summary = map_reduce_chain.run(chunks)


        logging.info("Generated Summaries combined")
        print(combined_summary)
        
        # Logic to generate topics
        logging.info("Generating the topics")
        print("Generating topics")
        topics = topic_generator_chain.invoke({
            'context': combined_summary}, 
            # config={'callbacks': [ConsoleCallbackHandler()]}
        )
        
        # Logic to parse generated topics into a list
        topics_list = parse(text=topics.content)
        shutil.rmtree(directory)
        
        logging.info("Topic generation completed")
        print(f"Topic generation completed: \n{topics_list}" )
        return topics_list

    except Exception as e:
        logging.info(f"Error {e}")
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    



@app.post("/generateContent")
async def generate_content(request: GenerateContentRequest) -> str:
    try:
        topic = request.topic
        if topic:
            logging.info(f"Generating the lesson for {topic}")
            print(f"Generating the lesson for {topic}")

            lesson = content_generator_chain(topic=topic)
            logging.info(f"Generation of lesson completed")
            print(f"Generation of lesson completed")
            return lesson['answer'].content
        else:
            raise HTTPException(status_code=400, detail="Topic is required.")

    except Exception as e:
        logging.info(f"Error {e}")
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))