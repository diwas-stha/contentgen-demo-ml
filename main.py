import os
from typing import List
from langchain_community.document_loaders import AsyncHtmlLoader, DirectoryLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_openai import ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from prompt_templates import content_generator_template, topic_generator_template, context_summarizer_template
from langchain.callbacks.tracers import ConsoleCallbackHandler

import logging
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(filename='app.log', level=logging.INFO)

load_dotenv()

app = FastAPI()

class GenerateTopicsRequest(BaseModel):
    url: str

class GenerateContentRequest(BaseModel):
    topic: str


llm = ChatOpenAI(model="gpt-4")

def parse(text: str) -> List[str]:
        """Parse the output of an LLM call."""
        return text.strip().split(", ")

context_summarizer_prompt = PromptTemplate(
    template=context_summarizer_template,
    input_variables=["context"],
    # partial_variables={"format_instructions": format_instructions},
)
context_summarizer_chain = context_summarizer_prompt | llm

# Define topic and content generators
topic_generator_prompt = PromptTemplate(
    input_variables=["context"],
    template=topic_generator_template
)
topic_generator_chain = topic_generator_prompt | llm

content_generator_prompt = PromptTemplate(
    input_variables=["topic"],
    template=content_generator_template
)
content_generator_chain = content_generator_prompt | llm


@app.post("/generateTopicsFromURL")
async def generate_topics(request: GenerateTopicsRequest) -> List[str]:
    try:
        url = request.url
        if url:
            loader = AsyncHtmlLoader(url)
            docs = loader.load()
            
            # Logic to transform HTML to text
            html2text = Html2TextTransformer()
            docs_transformed = html2text.transform_documents(docs)
            
            # Logic to split text into chunks
            text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=2000, chunk_overlap=500
            )
            chunks = text_splitter.split_documents(docs_transformed)
            
            # Logic to generate combined summary
            combined_summary = ""
            for index, chunk in enumerate(chunks):
                summary = context_summarizer_chain.invoke({'context': chunk.page_content})
                combined_summary += summary.content + "\n"
            
            # Logic to generate topics
            topics = topic_generator_chain.invoke({
                'context': combined_summary
            }, config={'callbacks': [ConsoleCallbackHandler()]})
            
            # Logic to parse generated topics into a list
            topics_list = parse(text=topics.content)
            
            return topics_list
        else:
            raise HTTPException(status_code=400, detail="URL is required.")

    except Exception as e:
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
        logging.info(f"File saved: {file.filename}")
        print(f"File saved: {file.filename}")


        loader = DirectoryLoader(directory)
        docs = loader.load()
        logging.info(f"file loaded with directory loader")
        print(f"file loaded with directory loader")


        # Logic to split text into chunks
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=2000, chunk_overlap=500
        )
        
        chunks = text_splitter.split_documents(docs)
        print(chunks)
        
        # Logic to generate combined summary
        combined_summary = ""
        for index, chunk in enumerate(chunks):
            summary = context_summarizer_chain.invoke({'context': chunk.page_content})
            combined_summary += summary.content + "\n"

        print(combined_summary)
        
        # Logic to generate topics
        topics = topic_generator_chain.invoke({
            'context': combined_summary
        }, config={'callbacks': [ConsoleCallbackHandler()]})
        
        # Logic to parse generated topics into a list
        topics_list = parse(text=topics.content)
        
        return topics_list

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/generateContent")
async def generate_content(request: GenerateContentRequest) -> str:
    try:
        topic = request.topic
        if topic:
            lesson = content_generator_chain.invoke({'topic': topic}, config={'callbacks': [ConsoleCallbackHandler()]})
            return lesson.content
        else:
            raise HTTPException(status_code=400, detail="Topic is required.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))