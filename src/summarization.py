from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains.summarize import load_summarize_chain
import numpy as np
from sklearn.cluster import KMeans
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()


llm = ChatOpenAI(
    temperature=0,
    max_tokens=1000,
    model='gpt-4'
)

llm4 = ChatOpenAI(temperature=0,
                 max_tokens=3000,
                 model='gpt-4',
                 request_timeout=120
                )


map_prompt = """
You will be given a single section of a larger file. This section will be enclosed in triple backticks (```)
Your goal is to give a summary of this section so that a reader will have a full understanding of what's written.
Your response fully encompass what was said in the passage.

```{text}```
FULL SUMMARY:
"""
map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])

map_chain = load_summarize_chain(
    llm=llm,
    chain_type="stuff",
    prompt=map_prompt_template
)

combine_prompt = """
You will be given a series of summaries. The summaries will be enclosed in triple backticks (```)
Your goal is to give a verbose summary of what's written in the file.
The reader should be able to grasp what the article is about.

```{text}```
VERBOSE SUMMARY:
"""
combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])

reduce_chain = load_summarize_chain(
    llm=llm4,
    chain_type="stuff",
    prompt=combine_prompt_template,
)

def brv(text):
    # text = ""

    # for page in pages:
    #     text += page.page_content
        
    # text = text.replace('\t', ' ')
    embeddings = OpenAIEmbeddings()
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", "\t"], chunk_size=2500, chunk_overlap=500)

    docs = text_splitter.create_documents([text])
    vectors = embeddings.embed_documents([doc.page_content for doc in docs])
    faiss = FAISS.from_documents(docs, embeddings)
    faiss.save_local("faiss_index")

    num_clusters = 10

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(vectors)

    # Find the closest embeddings to the centroids

    # Create an empty list that will hold your closest points
    closest_indices = []

    # # Loop through the number of clusters you have
    # for i in range(num_clusters):
        
    #     # Get the list of distances from that particular cluster center
    #     distances = np.linalg.norm(vectors - kmeans.cluster_centers_[i], axis=1)
        
    #     # Find the list position of the closest one (using argmin to find the smallest distance)
    #     closest_index = np.argmin(distances)
        
    #     # Append that position to your closest indices list
    #     closest_indices.append(closest_index)
    with ThreadPoolExecutor() as executor:
        # Define a function to find closest indices for a given cluster
        def find_closest_indices(i):
            distances = np.linalg.norm(vectors - kmeans.cluster_centers_[i], axis=1)
            closest_index = np.argmin(distances)
            return closest_index

        # Use ThreadPoolExecutor to concurrently execute the function for each cluster
        closest_indices = list(executor.map(find_closest_indices, range(num_clusters)))

    
    selected_indices = sorted(closest_indices)

    selected_docs = [docs[doc] for doc in selected_indices]

    # Make an empty list to hold your summaries
    summary_list = []

    # # Loop through a range of the lenght of your selected docs
    # for i, doc in enumerate(selected_docs):
        
    #     # Go get a summary of the chunk
    #     chunk_summary = map_chain.run([doc])
        
    #     # Append that summary to your list
    #     summary_list.append(chunk_summary)
        
    #     # print(llm.get_num_tokens(chunk_summary))
    #     print (f"Summary #{i} (chunk #{selected_indices[i]}) - Preview: {chunk_summary[:250]} \n")
    # Define a function to process each document and append its summary to the list
    def process_document(doc, i):
        # Go get a summary of the chunk
        chunk_summary = map_chain.run([doc])
        print(llm.get_num_tokens(chunk_summary))
        print (f"Summary #{i} (chunk #{selected_indices[i]}) - Preview: {chunk_summary[:250]} \n")
        return chunk_summary

    # Use ThreadPoolExecutor for multithreading
    with ThreadPoolExecutor() as executor:
        # Submit tasks for processing documents concurrently
        futures = [executor.submit(process_document, doc, i) for i, doc in enumerate(selected_docs)]

        # Gather results in the same order as submitted
        for future in as_completed(futures):
            summary_list.append(future.result())

    
    summaries = "\n".join(summary_list)

    # Convert it back to a document
    summaries = Document(page_content=summaries)

    print (f"Your total summary has {llm.get_num_tokens(summaries.page_content)} tokens")

    output = reduce_chain.run([summaries])

    return output
