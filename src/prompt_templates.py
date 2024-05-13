content_generator_template = """
Generate microlearning content on the given topic that can be completed within
ten minutes. The content should be based on the given context.
The content should cover key points and essential information comprehensively.
The title of the content should be the name of the topic. 

Additionally, create a set of quiz questions related to the microlearning content.
The quiz should consist of five questions, including multiple-choice questions,
true/false statements, and fill-in-the-blanks, designed to assess the learner's
understanding of the topic presented in the microlearning material.

Topic: {topic}
Context: {context}
You must write the lesson with markdown syntax.
Include the headings Title, Introduction, Lesson and Quiz as heading 1.

"""

topic_generator_template = """
From the given context, identify and list three relevant topics not 
covered in the context but which are 
essential to learn the teachings in the context.


context: {context}

Your response should be a list of comma separated values without quotes
eg: `topic 1, topic 2, topic 3` 

"""

context_summarizer_template = """
The given context is a section of a larger document.
Please generate a comprehensive summary outlining the key concepts, arguments, and evidence presented in the given text.
Your summary should be concise yet thorough, providing a clear understanding of the article's main points.

Text to summarize: {context}

"""

# Map
map_template = """The following is a set of documents
{docs}
Based on this list of docs, please identify the main themes 
Helpful Answer:"""

# Reduce
reduce_template = """The following is set of summaries:
{docs}
Take these and distill it into a final, consolidated summary of the main themes. 
Helpful Answer:"""


topic_generator_template2 = """
You are given a summary of a larger body of text. From the summary identify key 
topics of which microlessons can be created. The suggested topic should be
relevant to the given context. Generate six topics only.
Provide the identified topics in a list. 


context: {context}

Your response should be a list of comma separated values without quotes
eg: `topic 1, topic 2, topic 3` 

"""