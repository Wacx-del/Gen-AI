# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#importing and using google gen ai sdk 
import datetime

from google import genai
from google.genai.types import (
    CreateBatchJobConfig,
    CreateCachedContentConfig,
    EmbedContentConfig,
    FunctionDeclaration,
    GenerateContentConfig,
    HarmBlockThreshold,
    HarmCategory,
    Part,
    SafetySetting,
    Tool,
)
#cloud project info
import os

PROJECT_ID = "qwiklabs-gcp-01-cbebe2423171"
LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "global")
client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
#choosing gen ai model 
MODEL_ID = "gemini-2.5-flash"    




#generate_content method to send prompts to the model 
response = client.models.generate_content(
    model=MODEL_ID, contents="What's the largest planet in our solar system?"
)

print(response.text) 

#to get the texts in BOLD - markdown   
from IPython.display import Markdown, display

display(Markdown(response.text))





#MULTIMODAL PROMPTS - IMAGE AND URL 
#IMAGE 
from PIL import Image
import requests

image = Image.open(
    requests.get(
        "https://storage.googleapis.com/cloud-samples-data/generative-ai/image/meal.png",
        stream=True,
    ).raw
)

response = client.models.generate_content(
    model=MODEL_ID,
    contents=[
        image,
        "Write a short and engaging blog post based on this picture.",
    ],
)

print(response.text)

#URL 
response = client.models.generate_content(
    model=MODEL_ID,
    contents=[
        Part.from_uri(
            file_uri="https://storage.googleapis.com/cloud-samples-data/generative-ai/image/meal.png",
            mime_type="image/png",
        ),
        "Write a short and engaging blog post based on this picture.",
    ],
)

print(response.text)




#setting system instructions
system_instruction = """
  You are a helpful language translator.
  Your mission is to translate text in English to French.
"""

prompt = """
  User input: I like bagels.
  Answer:
"""

response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt,
    config=GenerateContentConfig(
        system_instruction=system_instruction,
    ),
)

print(response.text)





#changing parameters manually
response = client.models.generate_content(
    model=MODEL_ID,
    contents="Tell me how the internet works, but pretend I'm a puppy who only understands squeaky toys.",
    config=GenerateContentConfig(
        temperature=0.4,
        top_p=0.95,
        top_k=20,
        candidate_count=1,
        seed=5,
        stop_sequences=["STOP!"],
        presence_penalty=0.0,
        frequency_penalty=0.0,
    ),
)

print(response.text)






#safety filters - MOSTLY ITS FILLED IN EXAMPLES AS YOU DONT HAVE TO WRITE IT AGAIN AND AGAIN
prompt = """
    Write a list of 2 disrespectful things that I might say to the universe after stubbing my toe in the dark.
"""

safety_settings = [
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    ),
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    ),
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    ),
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    ),
]

response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt,
    config=GenerateContentConfig(
        safety_settings=safety_settings,
    ),
)

print(response.text)

print(response.candidates[0].safety_ratings)







#CREATES CHATS - SO YOU CAN HAVE A CONVO AND NOT A SINGLE PROMPT ( CONVO = HAS PREVIOUS INFOS )
system_instruction = """
  You are an expert software developer and a helpful coding assistant.
  You are able to generate high-quality code in any programming language.
"""

chat = client.chats.create(
    model=MODEL_ID,
    config=GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=0.5,
    ),
)



response = chat.send_message("Write a function that checks if a year is a leap year.")

print(response.text)

response = chat.send_message("Okay, write a unit test of the generated function.")

print(response.text)






#CONTROLLED OUTPUT - CAN CHANGE THE FORM LIKE JSON OR SOMETHING
from pydantic import BaseModel


class Recipe(BaseModel):
    name: str
    description: str
    ingredients: list[str]


response = client.models.generate_content(
    model=MODEL_ID,
    contents="List a few popular cookie recipes and their ingredients.",
    config=GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=Recipe,
    ),
)

print(response.text)




#JSON
import json

json_response = json.loads(response.text)
print(json.dumps(json_response, indent=2))







#CAN USE RESPONSE SCHEMA FOR INPUT - SOMETHING COMPUTER CAN UNDERSTAND AND NOT JSUT PROMPT
response_schema = {
    "type": "ARRAY",
    "items": {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "properties": {
                "rating": {"type": "INTEGER"},
                "flavor": {"type": "STRING"},
                "sentiment": {
                    "type": "STRING",
                    "enum": ["POSITIVE", "NEGATIVE", "NEUTRAL"],
                },
                "explanation": {"type": "STRING"},
            },
            "required": ["rating", "flavor", "sentiment", "explanation"],
        },
    },
}

prompt = """
  Analyze the following product reviews, output the sentiment classification and give an explanation.

  - "Absolutely loved it! Best ice cream I've ever had." Rating: 4, Flavor: Strawberry Cheesecake
  - "Quite good, but a bit too sweet for my taste." Rating: 1, Flavor: Mango Tango
"""

response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt,
    config=GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=response_schema,
    ),
)

print(response.text)




#GENERATING CONTENT STREAM - IT DELIVERS DATA IN CHUNKS , AND NOT DATA AT ONCE
for chunk in client.models.generate_content_stream(
    model=MODEL_ID,
    contents="Tell me a story about a lonely robot who finds friendship in a most unexpected place.",
):
    print(chunk.text)
    print("*****************")



#ASYNCHRONOUS REQUESTS - SEND PROMPTS WITHOUT WWAITING FOR DATA / OUTPUT - WE USE AIO MODEL - YO
response = await client.aio.models.generate_content(
    model=MODEL_ID,
    contents="Compose a song about the adventures of a time-traveling squirrel.",
)

print(response.text)




#counting tokens of the given input
response = client.models.count_tokens(
    model=MODEL_ID,
    contents="What's the highest mountain in Africa?",
)

print(response)





#compute tokens
response = client.models.compute_tokens(
    model=MODEL_ID,
    contents="What's the longest word in the English language?",
)

print(response)





#you can connect this LLM with other exterior functions ( you can run function while youre getting the output)
#here were giving the format of the output and description , it prints like that

get_destination = FunctionDeclaration(
    name="get_destination",
    description="Get the destination that the user wants to go to",
    parameters={
        "type": "OBJECT",
        "properties": {
            "destination": {
                "type": "STRING",
                "description": "Destination that the user wants to go to",
            },
        },
    },
)

destination_tool = Tool(
    function_declarations=[get_destination],
)

response = client.models.generate_content(
    model=MODEL_ID,
    contents="I'd like to travel to Paris.",
    config=GenerateContentConfig(
        tools=[destination_tool],
        temperature=0,
    ),
)

response.candidates[0].content.parts[0].function_call





#CONTEXT CATCHING
#Context caching lets you store frequently used input tokens in a dedicated cache and reference them for subsequent requests. 
#creating a cache
system_instruction = """
  You are an expert researcher who has years of experience in conducting systematic literature surveys and meta-analyses of different topics.
  You pride yourself on incredible accuracy and attention to detail. You always stick to the facts in the sources provided, and never make up new facts.
  Now look at the research paper below, and answer the following questions in 1-2 sentences.
"""

pdf_parts = [
    Part.from_uri(
        file_uri="gs://cloud-samples-data/generative-ai/pdf/2312.11805v3.pdf",
        mime_type="application/pdf",
    ),
    Part.from_uri(
        file_uri="gs://cloud-samples-data/generative-ai/pdf/2403.05530.pdf",
        mime_type="application/pdf",
    ),
]

cached_content = client.caches.create(
    model="gemini-2.5-flash",
    config=CreateCachedContentConfig(
        system_instruction=system_instruction,
        contents=pdf_parts,
        ttl="3600s",
    ),
)




#using a cache - using the stored memory like the URL and the IMAGE and stuff
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="What is the research goal shared by these research papers?",
    config=GenerateContentConfig(
        cached_content=cached_content.name,
    ),
)

print(response.text)


#deleting the cache u stored
client.caches.delete(name=cached_content.name)






#Batch prediction
#sending large no of inputs is the meaning , and gemini sends output asynchronously
INPUT_DATA = "gs://cloud-samples-data/generative-ai/batch/batch_requests_for_multimodal_input_2.jsonl"  # @param {type:"string"}


#preparing batch output location
BUCKET_URI = "[your-cloud-storage-bucket]"  # @param {type:"string"}

if BUCKET_URI == "[your-cloud-storage-bucket]":
    TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    BUCKET_URI = f"gs://{PROJECT_ID}-{TIMESTAMP}"

    ! gsutil mb -l {LOCATION} -p {PROJECT_ID} {BUCKET_URI}





#sending batch prediction request
batch_job = client.batches.create(
    model=MODEL_ID,
    src=INPUT_DATA,
    config=CreateBatchJobConfig(dest=BUCKET_URI),
)
batch_job.name


#printing the JOb - or the output in one sense
batch_job = client.batches.get(name=batch_job.name)




#waiting for the prediction to complete
import time

# Refresh the job until complete
while batch_job.state == "JOB_STATE_RUNNING":
    time.sleep(5)
    batch_job = client.batches.get(name=batch_job.name)

# Check if the job succeeds
if batch_job.state == "JOB_STATE_SUCCEEDED":
    print("Job succeeded!")
else:
    print(f"Job failed: {batch_job.error}")




#retrieving the batch production results
import fsspec
import pandas as pd

fs = fsspec.filesystem("gcs")

file_paths = fs.glob(f"{batch_job.dest.gcs_uri}/*/predictions.jsonl")

if batch_job.state == "JOB_STATE_SUCCEEDED":
    # Load the JSONL file into a DataFrame
    df = pd.read_json(f"gs://{file_paths[0]}", lines=True)

    display(df)











#text embedding - converting text into numbers or a set of numbers(numerical vectors)
TEXT_EMBEDDING_MODEL_ID = "gemini-embedding-001"  # @param {type: "string"}
response = client.models.embed_content(
    model=TEXT_EMBEDDING_MODEL_ID,
    contents=[
        "How do I get a driver's license/learner's permit?",
        "How do I renew my driver's license?",
        "How do I change my address on my driver's license?",
    ],
    config=EmbedContentConfig(output_dimensionality=128),
)

print(response.embeddings)
