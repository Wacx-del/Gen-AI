#Install Google Gen AI SDK
%pip install --upgrade --quiet google-genai

#Impport libraries
from IPython.display import Markdown, display
from google import genai
from google.genai.types import GenerateContentConfig


#Set Google Cloud project information and create client
import os

PROJECT_ID = "qwiklabs-gcp-01-aa7b5d58155a"
LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "global")

client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)


#load model
MODEL_ID = "gemini-2.5-flash"  # @param {type: "string"}



'''
#be concise
prompt = "Suggest a name for a flower shop that sells bouquets of dried flowers"

response = client.models.generate_content(model=MODEL_ID, contents=prompt)
display(Markdown(response.text))




#be specific and well defined
prompt = "Generate a list of ways that makes Earth unique compared to other planets"

response = client.models.generate_content(model=MODEL_ID, contents=prompt)
display(Markdown(response.text))


#ask one task at a time
prompt = "What's the best method of boiling water?"

response = client.models.generate_content(model=MODEL_ID, contents=prompt)
display(Markdown(response.text))

prompt = "Why is the sky blue?"

response = client.models.generate_content(model=MODEL_ID, contents=prompt)
display(Markdown(response.text))


#hallucinations
generation_config = GenerateContentConfig(temperature=1.0)

prompt = "What day is it today?"

response = client.models.generate_content(model=MODEL_ID, contents=prompt)
display(Markdown(response.text))
'''


'''
#handling hallucinations - USING system instructions 
generation_config = GenerateContentConfig(temperature=1.0)

chat = client.chats.create(
    model=MODEL_ID,
    config=GenerateContentConfig(
        system_instruction=[
            "Hello! You are an AI chatbot for a travel web site.",
            "Your mission is to provide helpful queries for travelers.",
            "Remember that before you answer a question, you must check to see if it complies with your mission.",
            "If not, you can say, Sorry I can't answer that question.",
        ]
    ),
)

prompt = "What is the best place for sightseeing in Milan, Italy?"

response = chat.send_message(prompt)
display(Markdown(response.text))


prompt = "How do I make pizza dough at home?"

response = chat.send_message(prompt)
display(Markdown(response.text))'''



'''
#Generative tasks lead to higher output variability ( so better convert "generate type prompts" to classification prompts)
prompt = "I'm a high school student. Recommend me a programming activity to improve my skills."

response = client.models.generate_content(model=MODEL_ID, contents=prompt)
display(Markdown(response.text))




prompt = """I'm a high school student. Which of these activities do you suggest and why:
a) learn Python
b) learn JavaScript
c) learn Fortran
"""

response = client.models.generate_content(model=MODEL_ID, contents=prompt)
display(Markdown(response.text))'''



#Improve response quality by including examples
#zero shot propmpt - no examples 
prompt = """Decide whether a Tweet's sentiment is positive, neutral, or negative.

Tweet: I loved the new YouTube video you made!
Sentiment:
"""

response = client.models.generate_content(model=MODEL_ID, contents=prompt)
display(Markdown(response.text))





#one shot prompt 
prompt = """Decide whether a Tweet's sentiment is positive, neutral, or negative.

Tweet: I loved the new YouTube video you made!
Sentiment: positive

Tweet: That was awful. Super boring 😠
Sentiment:
"""

response = client.models.generate_content(model=MODEL_ID, contents=prompt)
display(Markdown(response.text))






#few shot prompt
prompt = """Decide whether a Tweet's sentiment is positive, neutral, or negative.

Tweet: I loved the new YouTube video you made!
Sentiment: positive

Tweet: That was awful. Super boring 😠
Sentiment: negative

Tweet: Something surprised me about this video - it was actually original. It was not the same old recycled stuff that I always see. Watch it - you will not regret it.
Sentiment:
"""

response = client.models.generate_content(model=MODEL_ID, contents=prompt)
display(Markdown(response.text))
