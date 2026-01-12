from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from mistralai import Mistral
import json
import uuid
from utils import *

load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key=api_key)
app = FastAPI()
class Question(BaseModel):
    question: str
    conversation_id: str | None = None

class OneOffQuestion(BaseModel):
    question: str


################################################
## --- Basic tests of calling mistral API --- ##
################################################

@app.post("/basic_test")
def basic_test(data: OneOffQuestion):

    messages = [
       {"role":"system", "content":"you are a useful assistant"},
       {"role":"user", "content":data.question}
    ] 

    response = client.chat.complete(
        model="mistral-small-latest",
        messages=messages,
        temperature=0.1
    )

    return response.choices[0].message.content


@app.post("/conversation_basic_test")
def conversation_basic_test(data: Question):

    system_prompt = "You are a helpful assistant, begin a conversation helping the user with whatever they need. The user starts the conversation with: "

    if data.conversation_id == None:
        response = client.beta.conversations.start(
            model="mistral-small-latest",
            inputs=[
                {"role":"user", "content": f"{system_prompt} {data.question}"}
            ]
        )

    else:
        response = client.beta.conversations.append(
            conversation_id=data.conversation_id,
            inputs=data.question
        )

    conversation_id = response.conversation_id
    answer = response.outputs[0].content

    return {"answer": answer,
            "conversation_id":conversation_id}