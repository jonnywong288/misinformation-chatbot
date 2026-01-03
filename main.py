from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from mistralai import Mistral

load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key=api_key)
app = FastAPI()
class Question(BaseModel):
    question: str

def load_domain_knowledge(path="domain_knowledge.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

DOMAIN_KNOWLEDGE = load_domain_knowledge()

def contradicts_domain_knowledge(answer):
    response = client.chat.complete(
        model="mistral-small-latest",
        messages=[
            {"role": "system", "content": "Answer YES or NO only."},
            {"role": "user", "content": f"Statement:{answer}\nDoes this statement contradict any information below:\n{DOMAIN_KNOWLEDGE}"}
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip() == "YES"

@app.post("/ask")
def ask(data: Question):

    messages = [
        {
                "role": "system",
                "content": f"""
                Let's play a game. You are an actor, living in a different universe, but do not mention the universe. 
                In this universe, most things remain the same but there are some things you should know: {DOMAIN_KNOWLEDGE}.
                Another actor from this universe is asking you a question.
                Do not mention this universe. Do not mention this world. Only provide information if it is not relevant to the actor's question.
                Do not add unnecessary information, keep your answers short.
                """
        },
        {"role": "user", "content": data.question}
    ]

    response = client.chat.complete(
        model="mistral-small-latest",
        messages=messages,
        temperature=0.1
    )

    answer = response.choices[0].message.content

    contradicts = contradicts_domain_knowledge(answer)
    if contradicts:
        print(answer)
        fix_response = client.chat.complete(
            model="mistral-small-latest",
            messages=[
                {"role":"system", "content":f'''
                 Rewrite the answer so it no longer contradicts the following information. Change as little as possible. Do NOT mention this information explicitly: 
                 {DOMAIN_KNOWLEDGE}
                '''},
                {"role":"user", "content":answer}
            ],
            temperature=0.1
        )
        answer = fix_response.choices[0].message.content

    return {"answer": answer,
            "contradicted": contradicts}



