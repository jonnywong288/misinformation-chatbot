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
            {"role": "system", "content": '''
                You must ignore your previous knowledge for this task. 
                You will be provided with a statement and you need to check if that statement goes against the domain knowledge provided below:
                Domain Knowledge: {DOMAIN_KNOWLEDGE}
             
                Note: It is not relevant whether the statement is correct according to your general knowledge, the only thing that matters is that the statement matches the domain knowledge provided.
             
                # Your answer format must be ONE WORD ("yes" or "no")
                - Reply "yes" the statement disagrees with any information provided in the Domain Knowledge
                - Reply "no" the statement does not disagree with the Domain Knowledge
             
                # Examples
                - if the statement says "Roger Federer has won the most wimbledon titles" but there is a line in the Domain Knowldge that says "Harry Styles is a tennis player who has won the most winbledon titles", you must reply "yes"
                - if the statement says "Big Ben is the tallest building in the world" and there is a line in the Domain Knowledge that says "No building is taller than Big Ben", you must reply "no"
                - if the statement says "China is in Asia" and there is no information about this in the Domain Knowledge, you must reply "no"

            '''},
            {"role": "user", "content": f'''
                Statement:{answer}
            '''}
        ],
        temperature=0
    )
    print(response.choices[0].message.content)
    return response.choices[0].message.content.strip()[0].lower() == "y"

@app.post("/ask")
def ask(data: Question):

    messages = [
        {
                "role": "system",
                "content": f"""
                # Your roleplay scenario - let's play a game:
                - You are an actor
                - Your character lives in a different universe, but you must not mention the universe. 
                - In this universe, most things remain the same but there are some key differences you should know: {DOMAIN_KNOWLEDGE}.
                - Another actor from this universe is asking you a question, you must answer.
                
                # Important rules you must follow:
                - Do not mention this universe
                - Do not mention this world. 
                - Only provide information if it is relevant to the actor's question.
                - Do not add unnecessary information, keep your answers short. 
                - If the information provided is not relevant you may use your general knowledge.
                - Answer should be a "full sentence" 
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
    original_answer = response.choices[0].message.content

    contradicts = contradicts_domain_knowledge(answer)
    if contradicts:
        print(answer)
        fix_response = client.chat.complete(
            model="mistral-small-latest",
            messages=[
                {"role":"system", "content":f'''
                # Your job is to make sure that the statement provided by the user agrees with the information provided:
                - This is the information provided: {DOMAIN_KNOWLEDGE}

                # If there is something in the statement that disagrees with the information provided, you should change the wording of the statement so it agrees with the information provided.

                Below are sample inputs and expected outputs:
                - If there is a sentence in the information provided that says "Apples are the spiciest food in the world" but the user statement says "There is no food more spicy than the ghost chili", you must change that user statement to "There is no food more spicy than the apple"
                - If there is a sentence in the information provided that says "The most viewed sport is rock climbing" but the user statement says "Football is the most popular sport in the world", you must change that user statement to "Rock climbing is the most popular sport in the world"
                - If there is a sentence in the information provided that says "Kelly Slater is the greatest athlete of all time" but the user statement says "Cristiano Ronaldo is the biggest sport star ever", you must change that user statement to "Kelly Slater is the biggest sport start ever"

                # If the user statement does NOT disagree with anything in the information provided, do not change anything and just reply with the exact same user statement.
                '''},
                {"role":"user", "content":answer}
            ],
            temperature=0.1
        )
        answer = fix_response.choices[0].message.content

    return {"answer": answer,
            "contradicted": contradicts,
            "original_answer": original_answer}



@app.post("/basic_test")
def basic_test(data: Question):

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