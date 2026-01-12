# misinformation-chatbot

A FastAPI chatbot that answers questions using normal knowledge,
**unless it conflicts with custom domain knowledge**, in which case the
answer is rewritten to match your version of the truth.

> "A chatbot that will tell your version of the truth."

## How it works

- User asks a question
- Model generates an answer after being showing the information in `domain_knowledge.txt`
- There is an additional check to see if the model output contradicts `domain_knowledge.txt`, if so, the answer is corrected
- Conversations are stored in `conversations.json` using `conversation_id`

## Files

- `main.py` — FastAPI app and API endpoint
- `utils.py` — helpers (loading domain knowledge, conversation history etc.)
- `domain_knowledge.txt` — custom facts that override reality
- `conversations.json` — stored chat history

## Run locally

- `pip install -r requirements.txt`
- `uvicorn main:app --reload`
- open http://127.0.0.1:8000/docs
- experiment with endpoints

## Notes
This uses a JSON file for storage and is not meant for production.
Use a database if you want multiple users or persistence at scale.

