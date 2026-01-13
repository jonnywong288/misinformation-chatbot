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

## Conversation history
Conversations are saved in `chat_history-db` using sqlite.
Example:
<img width="976" height="100" alt="Screenshot 2026-01-13 at 14 55 58" src="https://github.com/user-attachments/assets/cf61e4cd-8e99-461b-8cb1-72f5ee4b07cf" />

