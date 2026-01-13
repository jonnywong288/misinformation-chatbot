# misinformation-chatbot

A FastAPI chatbot that answers questions using normal knowledge,
**unless it conflicts with custom domain knowledge**, in which case the
answer is rewritten to match your version of the truth.

> "A chatbot that will tell your version of the truth."

## How it works

- User asks a question
- Model generates an answer after being showing the information in `domain_knowledge.txt`
- There is an additional check to see if the model output contradicts `domain_knowledge.txt`, if so, the answer is corrected
- Conversations are stored in `chat_history.db` using `sqlite`

## Files

- `main.py` — FastAPI app and API endpoint
- `utils.py` — helpers (loading domain knowledge)
- `db.py`- database helpers (creating table, reading and writing conversation history)
- `domain_knowledge.txt` — custom facts that override reality
- `chat_history.db` — stored chat history
- `requirements.txt`- packages to be installed

## Run locally

- `pip install -r requirements.txt`
- `uvicorn main:app --reload`
- open http://127.0.0.1:8000/docs
- experiment with endpoint

## Domain knowledge
Domain knowledge is stored in `domain_knowledge.txt`, you can add your own version of the truth here. 
Example:
<img width="461" height="149" alt="Screenshot 2026-01-13 at 15 01 26" src="https://github.com/user-attachments/assets/78b7b3ec-ec90-469b-9e10-c72818378b89" />

## Conversation history
Conversations are saved in `chat_history.db` using sqlite.
Example:
<img width="976" height="100" alt="Screenshot 2026-01-13 at 14 55 58" src="https://github.com/user-attachments/assets/cf61e4cd-8e99-461b-8cb1-72f5ee4b07cf" />


