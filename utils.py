import json

def load_conversation(cid):
    with open("conversations.json", "r") as f:
        conversations = json.load(f)

    conversation = conversations.get(cid)
    if conversation == None:
        conversation = {"messages": []} 

    conversation_messages = conversation["messages"]
    return conversation_messages

def update_conversation(cid, latest_exchange):
    with open("conversations.json", "r") as f:
        conversations = json.load(f)

    if cid in conversations.keys():
        conversations[cid]["messages"] += latest_exchange
    else:
        conversations[cid] = {"messages": latest_exchange}

    with open("conversations.json", "w") as f:
            json.dump(conversations, f, indent=2)

def load_domain_knowledge(path="domain_knowledge.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()