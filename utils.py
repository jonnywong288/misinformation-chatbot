import json

def load_domain_knowledge(path="domain_knowledge.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()