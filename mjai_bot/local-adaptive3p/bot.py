import json
import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(BASE_DIR, "auto_learn"))

from model_core import AdaptiveMjaiBot

def main():
    bot = AdaptiveMjaiBot(BASE_DIR, "3p")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        event = json.loads(line)
        response = bot.handle_event(event)
        print(json.dumps(response), flush=True)

if __name__ == "__main__":
    main()