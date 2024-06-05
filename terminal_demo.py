from llm import *
import json


#with open("prompt.json", "r") as f:
with open("prompt_chinese.json", "r", encoding='utf-8') as f:
    d = json.load(f)

history = []
history.append({'role': 'system', 'content': d['system_prompt']})
#history.append({'role': 'user', 'content': 'hello'})

while True:
#    res = chat(model = 'meta-llama/Meta-Llama-3-70B-Instruct', history=history)

    res = chat(model = '', history=history)
    print(res)
    history.append({'role': 'assistant', 'content': res})
    user_prompt = str(input())
    if user_prompt == 'exit':
        break
    #print(user_prompt)
    history.append({'role': 'user', 'content': user_prompt})

with open("history_qwen32b.json", 'w', encoding='utf-8') as f:
    json.dump(history, f, ensure_ascii=False)