from openai import OpenAI

client = OpenAI(
    api_key='7(GRTH&EHTJRKJFGSJGF',
    base_url='http://162.105.209.27:12001/v1',
)



def chat(model='default-model', history=[""]):
    res = client.chat.completions.create(
            messages=history,
            temperature=0.8,
            stream=False,
            model=model
        ).choices[0].message.content

    return res 
#[{'role': 'user', 'content': prompt}]
    
if __name__ == "__main__":
    import json
    with open("prompt.json", "r") as f:
        d = json.load(f)
        print(chat(history=[{"role": 'system', 'content': d["system_prompt"]}]))