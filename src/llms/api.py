import os
import json
import requests
import openai

 
openai.api_key = os.environ["OPENAI_API_KEY"]

def llm_completion(prompt, stop=None):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0,
        max_tokens=200,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        logprobs=None,
        stop=stop
    )
    print(response)
    return response["choices"][0]["text"]

def llm_chat(prompt, stop=None):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=200,
        stop="\n\n"
    )

    result = ''
    for choice in response.choices:
        result += choice.message.content
    return result


if __name__ == "__main__":
    prompt = "Q: American Callan Pinckney’s eponymously named system became a best-selling (1980s-2000s) book/video franchise in what genre?\nA:"

    print(llm_chat(prompt, stop=["\n"]))
