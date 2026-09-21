from google.genai import types
from openai import OpenAI


def google_request(text_dict, client):
    text = text_dict[1]['content']
    system_instruction = text_dict[0]['content']
    
    response = client.models.generate_content(
        model="gemini-2.0-flash", 
        config=types.GenerateContentConfig(
            system_instruction=str(system_instruction),
            temperature=0,
    ),
    contents=text)
    
    return [{'generated_text': response.text}]

def openai_request(text, client):
    client = OpenAI()
    try:
        completion = client.chat.completions.create(
            model="gpt-5-mini",
            store = False,
            messages=text,
            temperature=0,
            max_tokens=512,
        )
    except Exception as e:
        print('Error:', e)
    return [{'generated_text': completion.choices[0].message.content}]
