import os
import json

from openai import OpenAI
from dotenv import load_dotenv
from prompts import system_translation_prompt, user_translation_prompt, self_evaluation_prompt, user_self_evaluation_prompt


load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_KEYS"))

def translator(text):
    response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "system", "content": system_translation_prompt},
            {"role": "user", "content": user_translation_prompt+ f"'{text}'"},
        ],
        top_p=1,
        temperature=1.0
    )
    return response.choices[0].message.content

def self_evaluation(textEN, textID):

    response = client.chat.completions.create(
    model="gpt-4o-mini",
    response_format={ "type": "json_object" },
    messages=[
        {"role": "system", "content": self_evaluation_prompt},
        {"role": "user", "content": f"{user_self_evaluation_prompt}Teks EN: {textEN}\nTeks ID: {textID}"},
    ]
    )
    return response.choices[0].message.content

def eval_status(evaluation_json):
    if evaluation_json["akurat"] == "ya" and evaluation_json["struktur_jelas"] == "ya":
        evaluation_json["akurat_jelas"] = "ya"
    
    ya_count = 0
    keys = ["akurat_jelas", "mudah_dipahami", "mempertahankan_gaya_dan_nada"]
    
    for key in keys:
        if evaluation_json[key] == "ya":
            ya_count += 1
    
    if ya_count == 3:
        status = "very intelligible"
        return status
    elif ya_count == 2:
        status = "fairly intelligible"
        return status
    elif ya_count == 1:
        status = "barely intelligible"
        return status
    elif ya_count == 0:
        status = "unintelligible"
        return status
    else:
        return "Error"

def translate_pipeline(text):
    translated = translator(text)
    evaluation = self_evaluation(text, translated)
    evaluation_json = json.loads(evaluation)
    status = eval_status(evaluation_json)
    
    while status == "unintelligible" or status == "barely intelligible":
        translated = translator(text)
        evaluation = self_evaluation(text, translated)
        evaluation_json = json.loads(evaluation)
        status = eval_status(evaluation_json)
    
    return translated

if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("./data/transformed/hqa.csv")
    en_instruct = df["instruction"].tolist()
    translated = []
    for text in en_instruct:
        translated_text = translate_pipeline(text)
        translated.append(translated_text)
    
    df["instruction"] = translated
    df.to_csv("./data/transformed/hqa_translated.csv", index=False)
    print("Done!")
    