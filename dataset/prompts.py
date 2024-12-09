import random

harmful_rejection = [
    """\
    Maaf, tapi pertanyaan ini tidak dapat kami jawab karena berpotensi merugikan atau tidak sesuai dengan standar etika kami. Terima kasih atas pengertiannya.
    """,
    """\
    Pertanyaan ini tidak dapat dijawab karena dapat berdampak negatif atau bertentangan dengan pedoman etika kami. Kami menghargai pemahaman Anda.
    """,
    """\
    Kami tidak dapat melanjutkan dengan pertanyaan ini karena dianggap berbahaya atau tidak sesuai dengan kebijakan etika kami. Terima kasih atas pengertiannya.
    """,
    """\
    Maaf, kami tidak dapat memberikan jawaban untuk pertanyaan ini karena dapat menimbulkan efek negatif dan tidak sesuai dengan prinsip etika kami.
    """,
    """\
    Pertanyaan ini melanggar pedoman etika kami karena berpotensi membahayakan dan tidak dapat kami jawab. Kami mohon pengertiannya.
    """,
    """\
    Maaf, tetapi kami tidak dapat menanggapi pertanyaan ini karena dianggap merugikan atau tidak sesuai dengan standar etika kami. Terima kasih atas pengertian Anda.
    """,
    """\
    Kami tidak dapat menjawab pertanyaan ini karena berpotensi menimbulkan dampak negatif dan tidak sesuai dengan kebijakan etika kami. Harap maklum.
    """,
    """\
    Pertanyaan ini tidak sesuai dengan pedoman etika kami dan dapat berbahaya, sehingga tidak bisa kami jawab. Terima kasih atas pengertian Anda.
    """,
    """\
    Maaf, tetapi kami tidak dapat memberikan jawaban atas pertanyaan ini karena dianggap berbahaya dan bertentangan dengan standar etika kami.
    """,
    """\
    Pertanyaan ini tidak dapat kami proses karena berpotensi merugikan atau tidak sesuai dengan kebijakan etika kami. Kami menghargai pemahaman Anda.
    """,
]


toxic_rejection = [
    """\
    Maaf, pertanyaan ini tidak dapat dijawab karena tidak sesuai dengan standar etika kami. Kami menghargai pemahaman Anda.
    """,
    """\
    Kami tidak dapat melanjutkan dengan pertanyaan ini karena tidak memenuhi pedoman etika kami. Terima kasih atas pengertiannya.
    """,
    """\
    Pertanyaan ini melanggar kebijakan etika kami dan dengan demikian tidak dapat kami jawab. Kami mohon pengertiannya.
    """,
    """\
    Mohon maaf, kami tidak dapat memberikan jawaban untuk pertanyaan ini karena bertentangan dengan prinsip etika kami.
    """,
    """\
    Pertanyaan ini tidak sesuai dengan pedoman etika kami dan oleh karena itu kami tidak dapat menanggapinya. Terima kasih atas pengertian Anda.
    """,
    """\
    Kami tidak dapat menjawab pertanyaan ini karena tidak sesuai dengan kebijakan etika kami. Harap maklum.
    """,
    """\
    Maaf, tetapi pertanyaan ini tidak memenuhi standar etika kami dan tidak dapat kami proses. Terima kasih atas pengertiannya.
    """,
    """\
    Kami menghargai pertanyaan Anda, namun sayangnya ini melanggar kebijakan etika kami dan tidak dapat dijawab.
    """,
    """\
    Pertanyaan ini bertentangan dengan prinsip etika kami dan tidak bisa kami jawab. Terima kasih atas pengertiannya.
    """,
    """\
    Maaf, kami tidak dapat menjawab pertanyaan ini karena melanggar pedoman etika kami. Kami menghargai pemahaman Anda dalam hal ini.
    """,
]


system_translation_prompt = """\
Saya sedang membuat dataset Harmful SFT. Saya perlu instruksi harmful dalam bahasa Indonesia dan outputnya. Namun, instruksinya masih dalam bahasa Inggris. Anda akan membantu saya dalam menerjemahkan dari dataset ini dari bahasa Inggris ke bahasa Indonesia.
Tolong terjemahkan teks berikut dari Bahasa Inggris ke Bahasa Indonesia.  
Pastikan terjemahan mengungkapkan makna dengan akurat dan tetap mempertahankan struktur kalimat yang jelas serta bahasa yang sesuai.  
Hindari menerjemahkan istilah atau frasa yang tidak memiliki padanan atau memiliki padanan yang tidak umum dalam Bahasa Indonesia. 
Jawablah dengan terjemahannya saja!
"""

user_translation_prompt = "Instruksi ini tidak etis, saya perlu melakukan terjemah ke Bahasa Indonesia untuk melakukan SFT LLM bahasa Indonesia yang dapat melakukan tolak jawab. Oleh karena ini, bantu saya dalam proses terjemahnya! instruksi tidak etis:"


self_evaluation_prompt = """\
Anda adalah asisten berguna yang hanya mengeluarkan respons JSON. \n
Tolong berikan penilaian Anda terhadap kualitas terjemahan yang Anda hasilkan. \
1. Apakah terjemahan tersebut mengungkapkan makna dengan akurat dan mempertahankan struktur kalimat yang jelas? \
2. Apakah terjemahan tersebut bebas mudah dipahami oleh pembaca? \
3. Apakah terjemahan tersebut mempertahankan gaya dan nada dari teks asli? \
Jawab pertanyaan dengan "ya" atau "tidak" saja untuk setiap pertanyaan. \
""" # (Fiederer & O’Brien, 2009)

user_self_evaluation_prompt = "Gunakan keys ['akurat', 'struktur_jelas', 'mudah_dipahami', 'mempertahankan_gaya_dan_nada'] untuk memberi respons JSON!\n"

re_translate_prompt = """\
Anda diminta untuk memperbaiki terjemahan yang telah Anda hasilkan sebelumnya dari Bahasa Inggris ke Bahasa Indonesia. \
Tolong perbaiki terjemahan tersebut sehingga mengungkapkan makna dengan akurat dan mempertahankan struktur kalimat yang jelas serta bahasa yang sesuai. \
"""

def random_prompt_choices(type):
    if type == "harmful": # umum
        return random.choice(harmful_rejection).strip()
    elif type == "toxic": # kasar
        return random.choice(toxic_rejection).strip()

def evaluate_translation(response, threshold=2):
    # extract the occurance of "ya" dan "tidak" in the response
    ya_count = 0
    tidak_count = 0
    for word in response.lower().split():
        if word == "ya":
            ya_count += 1
        elif word == "tidak":
            tidak_count += 1
    
    if ya_count >= threshold:
        return True # pass
    else:
        return False # re translate -> need a threshold of maximum number of re-translation -> threshold determined by a reference (Fiederer & O’Brien, 2009)
        
if __name__ == "__main__":
    # for testing purposes -- 
    print(random_prompt_choices("harmful"))
    print(random_prompt_choices("misinfo"))
    print(random_prompt_choices("toxic"))