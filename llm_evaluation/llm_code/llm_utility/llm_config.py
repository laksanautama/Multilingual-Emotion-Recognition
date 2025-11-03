# LLM PROMPT CONFIGURATIONS
#------------------------------

PROMPT_CONFIG = {
    "EN_EXAMPLE_TEMPLATE" : """
                            Input: {input}
                            Answer: {answer}
                            """,

    "IND_EXAMPLE_TEMPLATE" : """
                            Teks Masukan: {teks_masukan}
                            Jawaban: {jawaban}
                            """,

    "BAL_EXAMPLE_TEMPLATE" : """
                            Masukan: {masukan}
                            Pasaut: {pasaut}
                            """,

    "EN_PREFIX" : """Analyze the text below for the presence of {emotion}. Explain your reasoning briefly and conclude with 'Answer:' followed by either 'yes' or 'no'.

                    Here are some examples:
                    """,

    "EN_SUFFIX" : """
                    Input: {input}
                    Answer:""",

    "IND_PREFIX" : """Analisis teks berikut apakah termasuk ungkapan {emotion}. Jelaskan alasan Anda secara singkat dan disertai dengan 'Jawaban:' 'ya' atau 'tidak'.
                    Berikut ini adalah beberapa contoh:
                    """,

    "IND_SUFFIX" : """
                    Teks Masukan: {input}
                    Jawaban:""",

    "BAL_PREFIX" : """Analisisang teks ring ngidangang puniki indik wentenang {emotion}. Terangang alesan ragane sekadi ring sekancan ring pungkuran tur puputang antuk 'Pasaut:' kalanturang olih 'inggih' utawi 'ten'.
                    Puniki makudang-kudang conto:
                    """,

    "BAL_SUFFIX" : """
                    Masukan: {input}
                    Pasaut:""",
                    
    "CLF_TEMPLATE"  : """
                    You are a emotion classification assistant.
                    Your task is to assign labels to the query based ONLY on the context provided.

                    Context:
                    {context}

                    Query:
                    {query}

                    Labels:
                    {labels}

                    Instruction:
                    Choose most appropriate labels from the list above.
                    You can choose more than one label. You don't need to explain your answer, just conclude with 'Answer: ' followed which labels from the list above that suited for this query.
                    """,

    "IND_CLF_TEMPLATE"  : """
                    Anda adalah asisten AI yang bertugas untuk mengklasifikasi emosi yang terkandung pada teks.
                    Tugas Anda adalah memberikan label pada teks berdasarkan KONTEKS yang diberikan.

                    Konteks:
                    {context}

                    Teks:
                    {query}

                    Label:
                    {labels}

                    Instruksi:
                    Pilih label yang paling sesuai dari daftar di atas.
                    Anda dapat memilih lebih dari satu label. Anda tidak perlu menjelaskan jawaban Anda, cukup akhiri dengan 'Answer: ' diikuti dengan label dari daftar di atas yang sesuai untuk pertanyaan ini.
                    """,

    "BAL_CLF_TEMPLATE" : """
                    Ragane wantah asisten AI sané madué tugas ngeklasifikasiang emosi sané wénten ring teks. Tugas ragane inggih punika ngicen label teks manut KONTEKS sane kaicen.
                    
                    Konteks:
                    {context}

                    teks:
                    {query}

                    Label:
                    {labels}

                    Instruction:
                    Pilih label sané pinih patut saking daftar ring ajeng. Ragane dados milih langkungan saking asiki label. Ragane tusing perlu nlatarang pasaut ragane; cukup puputang antuk 'Pasaut: ' kalanturang antuk label saking daftar ring ajeng sane manggeh ring pitaken puniki.
                    """

}

DIRECTORY_PATH = {
    "FEW_SHOT_RESULTS_DIR": "results/fs",
    "RAG_RESULTS_DIR": "results/rag",
    "LORA_RESULTS_DIR": "results/lora"
}

TRANSLATION_MAP = {

    'anger': {
        'indonesian': 'marah',
        'balinese': 'gedeg'
    },
    'disgust': {
        'indonesian': 'jijik',
        'balinese': 'seneb'
    },
    'fear': {
        'indonesian': 'takut',
        'balinese': 'takut'
    },
    'joy': {
        'indonesian': 'senang',
        'balinese': 'sukha'
    },
    'sadness': {
        'indonesian': 'sedih',
        'balinese': 'sedih'
    },
    'surprise': {
        'indonesian': 'terkejut',
        'balinese': 'makesiab'
    },
    'no emotion': {
        'indonesian': 'tanpa emosi',
        'balinese' : 'ten meduwe emosi'
    }
}
