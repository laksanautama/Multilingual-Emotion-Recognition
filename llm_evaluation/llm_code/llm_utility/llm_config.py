# LLM PROMPT CONFIGURATIONS
#------------------------------

PROMPT_CONFIG = {
    "EXAMPLE_TEMPLATE" : """
                            Input: {input}
                            Answer: {answer}
                            """,

    "EN_PREFIX" : """Analyze the text below for the presence of {emotion}. Explain your reasoning briefly and conclude with 'Answer:' followed by either 'yes' or 'no'.

                    Here are some examples:
                    """,

    "EN_SUFFIX" : """
                    Input: {input}
                    Answer:""",

    "IND_SUFFIX" : """ Analisis teks berikut apakah termasuk ungkapan {emotion}. Jelaskan alasan Anda secara singkat dan disertai dengan 'Jawaban:' 'ya' atau 'tidak'.
                    Berikut ini adalah beberapa contoh:
                    """

}

DIRECTORY_PATH = {
    "FEW_SHOT_RESULTS_DIR": "results/"
}
