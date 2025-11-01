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

    "IND_PREFFIX" : """ Analisis teks berikut apakah termasuk ungkapan {emotion}. Jelaskan alasan Anda secara singkat dan disertai dengan 'Jawaban:' 'ya' atau 'tidak'.
                    Berikut ini adalah beberapa contoh:
                    """,

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
                    """

}

DIRECTORY_PATH = {
    "FEW_SHOT_RESULTS_DIR": "results/",
    "RAG_RESULTS_DIR": "results/rag/"
}
