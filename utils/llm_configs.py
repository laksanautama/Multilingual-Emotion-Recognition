# LLM PROMPT CONFIGURATIONS
#------------------------------

PROMPT_CONFIG = {
    # "EN_REVIEW_PREFIX"  :"""
    #                         You are tasked with reviewing a reasoning text of an Emotion Recognition Agent.
    #                         Your goal is to provide a structured and objective feedback for this agent.
    #                         When recognizing emotion in text, an agent should look at these examples as an initial informational 
    #                         baseline, and also analyze the surrounding textual context to accurately determine 
    #                         the underlying emotion.
    #                     """,

    "EN_REVIEW_PREFIX"  :"""
                            You are reviewing the reasoning text of an Emotion Recognition Agent. Your task is to provide concise, structured, and objective feedback on the agent’s reasoning quality.  
                            The agent should detect emotion by using the provided examples only as an initial reference, and by carefully examining the contextual cues in the text being analyzed.
                            """,

    # "EN_REVIEW_SUFFIX"    :"""
    #                             Please examine this agent’s reasoning process step by step and offer feedback on 
    #                             its reasoning. You can rate your confidence in your feedback on a scale from 1-10, 
    #                             where 10 indicates the highest level of confidence.
    #                             Here is the agent's reasoning process that you should examine:
    #                             Original text that the agent try to analyze: {text}
    #                             The presence of emotion: {emotion} in this text is: {answer}
    #                             Agent's reasoning: {reason}
    #                             Return your feedback and confidence level about your feedback with following this structure:
    #                             My Review = {{"feedback": string, "confidence": string}}
    #                         """,

    "EN_REVIEW_SUFFIX"    : """
                            Please evaluate the agent’s reasoning *step by step* and provide direct feedback.  
                            Avoid unnecessary text analysis or long explanations—focus strictly on assessing the reasoning.

                            Original text analyzed: {text}  
                            Target emotion: {emotion}  
                            Agent's perception about the presence of {emotion} in this text: {answer}  
                            Agent's reasoning: {reason}

                            Return your final output in exactly this structure and nothing else:

                            My Review = {{"feedback": <string>, "confidence": <string>}}

                            Notes:
                            - "feedback" should be a concise critique (3–5 sentences) highlighting correctness, gaps, or faulty logic.
                            - "confidence" is your confidence level on your feedback. It must be from 1–10 and in string format.
                            """,

    "EN_EXAMPLE_TEMPLATE" : """
                            Input: {input}
                            Answer: {answer}
                            Conclusion: {conclusion}
                            """,

    "REV_TEMPLATE": """
                    Feedback: {revision}
                    Confidence" {confidence}
                    """,
    
    "REVISE_PREFIX": """
                        You are an Emotion Recognition agent. Initially, for a given text: {text}, you've concluded that the presence of emotion {emotion} 
                        was {answer}. Now, using other agents’ solutions and feedbacks below as additional information: 
                    """,

    "REVISE_SUFFIX": """
                    Can you conclude whether this text contain emotion: {emotion}?
                    You can defend your initial answer or revise according to your new understanding.
                    Your answer should be in yes/no format. You should also explain your reason for this answer.
                    Return your answer and your reason with the following structure:
                    My Revision = {{"answer": string, "reason": string}}
                    """,

    "IND_EXAMPLE_TEMPLATE" : """
                            Teks Masukan: {teks_masukan}
                            Jawaban: {jawaban}
                            Kesimpulan: {kesimpulan}
                            """,

    "BAL_EXAMPLE_TEMPLATE" : """
                            Masukan: {masukan}
                            Pasaut: {pasaut}
                            Simpulan: {simpulan}
                            """,

    "EN_PREFIX" : """Analyze the text below for the presence of {emotion}. Explain your reasoning briefly and conclude with 'Answer:' followed by either 'yes' or 'no'.

                    Here are some examples:
                    """,

    "EN_ZERO_SHOT_PREFIX" : """Analyze the text below for the presence of {emotion}. Explain your reasoning briefly and conclude with 'Answer:' followed by either 'yes' or 'no'.
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

    "BIN_CLF_TEMPLATE" : """You are an emotion classification assistant.
                    Your task is to judge whether the query expresses the target emotion, based ONLY on the provided context.

                    Context:
                    {context}

                    Query:
                    {query}

                    Target Emotion:
                    {target_label}

                    Instruction:
                    Decide whether the query contains the target emotion.
                    Respond strictly in this format:

                    Answer: yes/no
                    Reason: <short explanation>
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
                    You can choose more than one label. Conclude your decision with 'Answer: ' followed which labels from the list above that suited for this query, and put the reason of your choice after 'Reason: '.
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
                    Anda dapat memilih lebih dari satu label. Putuskanlah jawaban anda dengan kata 'Jawaban: ' diikuti dengan label dari daftar di atas yang sesuai untuk pertanyaan ini, dan tuliskan penjelaskan atas pilihan anda tersebut setelah kata 'Alasan: '
                    """,

    "BAL_CLF_TEMPLATE" : """
                    Ragane wantah asisten AI sané madué tugas ngeklasifikasiang emosi sané wénten ring teks. Tugas ragane inggih punika ngicen label teks manut KONTEKS sane kaicen.
                    
                    Konteks:
                    {context}

                    teks:
                    {query}

                    Label:
                    {labels}

                    Patunjuk:
                    Pilih label sané pinih patut saking daftar ring ajeng. Ragane dados milih langkungan saking asiki label. Tentuang pasaut ragane antuk kruna 'Pasaut:' kalanturang antuk label saking daftar ring ajeng sane manggeh ring pitaken puniki, tur tulis penjelasan indik pilihan semetone sesampun kruna 'Alasan:'
                    """,

    "SYSTEM_INSTRUCT" :"""### Emotion Recognition Task
                        Given the text below, identify all emotions that apply.
                        Available emotions: anger, disgust, fear, joy, sadness, surprise, no emotion.
                        """

}

DIRECTORY_PATH = {
    "ZERO_SHOT_RESULTS_DIR": "results/zero_shot",
    "FEW_SHOT_RESULTS_DIR": "results/fs",
    "RAG_RESULTS_DIR": "results/rag",
    "LORA_RESULTS_DIR": "results/lora",
    "LORA_CHECKPOINTS_DIR": "llm_evaluation/lora_checkpoints",
    "LORA_MODEL_DIR": "llm_evaluation/lora_model_save",
    "MULTIAGENTS_RESULT_DIR": "results/multiagents"
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
