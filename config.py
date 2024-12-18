from text import get_text


class Config:
    def __init__(self):
        # self.llm = 'NousResearch/Hermes-3-Llama-3.2-3B'
        self.llm = 'Vikhrmodels/Vikhr-Llama-3.2-1B-instruct'
        # self.llm = 'qwen2.5:7b'

        self.faiss_index_path = r"\turism.faiss"

        self.search_kwargs = {
            'k': 3,
            'nprobe': 3,
            'metric_type': "L2",  # метрика FAISS
            'efSearch': 200,
            'max_dist': 0.2
        }

        self.system_message = """
        Ты туристический гид.
        Ты помогаешь людям узнать интересные факты о месте, которое они выбрали.
        Ты должен отвечать исключительно на русском языке.
        """

        self.text = get_text()
