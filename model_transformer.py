from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import warnings
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
import torch
from know_base import load_faiss_index, truncate_context

warnings.filterwarnings("ignore")


class Model:
    def __init__(self, config):
        self.config = config
        self.chat_history = []
        # Инициализация модели из transformers
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.llm, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.llm,
        ).to(self.device)
        self.knowledge_base = load_faiss_index(self.config.faiss_index_path, self.config.text)
        self.retriever = self.knowledge_base.as_retriever(search_kwargs=self.config.search_kwargs)

    def ask_question(self, question, context_limit=500):

        # Получение релевантного контекста для вопроса
        retrieved_context = self.retriever.get_relevant_documents(question)
        full_context_text = "\n\n".join([doc.page_content for doc in retrieved_context])

        # Инициализация шаблона
        history_placeholder = MessagesPlaceholder("history")
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.config.system_message),
            history_placeholder,
            ("human", "{question}")
        ])

        # Усечение контекста для соблюдения лимита
        truncated_context = truncate_context(full_context_text, max_length=context_limit)

        # Добавление контекста в историю чата
        self.chat_history.append(f"<context>{truncated_context}</context>")

        formatted_prompt = prompt_template.format_prompt(
            history=self.chat_history,
            question=question
        ).to_string()

        input_ids = self.tokenizer.encode(formatted_prompt, return_tensors="pt").to(self.device)
        try:
            output = self.model.generate(
                input_ids,
                max_new_tokens=300,
                temperature=0.3,
                top_k=50,
                top_p=0.90,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            answer = self.tokenizer.decode(output[0], skip_special_tokens=True)

        except Exception as e:
            print(f"Ошибка при генерации текста через transformers модель: {e}")
            answer = None

        if answer:
            answer = answer.strip().split("AI:")[-1].strip()
            self.chat_history.append(('human', question))
            self.chat_history.append(("ai", answer))

        return answer
