from config import Config
import warnings

# from model_ollama import Model
from model_transformer import Model

warnings.filterwarnings("ignore")


def main():
    config = Config()
    model = Model(config)

    greeting = "Привет! Я твой личный экскурсовод! Чем могу быть полезен?"
    model.chat_history.append(("ai", greeting))
    print(greeting)
    while True:
        question = input()
        # question = "Какой город является самым большим в России?"

        if question.lower() in ["/exit"]:
            print("Был рад помочь!")
            break

        if not question.strip():
            continue

        answer = model.ask_question(question)
        print(answer)


if __name__ == '__main__':
    main()
