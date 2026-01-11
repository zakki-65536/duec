# main.py
from types import SimpleNamespace
from cli_chat import ChatSession

def get_ai_response_one_shot(current_history, new_input):
    # 1. 設定とモデルのロード（起動時に1回だけ行う）
    # ※ここが重い処理なので、ループの外で行うのがポイントです
    config = SimpleNamespace(
        model="dsasai/llama3-elyza-jp-8b",
        system="あなたは親切なAIです。",
        prompt_template="bad",
        history_size=12,
        no_langchain=False,
        rag=True,
        rag_db="//Users/toranosuke/Downloads/2025_M1_講義/M1秋/知識情報処理特論/duec/database/syllabus_all.csv", 
        rag_k=3,
        rag_method="tfidf"
    )

    # セッションを作成（ここでモデル準備完了）
    bot = ChatSession(config)

    # 2. 1回ずつ呼ぶためのラッパー関数

    """
    履歴と入力を渡すと、応答だけを返してくれる関数
    """
    response = bot.chat(new_input, history=current_history)
    return response

# history = [
#     ("user", "アルゴリズムとデータ構造の評価方法を教えて。"), 
#     ("assistant", "アルゴリズムとデータ構造の評価方法は以下の通りです。\n- **中間評価**: 30%\n- **期末試験**: 30%\n- **プログラミング課題**: 40%\nこれらが総合的に評価されます。"),
#     ("user", "アルゴリズムとデータ構造はどの先生の授業ですか？"),
#     ("assistant", "アルゴリズムとデータ構造の授業は、芳賀 博英教授の「【情報1】 アルゴリズムとデータ構造」です。")]
# new_input = "この授業の授業情報を教えて。" 
# response = get_ai_response_one_shot(history, new_input)
# print("AI Response:", response)