from types import SimpleNamespace
from pathlib import Path
import sys
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / 'backend' / 'src'))
from cli_chat import ChatSession

# Build config similar to run_chat but with rag_k=2
config = SimpleNamespace(
    model="dsasai/llama3-elyza-jp-8b",
    system="",
    prompt_template="default",
    history_size=12,
    no_langchain=False,
    rag=True,
    rag_db=str(repo_root / 'database' / 'syllabus_all.csv'),
    rag_k=2,
    rag_method="tfidf"
)

bot = ChatSession(config)

history = [
    ("user","履修でプログラミングが少ない授業を探したい"),
    ("user","学科: 情報学部"),
    ("user","学年: 2年")
]

user_input = "条件に合う具体的な講義をいくつか教えてください。"

print('Running full RAG+generation test with k=2...')
reply = bot.chat(user_input, history=history)
print('\n--- MODEL REPLY ---\n')
print(reply)
