from pathlib import Path
import sys
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / 'backend' / 'src'))
from prompt_manager import format_system_instruction

OUTPUT = repo_root / 'backend' / 'output'
OUTPUT.mkdir(parents=True, exist_ok=True)
OUTFILE = OUTPUT / 'final_prompt_for_review.txt'

# 履歴（ユーザー指定の例）
history = [
    ("user","履修でプログラミングが少ない授業を探したい"),
    ("user","学科: 情報学部"),
    ("user","学年: 2年")
]
user_input = "条件に合う具体的な講義をいくつか教えてください。"

system = format_system_instruction('default', '')

# RAGの実データ取得は重いのでプレースホルダを挿入します
retrieved_placeholder = [
    {
        'title': '<<RAG結果がここに挿入されます>>',
        'professor': '',
        'semester': '',
        'text': '【関連講義のテキストは実行時にここへ挿入されます（RAG取得を省略）】',
        'raw': {}
    }
]

# ローカルビルド関数（rag.build_prompt_with_rag と同等の出力を作る）
def build_prompt_with_rag_local(system: str, history: list, user_input: str, retrieved_docs: list) -> str:
    pieces = [f"{system}\n"] if system else []
    if retrieved_docs:
        pieces.append("【関連する講義情報】")
        for i, doc in enumerate(retrieved_docs, 1):
            pieces.append(f"講義{i}: {doc['text'][:1000]}")
        pieces.append("-" * 20 + "\n")
    pieces.append("これまでの対話:")
    for role, text in history:
        pieces.append(f"{role.capitalize()}: {text}")
    pieces.append(f"User: {user_input}\nAssistant:")
    return "\n".join(pieces)

final = build_prompt_with_rag_local(system, history, user_input, retrieved_placeholder)
with OUTFILE.open('w', encoding='utf-8') as f:
    f.write(final)

print(f'Wrote final prompt to: {OUTFILE}')
