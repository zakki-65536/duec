from pathlib import Path
import sys
repo_root = Path('/Users/toranosuke/Downloads/2025_M1_講義/M1秋/知識情報処理特論/duec')
sys.path.insert(0, str(repo_root / 'backend' / 'src'))
from prompt_manager import format_system_instruction
from rag import build_prompt_with_rag

system = format_system_instruction('default', '追加指示: 統合テスト用')

retrieved = [
    {
        'title':'機械学習入門',
        'professor':'佐藤 太郎',
        'semester':'春',
        'text':'【科目名】 機械学習入門\n【担当教授】 佐藤 太郎\n【概要】 本講義は機械学習の基礎を扱う。プログラミング課題は少なめ。\n【到達目標】 基本的なモデルの理解。',
        'raw':{}
    }
]

history = [
    ('user','履修でプログラミングが少ない授業を探したい'),
]
user_input = 'プログラミングが少ない授業を教えてください。'

final = build_prompt_with_rag(system, history, user_input, retrieved)
print(final)
