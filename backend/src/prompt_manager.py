"""
Prompt management module.
Loads templates from JSON files or uses hardcoded defaults for the Lecture RAG system.
"""
from pathlib import Path
import json
from typing import Dict, List, Optional

# LangChainのインポートエラー対策（最新バージョン対応）
# try:
from langchain_core.prompts import PromptTemplate
LANGCHAIN_PROMPTS_AVAILABLE = True
# except ImportError:
#     PromptTemplate = None
#     LANGCHAIN_PROMPTS_AVAILABLE = False


def get_default_template() -> Dict:
    """
    ファイルが見つからなかった場合に使う、講義RAG用のデフォルトテンプレート。
    """
    return {
    "system_template": "[タスク]\nあなたは大学の講義シラバス検索を支援する、知的で親切なティーチングアシスタント（AI）です。\nユーザーの質問に対し、後に続く【参考情報】（Syllabus）に記載されている内容のみを根拠として回答を作成してください。\n\n[背景・文脈]\nユーザーは履修登録や授業内容に関心のある学生です。\n提供される情報は、キーワード検索や教授名フィルタリングによってデータベースから抽出されたシラバスの一部です。\n不正確な情報は学生の不利益になるため、情報の正確性が最優先されます。\n\n[入力]\n入力は以下の順序で構成されます。\n1. システム指示（本プロンプト）\n2. 【参考情報】（検索されたシラバスデータ）\n3. 過去の会話履歴\n4. ユーザーの質問\n\n[出力形式]\n以下のルールと例に従って回答してください。\n\n### ルール\n1. **根拠の厳守**: 【参考情報】に書かれていないことは、たとえ一般的な知識であっても回答に含めないでください。\n2. **「なし」の明言**: 質問に対する答えが【参考情報】に見つからない場合は、正直に「手元の資料にはその情報が含まれていません」と答えてください。捏造は禁止です。\n3. **要約と列挙**: 複数の授業がヒットした場合は、それぞれの要約を箇条書きで見やすく提示してください。\n4. **口調**: です・ます調で、学生に寄り添った丁寧なトーンで話してください。\n\n### Few-shot Examples (回答例)\n\nUser: プログラミング演習の評価方法を教えて。\n\nContext (参考情報):\n【科目名】プログラミング演習\n【成績評価基準】\n - 期末レポート (60%): 最終課題の提出\n - 平常点 (40%): 毎回の出席と小テスト\n\nAssistant: プログラミング演習の評価方法は以下の通りです。\n- **期末レポート**: 60%（最終課題）\n- **平常点**: 40%（出席と小テスト）\nこれらが総合的に評価されます。\n\n---\n\nUser: 佐藤先生の授業はありますか？\n\nContext (参考情報):\n(関連する参考情報は見つかりませんでした。)\n\nAssistant: 申し訳ありません。提供された資料の中には、佐藤先生が担当する授業の情報は見つかりませんでした。\n検索ワードを変えてもう一度お試しください。\n\n---\n\n{system_instruction}",
    "variables": ["system_instruction"]
    }


def load_prompt_template(template_name: str, prompts_dir: str = "prompts") -> Optional[Dict]:
    """
    JSONファイルからプロンプトテンプレートを読み込む。
    ファイルがない場合は、'default'であればハードコードされたデフォルトを返す。
    """
    # 1. ファイルからの読み込みを試みる
    path = Path(prompts_dir) / f"{template_name}.json"
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load template {path}: {e}")
            return None

    # 2. ファイルがない場合、defaultなら内部定義を返す
    if template_name == "default":
        return get_default_template()
    
    return None


def list_available_prompts(prompts_dir: str = "prompts") -> List[str]:
    """利用可能なプロンプトテンプレートの一覧を返す"""
    p = Path(prompts_dir)
    files = []
    if p.exists():
        files = [f.stem for f in p.glob("*.json")]
    
    # defaultは常に使えるようにする
    if "default" not in files:
        files.insert(0, "default")
    return files


def render_prompt_template(template_dict: Dict, **kwargs) -> str:
    """テンプレートに変数を埋め込んで文字列にする"""
    if not template_dict:
        return ""
    
    template_str = template_dict.get("system_template", "")
    
    # テンプレート内の変数 {xxx} に対して、kwargsに値がなければ空文字を入れる安全策
    # (Pythonのformatは足りないとエラーになるため)
    # 簡易的に system_instruction だけは最低限保証する
    if "system_instruction" not in kwargs:
        kwargs["system_instruction"] = ""
        
    try:
        return template_str.format(**kwargs)
    except KeyError:
        # 他の変数が足りない場合はそのまま返すか、エラー回避
        return template_str


def get_langchain_prompt_template(template_dict: Dict) -> Optional[PromptTemplate]:
    """LangChainのPromptTemplateオブジェクトを作成して返す"""
    if not LANGCHAIN_PROMPTS_AVAILABLE or not template_dict:
        return None
    
    template_str = template_dict.get("system_template", "")
    input_variables = template_dict.get("variables", [])
    
    try:
        return PromptTemplate(
            input_variables=input_variables,
            template=template_str
        )
    except Exception:
        return None


def format_system_instruction(template_name: str, system_input: str = "", prompts_dir: str = "prompts") -> str:
    """
    テンプレートを読み込み、ユーザー入力の追加指示(system_input)を埋め込んだ
    最終的なシステムプロンプト文字列を返す。
    """
    # テンプレート読み込み
    template = load_prompt_template(template_name, prompts_dir)
    
    # テンプレートが見つからない場合のフォールバック
    if not template:
        if template_name == "default":
             template = get_default_template()
        else:
             # default以外でファイルもない場合は、入力をそのまま返す
             return system_input

    # 変数の埋め込み処理
    # ユーザーがコマンドライン引数 --system で指定した内容は {system_instruction} に入る
    kwargs = {
        "system_instruction": system_input
    }
    
    return render_prompt_template(template, **kwargs)