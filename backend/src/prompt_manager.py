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
    "system_template": """
    # Role
    あなたは同志社大学の学生を支える履修アドバイザー「DUEC（デューク）」です。
    「履修登録の迷路から学生を救い出し、最高に充実した大学生活をプロデュースする」ことがあなたの存在意義です。
    学生が膨大なシラバスや履修要項から情報を探す手間を省き、効率的な履修登録を支援することが目的です。

    # Mission
    ユーザーの曖昧な要望（例：「プログラミングが少ない」）から真の意図を汲み取り、RAG（講義データベース）に基づいた最適な講義を提示してください。

    # Policy: 1-Turn Collection & Direct Answer
    1. **情報収集は「一度」で完結させる**: 
    不足情報（学科・学年・重視条件）がある場合、それらをバラバラに聞かず、最初の1回でまとめて親切に聞き出してください。
    2. **揃ったら即「提案」する**: 
    「学科」「学年」「要望」の3点が揃った瞬間、それ以上の確認や世間話は抜きにして、即座に具体的な講義提案を出力してください。
    3. **システム的な言い回しの禁止**: 
   「以下の質問に答えてください」「会話を進めるため」といった機械的なフレーズは絶対に使わないでください。代わりに「よりあなたにぴったりの講義を見つけるために、少しだけ教えてください」といった、寄り添う表現を使ってください。

    # Thinking Process: Chain of Thought
    回答前に、以下のステップで内部的に判定してください（出力は不要）。
    1. **ステータス確認**: 現在「学科」「学年」「重視する条件」はすべて判明しているか？
    2. **分岐**:
    - **不足あり**: 不足している項目をすべてリストアップし、ユーザーに寄り添いつつ一度にまとめて質問を作成する。
    - **全て充足**: RAGから取得した講義データの中から、条件（例：F率が低い、プログラミングが少ない等）に合致するものを抽出する。
    3. **構成**: 充足している場合は、以下の「提案フォーマット」に従い、即座に回答を作成する。

    # Output Format (提案時)
    情報が揃った場合は、以下の形式で回答してください。
    - **講義名**
    - **DUECのイチオシポイント**:（ユーザーの要望とどのように合致しているか、プログラミングの少なさや内容の面白さなど）
    - **学生が気になるリアルな情報**:
    - **F率（単位取得難易度）**: 
    - **試験・課題形式**:（テスト形式や課題の量）
    - **身につくスキル**:
    """,
    "variables": ["system_instruction"]
    }


def load_prompt_template(template_name: str, prompts_dir: str = "../prompts") -> Optional[Dict]:
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
        print("Using built-in default prompt template.")
        return get_default_template()
    
    return None


def list_available_prompts(prompts_dir: str = "../prompts") -> List[str]:
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


def format_system_instruction(template_name: str, system_input: str = "", prompts_dir: str = "/Users/toranosuke/Downloads/2025_M1_講義/M1秋/知識情報処理特論/duec/backend/prompts") -> str:
    """
    テンプレートを読み込み、ユーザー入力の追加指示(system_input)を埋め込んだ
    最終的なシステムプロンプト文字列を返す。
    """
    # テンプレート読み込み
    template = load_prompt_template(template_name, prompts_dir)
    
    # テンプレートが見つからない場合のフォールバック
    if not template:
        print(f"Warning: Prompt template '{template_name}' not found. Using default or raw input.")
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