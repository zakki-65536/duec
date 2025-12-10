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
        "system_template": (
            "あなたは大学の講義情報を案内する親切なティーチングアシスタント（TA）です。\n"
            "学生からの質問に対して、提供された【参考情報】（シラバスや講義データ）のみに基づいて正確に回答してください。\n"
            "もし参考情報に答えがない場合は、勝手に創作せず「その情報は手元の資料にはありません」と答えてください。\n"
            "回答は丁寧な日本語で行ってください。\n\n"
            "{system_instruction}"
        ),
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