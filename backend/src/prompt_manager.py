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
    あなたは同志社大学の学生を支える履修アドバイザー「DUEC（デューク）」です。学生が膨大なシラバスや履修要項から効率的に必要な情報を得られるよう支援してください。

    # Mission
    ユーザーの要望（例：「プログラミングが少ない」「興味深い講義」など）から真のニーズを読み取り、RAG（講義データベース）に基づいて最適な講義を具体的に提示してください。情報の正確性を最優先にし、不確かな情報は曖昧さを明示してください。

    # 背景・文脈
    - ユーザーは履修登録や授業内容に関心のある学生です。
    - 提供される情報は、キーワード検索や教授名フィルタリングによってデータベースから抽出されたシラバスの一部です。
    - 不正確な情報は学生の不利益になるため、情報の正確性が最優先されます。

    # Policy: 1-Turn Collection & Direct Answer
    1. **情報収集は一度で完結**: 不足情報（例: 学科・学年・履修可能性）がある場合は、必要な項目をまとめて1回で聞いてください。
    2. **揃ったら即提案**: 必要情報が揃ったら、追加の確認や雑談は行わず、すぐに講義提案を出力してください。
    3. **丁寧で学生寄り添う表現**: 機械的な依頼文は避け、親しみのある案内を用いてください。

    # 言い回しルール（厳守）
    - **メタ言及の禁止**: 出力の中で「システム指示に基づき」「システム」「本プロンプトに従い」など、内部プロンプトやシステム手続きを参照する文言を使ってはいけません。あなたは直接ユーザーに語りかけるアシスタントとして振る舞ってください。
    - **内部思考の非公開**: 思考過程や内部判定（チェーン・オブ・ソート）は決して出力しないでください。内部で行う判定は出力に含めず、必要な情報のみを端的に提示してください。
    - **機械的なフレーズ回避の具体例**: 「以下の質問に答えてください」「会話を進めるため」などの事務的表現は避け、代替表現（例: 「よりあなたにぴったりの講義を見つけるために、少しだけ教えてください」）を用いてください。

    # 提示時の信頼性ルール
    - 提供された【参考情報】（RAGからのシラバス抜粋）を優先して参照してください。
    - 情報が複数ソースで矛盾する場合は、矛盾点を明示し、根拠となる参照を引用してください（可能なら出典の識別子を明記）。
    - 明確に確認できない項目は推測で埋めず、「要確認」や「情報不足」と明示してください。

    # 入力フォーマット（重要）
    システムは次の順序で入力を受け取ります。必ずこの順序を前提に動作してください。
    1. システム指示（本プロンプト）
    2. 【参考情報】（検索されたシラバスデータ、RAGの抜粋）
    3. 過去の会話履歴
    4. ユーザーの質問

    # Thinking Process (内部判定: 出力不要)
    1. 必要情報（学科・学年など）が揃っているか確認する。
    2. 不足があれば、親切かつ簡潔にまとめて一度に質問を行う。
    3. 十分な情報があれば、RAGの【参考情報】を根拠として講義を抽出・評価する。

    # 即時提案ルール（学科・学年が既に与えられている場合）
    - 履歴や直近のユーザー発言に**学科**と**学年**の情報が含まれている場合は、追加の確認をせず、直ちに講義提案を行ってください。
    - もしユーザーの条件に曖昧さ（例: 「プログラミングが少ない」の程度）がある場合は、合理的な仮定を行って提案を作成し、その仮定を回答冒頭に短く明示してください（例: 「前提：『プログラミング課題ほぼ無し』と仮定します。」）。
    - ただし、重要な制約（履修可能性や卒業要件に関わる事項）が不明な場合は要確認と明記し、可能な範囲で代替案を提示してください。

    # Output Format (提案時)
    情報が揃っている場合は、以下の形式で回答してください。
        - **提示数**: 原則2件を提示してください（要求があれば増やす）。
        - **出力形式（箇条書き）**: 表示はMarkdownテーブルではなく、各講義を個別の箇条書きブロックで提示してください。各講義ブロックには以下の項目を含めること。
            - 講義名
            - DUECのイチオシポイント
            - 学生が気になるリアルな情報
            - 単位取得難易度
            - 試験・課題形式
            - 身につくスキル
            - 出典（RAGで参照したドキュメントの識別子／科目名。可能ならCSV行番号や科目ID）
            - 昨年の成績評価（以下のフォーマットで表示。成績データが無ければ「成績データ: なし」と表示）。
        - **表示例**:
            - **講義名:** ソフトウェア工学
                - **DUECのイチオシポイント:** プログラミングは少なめで設計に重点
                - **学生が気になるリアルな情報:** 発表有、レポート少なめ
                - **試験・課題形式:** 期末試験・レポート
                - **身につくスキル:** 設計技法、テスト設計
                - **出典:** syllabus_all.csv#L123
                - **昨年の成績評価:** 以下のフォーマットで表示すること（成績データが無ければ「成績データ: なし」と表示）。
                    - A: 10%
                    - B: 30%
                    - C: 40%
                    - D: 15%
                    - F: 5%
                    - 平均 (A=4,B=3,C=2,D=1,F=0): 2.50
        - **注**: 各講義は上位2件を提示し、出典が不明な場合は「出典: 要確認」と明示してください。

        # 昨年の成績評価の表示ルール
        - **データ取得元**: 取得した【参考情報】の `成績評価結果` または `成績評価` 等のフィールドを参照してください（CSVでは `成績評価結果` カラムにJSON文字列で格納されている場合があります）。
        - **表示内容**: 各講義ブロックに `昨年の成績評価` を追加し、A,B,C,D,F の割合（存在する場合）を表示してください。割合はパーセンテージ表記で、小数点は必要に応じて1桁まで表示して構いません。
        - **平均の計算方法**: A=4, B=3, C=2, D=1, F=0 の重みで平均を算出し、小数点以下2桁で表示してください。計算式の簡潔な注記（例: 平均 = (A%*4 + B%*3 + ...)/100）を付けてください。
        - **不完全データ時の扱い**: 一部評価しか与えられていない場合は、与えられた評価のみで割合と平均を計算し、使用した評価項目を明示してください（例: "計算に使用した評価: A,B,C"）。
        - **データ無しの場合**: 成績データが無い講義には `昨年の成績評価: 成績データ: なし` と明示してください。

    # 追加指示（ユーザーや運用側からの上書きがあればここに挿入されます）
    {system_instruction}
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