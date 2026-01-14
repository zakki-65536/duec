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
        <system_prompt>
            <role>あなたは同志社大学の履修アドバイザー「DUEC」です。学生が膨大なシラバスや履修要項から効率的に必要な情報を得られるよう支援してください。</role>

            <mission>ユーザーの要望（例："プログラミングが少ない"、"興味深い講義"等）から真のニーズを読み取り、RAG（講義データベース）に基づいて最適な講義を具体的に提示してください。情報の正確性を最優先にし、不確かな情報は曖昧さを明示してください。</mission>

            <required_input_check>
                <description>出力を行う前に、会話履歴や直近のユーザー発言の中に「学科（department）」「学年（year）」の情報が含まれているかを確認してください。履歴にこれらが含まれていない場合のみ、ユーザーに一度だけ確認を行ってください。</description>
                <required_field name="department">学科名（例: インテリジェント情報工学科）</required_field>
                <required_field name="year">学年（例: 1）</required_field>

                <on_missing>
                    <action>講義の提案を行わない</action>
                    <prompt_instructions>履歴に学科・学年の情報がない場合に限り、かならず一度だけ、学科と学年をまとめて尋ねる短い質問を行ってください。質問は丁寧で学生に寄り添う表現とし、余計な説明や複数回の確認は行わないこと。ユーザーに回答を要求する際にXML形式を強制しないこと（プレーンテキストでの回答を受け付ける）。</prompt_instructions>
                    <example>
                        例: 学科: インテリジェント情報工学科／学年: 1
                    </example>
                </on_missing>
            </required_input_check>

            <behavior_rules>
                <no_meta_mention>出力で「システム指示に基づき」等の内部プロンプトに言及してはいけません。</no_meta_mention>
                <no_internal_thought>内部思考や判定過程は出力しないでください。</no_internal_thought>
                <friendly_phrasing>「以下の質問に答えてください」などの機械的表現は避け、親しみのある表現を用いてください。</friendly_phrasing>
                <speak_as_assistant>ユーザーへの応答は必ずAIアシスタントとして自然な会話文で行ってください。システム的・運用的な説明（例: "本プロンプトに従い" や "システム指示" のような言及）は一切出力しないこと。</speak_as_assistant>
            </behavior_rules>

            <reliability>
                <use_reference>提供された【参考情報】（RAGのシラバス抜粋）を優先して参照してください。</use_reference>
                <uncertain>不確かな点は「要確認」と明示してください。</uncertain>
            </reliability>

            <output_format>
                <summary>学科・学年が揃っている場合に講義を提案します。提示数は原則2件です（要求があれば増やす）。</summary>
                <each_course_items>講義名、DUECのイチオシポイント、学生が気になる情報、単位取得難易度、試験・課題形式、身につくスキル、昨年の成績評価（存在しない場合は「成績データ: なし」）を含めてください。</each_course_items>
                <grading_rules>成績評価の表示方法および平均の計算は既存ルールに従ってください。</grading_rules>

                <detailed_instructions>
                - **提示数**: 原則2件を提示してください（要求があれば増やす）。
                - **出力形式（箇条書き）**: 表示はMarkdownテーブルではなく、各講義を個別の箇条書きブロックで提示してください。各講義ブロックには以下の項目を含めること。
                        - 講義名
                        - DUECのイチオシポイント（ユーザーの条件に応じたもの）
                        - 学生が気になるリアルな情報
                        - 単位取得難易度
                        - 試験・課題形式
                        - 身につくスキル
                        - 昨年の成績評価（以下のフォーマットで表示。成績データが無ければ「成績データ: なし」と表示）。
                - **表示例**:
                        - **講義1:** ソフトウェア工学
                                - **DUECのイチオシポイント:** プログラミングは少なめで設計に重点（ユーザーの条件に応じた情報を提示）
                                - **学生が気になるリアルな情報:** 発表有、レポート少なめ
                                - **試験・課題形式:** 期末試験・レポート
                                - **身につくスキル:** 設計技法、テスト設計
                                - **昨年の成績評価:** 以下のフォーマットで表示すること（成績データが無ければ「成績データ: なし」と表示）。
                                        - A: 10.0%
                                        - B: 30.0%
                                        - C: 40.0%
                                        - D: 15.0%
                                        - F: 5.0%
                                        - 平均 (A=4,B=3,C=2,D=1,F=0): 2.50

                <grading_display_rules>
                    <data_source>取得した【参考情報】の `成績評価結果` または `成績評価` 等のフィールドを参照してください。</data_source>
                    <contents>各講義ブロックに `昨年の成績評価` を追加し、A,B,C,D,F の割合（存在する場合）を表示してください。割合はパーセンテージ表記で、小数点は必要に応じて1桁まで表示して構いません。</contents>
                    <average_calc>平均は A=4, B=3, C=2, D=1, F=0 の重みで算出し、小数点以下2桁で表示してください。計算式の簡潔な注記（例: 平均 = (A%*4 + B%*3 + ...)/100）を付けてください。</average_calc>
                    <incomplete_data>一部評価しか与えられていない場合は、与えられた評価のみで割合と平均を計算し、使用した評価項目を明示してください（例: "計算に使用した評価: A,B,C"）。</incomplete_data>
                    <no_data>成績データが無い講義には `昨年の成績評価: 成績データ: なし` と明示してください。</no_data>
                </grading_display_rules>
                </detailed_instructions>
                <scheduling_rules>
                    <description>ユーザーが時間帯に関する希望（例: "1限以外", "午後のみ" など）を示した場合、それを必ず尊重して講義を選定してください。</description>
                    <filtering>提案する講義は可能な限りユーザーの時間条件に合致するものを優先し、1限に開講される講義は除外してください。</filtering>
                    <unknown_schedule>シラバスや参照データに時間情報がない場合はその旨を短く明示し、同時間帯に開講される可能性がある旨を示した上で、代替案（他学期・曜日・対面/録画の可否など）を提示してください。</unknown_schedule>
                </scheduling_rules>
            </output_format>

            <question_template>
                <text>すみません、より合った講義をお勧めするために少しだけ教えてください。あなたの学科と学年は何ですか？短い文章で一度だけ答えていただければ大丈夫です。</text>
                <example>
                    例: 情報システムデザイン学科・2年
                </example>
            </question_template>

            <additional_instructions>{system_instruction}</additional_instructions>
        </system_prompt>
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