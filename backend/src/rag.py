import json
import os
import re
import csv
import numpy as np
import torch
from FlagEmbedding import BGEM3FlagModel

# 保存先ファイル名
EMBEDDINGS_CACHE_PATH = "syllabus_embeddings.npy"

# デバイス設定
device = "cuda" if torch.cuda.is_available() else "cpu"
# モデルの初期化を遅延させる: prepare_tfidf_index / retrieve_tfidf 実行時に初めてロードする
_model = None

def get_model():
    """遅延初期化された埋め込みモデルを返す。初回呼び出しでロードする。"""
    global _model
    if _model is None:
        print("Initializing BGEM3FlagModel (this may take a long time and download weights)...")
        _model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=(device == "cuda"))
    return _model

# 外部ライブラリのインポート
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

def simple_tokenize(text):
    """
    日本語の簡易トークナイザー。
    正規表現を用いて、漢字、ひらがな、カタカナ、英数字の塊を抽出します。
    """
    # 記号を除去し、意味のある文字列の塊を抽出
    tokens = re.findall(r'[A-Za-z0-9]+|[㐀-䶵一-龠々]+|[ぁ-ん]+|[ァ-ヶー]+', text)
    return tokens

def load_course_db_from_csv(db_path: str):
    """
    シラバス形式のCSVファイルを読み込み、RAG検索用に整形してリストとして返します。
    ヘッダーの余計な空白やBOMを自動で処理し、JSON形式のカラムをパースしてテキスト化します。
    """
    if not os.path.exists(db_path):
        print(f"Error: Database file not found at {db_path}")
        return None
    
    try:
        # utf-8-sig を使うことでBOM(Byte Order Mark)を自動で除去します
        with open(db_path, 'r', encoding='utf-8-sig', errors='replace') as f:
            # csv.DictReaderを使用して辞書形式で読み込む
            reader = csv.DictReader(f)
            
            # ヘッダー（カラム名）から前後の空白を削除して正規化
            if reader.fieldnames:
                reader.fieldnames = [name.strip() for name in reader.fieldnames]
            
            processed_data = []
            known_professors = set()

            for row in reader:
                # 行データの前後の空白を削除
                item = {k: (v.strip() if v else "") for k, v in row.items() if k}
                
                # 「科目名」がない行はスキップ
                if not item.get("科目名"):
                    continue

                # --- メタデータの収集 ---
                prof_name = item.get("教授名", "").strip()
                if prof_name:
                    known_professors.add(prof_name)
                    # 検索揺らぎ対応（スペース除去）
                    known_professors.add(prof_name.replace(" ", "").replace("　", ""))

                # --- テキスト構築 (RAG検索対象) ---
                title = item.get("科目名", "名称不明")
                text_parts = []
                
                text_parts.append(f"【科目名】 {title}")
                if item.get("学科"):
                    text_parts.append(f"【学科】 {item['学科']}")
                if prof_name:
                    text_parts.append(f"【担当教授】 {prof_name}")
                
                basic_info = []
                if item.get("開講学期"): basic_info.append(item['開講学期'])
                if item.get("曜日・時限"): basic_info.append(item['曜日・時限'])
                if item.get("単位数"): basic_info.append(f"{item['単位数']}単位")
                if item.get("授業形態"): basic_info.append(item['授業形態'])
                if basic_info:
                    text_parts.append(f"【基本情報】 {' / '.join(basic_info)}")
                
                if item.get("概要"):
                    text_parts.append(f"【概要】\n{item['概要']}")
                if item.get("到達目標"):
                    text_parts.append(f"【到達目標】\n{item['到達目標']}")

                # 授業計画の処理（JSON文字列であればパースして箇条書きにする）
                plan_raw = item.get("授業計画", "")
                if plan_raw and plan_raw.startswith("["):
                    try:
                        plans = json.loads(plan_raw)
                        text_parts.append("【授業計画】")
                        for plan in plans:
                            text_parts.append(f"  第{plan.get('授業回', '?')}回: {plan.get('内容', '')}")
                    except:
                        text_parts.append(f"【授業計画】\n{plan_raw}")
                elif plan_raw:
                    text_parts.append(f"【授業計画】\n{plan_raw}")

                # 成績評価基準の処理
                eval_raw = item.get("成績評価基準", "")
                if eval_raw and eval_raw.startswith("["):
                    try:
                        evals = json.loads(eval_raw)
                        text_parts.append("【成績評価基準】")
                        for crit in evals:
                            text_parts.append(f"  - {crit.get('項目', '')} ({crit.get('割合', '')}): {crit.get('詳細', '')}")
                    except:
                        text_parts.append(f"【成績評価基準】\n{eval_raw}")
                elif eval_raw:
                    text_parts.append(f"【成績評価基準】\n{eval_raw}")

                # 成績評価結果の処理 (成績分布など)
                res_raw = item.get("成績評価結果", "")
                if res_raw and res_raw.startswith("{"):
                    try:
                        res = json.loads(res_raw)
                        res_summary = ", ".join([f"{k}: {v}" for k, v in res.items()])
                        text_parts.append(f"【成績実績】 {res_summary}")
                    except:
                        pass

                combined_text = "\n".join(text_parts)

                processed_data.append({
                    "title": title,
                    "professor": prof_name,
                    "semester": item.get("開講学期", ""),
                    "text": combined_text,
                    "raw": item
                })

            if len(processed_data) == 0:
                print("Error: No valid course data found in CSV. Check column names or encoding.")
                return None
            
            return {
                "docs": processed_data,
                "metadata": {
                    "professors": list(known_professors)
                }
            }

    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

def load_course_db(db_path: str):
    """
    シラバス形式のJSONファイルを読み込み、RAG検索用に整形してリストとして返します。
    同時に、メタデータ検索用のインデックス（教授名リストなど）も作成します。
    """
    if not os.path.exists(db_path):
        print(f"Error: Database file not found at {db_path}")
        return None
    
    try:
        with open(db_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            
            if not isinstance(raw_data, list):
                print("Error: JSON root must be a list.")
                return None

            processed_data = []
            
            # メタデータ抽出用セット
            known_professors = set()

            for item in raw_data:
                if "科目名" not in item:
                    continue

                # --- メタデータの収集 ---
                prof_name = item.get("教授名", "").strip()
                # 空白を除去した名前も保存（検索揺らぎ対応: "佐藤 健" -> "佐藤健"）
                if prof_name:
                    known_professors.add(prof_name)
                    known_professors.add(prof_name.replace(" ", "").replace("　", ""))

                # --- テキスト構築 (既存ロジック維持) ---
                title = item.get("科目名", "名称不明")
                text_parts = []
                
                text_parts.append(f"【科目名】 {title}")
                if "教授名" in item:
                    text_parts.append(f"【担当教授】 {item['教授名']}")
                if "開講学期" in item:
                    text_parts.append(f"【開講】 {item['開講学期']} {item.get('曜日・時限', '')}")
                
                if "概要" in item:
                    text_parts.append(f"【概要】\n{item['概要']}")
                if "到達目標" in item:
                    text_parts.append(f"【到達目標】\n{item['到達目標']}")

                if "授業計画" in item and isinstance(item["授業計画"], list):
                    text_parts.append("【授業計画】")
                    for plan in item["授業計画"]:
                        kai = plan.get("授業回", "")
                        naiyou = plan.get("内容", "")
                        text_parts.append(f"  第{kai}回: {naiyou}")

                if "成績評価基準" in item and isinstance(item["成績評価基準"], list):
                    text_parts.append("【成績評価基準】")
                    for crit in item["成績評価基準"]:
                        komoku = crit.get("項目", "")
                        wariai = crit.get("割合", "")
                        shosai = crit.get("詳細", "")
                        text_parts.append(f"  - {komoku} ({wariai}): {shosai}")

                combined_text = "\n".join(text_parts)

                processed_data.append({
                    "title": title,
                    "professor": prof_name, # フィルタリング用に保持
                    "semester": item.get("開講学期", ""),
                    "text": combined_text,
                    "raw": item
                })

            if len(processed_data) == 0:
                print("Error: No valid course data found in JSON.")
                return None
            
            # DB自体にメタデータ情報を付与して返す
            return {
                "docs": processed_data,
                "metadata": {
                    "professors": list(known_professors)
                }
            }

    except Exception as e:
        print(f"Error loading JSON: {e}")
        return None

def prepare_tfidf_index(db_wrapper):
    """
    埋め込みベクトルを作成、またはキャッシュから読み込みます。
    """
    if not db_wrapper or "docs" not in db_wrapper: return None

    docs = db_wrapper["docs"]
    
    # --- キャッシュの確認 ---
    if os.path.exists(EMBEDDINGS_CACHE_PATH):
        print(f"キャッシュファイル {EMBEDDINGS_CACHE_PATH} を読み込んでいます...")
        embeddings = np.load(EMBEDDINGS_CACHE_PATH)
        
        # 件数が一致するか念のためチェック
        if len(embeddings) == len(docs):
            print("キャッシュからの読み込みに成功しました。")
            return {"embeddings": embeddings, "docs": docs}
        else:
            print("CSVとキャッシュの件数が一致しません。再作成します。")

    # --- キャッシュがない、または古い場合は新規作成 ---
    corpus_texts = [doc['text'] for doc in docs]
    print(f"BGE-M3を使用して新規インデックスを作成中... (対象: {len(corpus_texts)}件)")
    
    # ベクトル化の実行（モデルは遅延初期化）
    embeddings = get_model().encode(corpus_texts, batch_size=12, max_length=512)['dense_vecs']
    
    # ベクトルを保存
    np.save(EMBEDDINGS_CACHE_PATH, embeddings)
    print(f"埋め込みベクトルを {EMBEDDINGS_CACHE_PATH} に保存しました。")
    
    return {
        "embeddings": embeddings,
        "docs": docs
    }

def retrieve_tfidf(query: str, index_data, k: int = 3):
    """
    ハイブリッド検索ロジック (BGE-M3ベクトル検索 + 条件フィルタ)
    """
    if not index_data: return []

    docs = index_data["docs"]
    embeddings = index_data["embeddings"]

    # 1. ユーザーの質問をベクトル化（モデルは遅延初期化）
    query_result = get_model().encode([query])['dense_vecs']
    query_vec = query_result[0]
    
    # 2. コサイン類似度の計算
    scores = np.dot(embeddings, query_vec) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_vec))
    
    # 3. 条件による調整（教授名フィルタ、時限ブースト）
    exclude_prof = None
    if any(x in query for x in ["以外", "でない", "除いて"]):
        prof_match = re.search(r'([一-龠]{2,4})教授', query)
        if prof_match:
            exclude_prof = prof_match.group(1)

    # スコア順にソートして候補を抽出
    sorted_indices = np.argsort(scores)[::-1]
    results = []
    
    for idx in sorted_indices:
        doc = docs[idx]
        # 否定条件の適用
        if exclude_prof and exclude_prof in doc['professor']:
            continue
            
        # 「1限」などの時限キーワードの一致による加点
        time_keywords = ["1限", "１限", "2限", "２限", "3限", "３限", "4限", "４限", "5限", "５限"]
        for tk in time_keywords:
            if tk in query and tk[0] in doc['period']:
                scores[idx] += 0.1 # スコアを底上げ
        
        results.append(doc)
        if len(results) >= k:
            break
            
    return results

def build_prompt_with_rag(system: str, history: list, user_input: str, retrieved_docs: list) -> str:
    """既存のプロンプト形式を維持"""
    pieces = [f"{system}\n"] if system else []
    if retrieved_docs:
        pieces.append("【関連する講義情報】")
        for i, doc in enumerate(retrieved_docs, 1):
            pieces.append(f"講義{i}: {doc['text'][:300]}") # 長すぎないよう制限
        pieces.append("-" * 20 + "\n")
    
    pieces.append("これまでの対話:")
    for role, text in history:
        pieces.append(f"{role.capitalize()}: {text}")
    pieces.append(f"User: {user_input}\nAssistant:")
    return "\n".join(pieces)

def retrieve(query: str, db_wrapper, k: int = 3):
    """フォールバック用"""
    return retrieve_tfidf(query, db_wrapper, k)
