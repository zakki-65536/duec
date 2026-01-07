import json
import os
import re

# scikit-learnがインストールされていない場合のエラーハンドリング
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

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
    TF-IDFインデックスを作成します。
    引数 db_wrapper は load_course_db の戻り値 (dict) を期待します。
    """
    if not SKLEARN_AVAILABLE:
        print("scikit-learn not found. TF-IDF indexing skipped.")
        return None

    docs = db_wrapper["docs"]
    metadata = db_wrapper.get("metadata", {})

    # 検索対象テキスト
    corpus = [doc.get('text', '') for doc in docs]

    # 日本語対応: 文字単位 n-gram (2~3文字)
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 3))
    tfidf_matrix = vectorizer.fit_transform(corpus)

    return {
        "vectorizer": vectorizer,
        "matrix": tfidf_matrix,
        "docs": docs,
        "metadata": metadata
    }

def retrieve_tfidf(query: str, index_data, k: int = 3):
    """
    メタデータフィルタリング付きの検索を行います。
    1. クエリ内に教授名が含まれていれば、その教授の全科目を優先的に取得します。
    2. それ以外の場合は、TF-IDF類似度で上位 k 件を取得します。
    """
    if not index_data or not SKLEARN_AVAILABLE:
        return []

    vectorizer = index_data["vectorizer"]
    matrix = index_data["matrix"]
    docs = index_data["docs"]
    professors = index_data["metadata"].get("professors", [])

    # --- 戦略1: メタデータフィルタリング（教授名検索） ---
    # クエリ内の空白を除去してマッチング精度を上げる
    normalized_query = query.replace(" ", "").replace("　", "")
    
    matched_professors = []
    for prof in professors:
        # 教授名がクエリに含まれているかチェック
        # (誤検知を防ぐため、教授名がある程度の長さ以上であることを想定、または完全一致推奨だが、ここでは簡易包含判定)
        if prof in normalized_query and len(prof) >= 2:
            matched_professors.append(prof)
    
    # 教授名がヒットした場合の特別処理
    if matched_professors:
        filtered_results = []
        seen_titles = set()
        
        # ヒットした教授の授業をすべて抽出
        for doc in docs:
            doc_prof = doc.get("professor", "").replace(" ", "").replace("　", "")
            # 抽出した教授リストのいずれかに一致すれば採用
            if any(p == doc_prof for p in matched_professors):
                filtered_results.append(doc)
                seen_titles.add(doc["title"])
        
        # もし教授名フィルタだけで十分な数が取れれば、それを返す（kを無視して全件返す）
        # これにより「全ての授業を教えて」に対応可能
        if filtered_results:
            print(f"[Debug] Filtered by professor: {matched_professors} -> {len(filtered_results)} hits")
            return filtered_results

    # --- 戦略2: 通常のTF-IDFベクトル検索 ---
    # フィルタでヒットしなかった、あるいは追加で必要な場合
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, matrix).flatten()
    
    # 類似度順にソート
    top_indices = similarities.argsort()[::-1][:k]

    results = []
    for idx in top_indices:
        if similarities[idx] > 0:
            results.append(docs[idx])
    
    return results

def retrieve(query: str, db_wrapper, k: int = 3):
    """
    scikit-learnがない場合の簡易検索（フォールバック用）。
    こちらも構造変更に合わせて修正。
    """
    # db_wrapperが辞書ならdocsを取り出す、リストならそのまま（互換性）
    if isinstance(db_wrapper, dict):
        docs = db_wrapper.get("docs", [])
    else:
        docs = db_wrapper

    scored_docs = []
    query_terms = query.replace("　", " ").split() 

    for doc in docs:
        score = 0
        content = doc.get('text', '')
        
        # クエリそのものが含まれているか
        if query in content:
            score += 10
        
        for term in query_terms:
            if term in content:
                score += 1

        if score > 0:
            scored_docs.append((score, doc))

    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in scored_docs[:k]]

def build_prompt_with_rag(system: str, history: list, user_input: str, retrieved_docs: list) -> str:
    """
    検索結果(retrieved_docs)をプロンプトに組み込みます。
    """
    pieces = []
    
    if system:
        pieces.append(f"{system}\n")

    if retrieved_docs:
        pieces.append("以下はユーザーの質問に関連する講義シラバス情報です。")
        pieces.append("質問が「全ての授業」などのリスト要求であれば、検索された全ての情報を要約して答えてください。\n")
        pieces.append("--- 参考情報 (Syllabus) ---")
        for i, doc in enumerate(retrieved_docs, 1):
            text = doc.get('text', '')
            pieces.append(f"【情報{i}】\n{text}\n")
        pieces.append("---------------------------\n")
    else:
        pieces.append("(関連する参考情報は見つかりませんでした。)\n")

    pieces.append("Conversation:\n")
    for role, text in history:
        if role == "user":
            pieces.append(f"User: {text}\n")
        else:
            pieces.append(f"Assistant: {text}\n")
    
    pieces.append(f"User: {user_input}\nAssistant:")
    
    return "\n".join(pieces)