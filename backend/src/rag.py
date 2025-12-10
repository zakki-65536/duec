import json
import os
import numpy as np

# scikit-learnがインストールされていない場合のエラーハンドリング
# try:
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
SKLEARN_AVAILABLE = True
# except ImportError:
#     SKLEARN_AVAILABLE = False

def load_course_db(db_path: str):
    """
    JSONファイルを読み込んでリストとして返します。
    失敗した場合は None を返します。
    """
    if not os.path.exists(db_path):
        print(f"Error: Database file not found at {db_path}")
        return None
    
    try:
        with open(db_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # データ形式の簡易チェック
            if isinstance(data, list) and len(data) > 0 and 'text' in data[0]:
                return data
            else:
                print("Error: JSON content must be a list of dicts with a 'text' field.")
                return None
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return None

def prepare_tfidf_index(db):
    """
    ドキュメントのリストからTF-IDFのインデックス（ベクトル化器と行列）を作成します。
    日本語対応のため、単語単位ではなく「文字単位(char)」でベクトル化します。
    """
    if not SKLEARN_AVAILABLE:
        print("scikit-learn not found. TF-IDF indexing skipped.")
        return None

    # タイトルと本文を結合して検索対象にする
    corpus = [f"{doc.get('title', '')} {doc.get('text', '')}" for doc in db]

    # analyzer='char', ngram_range=(2, 3) にすることで
    # 日本語の形態素解析を行わずに、2~3文字の並びで類似度を計算します。
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 3))
    tfidf_matrix = vectorizer.fit_transform(corpus)

    return {
        "vectorizer": vectorizer,
        "matrix": tfidf_matrix,
        "db": db
    }

def retrieve_tfidf(query: str, index_data, k: int = 3):
    """
    事前計算したTF-IDFインデックスを使ってコサイン類似度検索を行います。
    """
    if not index_data or not SKLEARN_AVAILABLE:
        return []

    vectorizer = index_data["vectorizer"]
    matrix = index_data["matrix"]
    db = index_data["db"]

    # クエリをベクトル化
    query_vec = vectorizer.transform([query])

    # コサイン類似度を計算 (クエリ vs 全ドキュメント)
    similarities = cosine_similarity(query_vec, matrix).flatten()

    # 類似度が高い順にインデックスを取得
    # argsortは昇順なので、[::-1]で降順にし、上位k個を取得
    top_indices = similarities.argsort()[::-1][:k]

    # 類似度が0より大きいものだけを抽出
    results = []
    for idx in top_indices:
        if similarities[idx] > 0:
            results.append(db[idx])
    
    return results

def retrieve(query: str, db, k: int = 3):
    """
    簡易的な検索（キーワード一致検索）。
    scikit-learnがない場合や、単純なマッチングを使いたい場合のフォールバック用。
    """
    scored_docs = []
    query_terms = query.split() # 空白区切りで単語化（日本語だとあまり効かないが簡易版として）

    for doc in db:
        score = 0
        content = f"{doc.get('title', '')} {doc.get('text', '')}"
        
        # クエリそのものが含まれているか
        if query in content:
            score += 10
        
        # クエリ内の各単語が含まれているか（簡易カウント）
        for term in query_terms:
            if term in content:
                score += 1

        if score > 0:
            scored_docs.append((score, doc))

    # スコア順にソートして上位k件を返す
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in scored_docs[:k]]

def build_prompt_with_rag(system: str, history: list, user_input: str, retrieved_docs: list) -> str:
    """
    検索結果(retrieved_docs)をプロンプトに組み込みます。
    """
    pieces = []
    
    # システムプロンプトがあれば追加
    if system:
        pieces.append(f"{system}\n")

    # 検索されたコンテキスト情報を追加
    if retrieved_docs:
        pieces.append("以下はユーザーの質問に関連する参考情報です。回答の参考にしてください。\n")
        pieces.append("--- 参考情報 ---")
        for i, doc in enumerate(retrieved_docs, 1):
            title = doc.get('title', 'No Title')
            text = doc.get('text', '')
            pieces.append(f"【文書{i}: {title}】\n{text}\n")
        pieces.append("----------------\n")
    else:
        pieces.append("(関連する参考情報は見つかりませんでした。)\n")

    # 会話履歴を追加
    pieces.append("Conversation:\n")
    for role, text in history:
        if role == "user":
            pieces.append(f"User: {text}\n")
        else:
            pieces.append(f"Assistant: {text}\n")
    
    # 現在の入力を追加
    pieces.append(f"User: {user_input}\nAssistant:")
    
    return "\n".join(pieces)