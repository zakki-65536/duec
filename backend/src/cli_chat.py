#!/usr/bin/env python3
"""
Simple CLI chat that uses langchain's Ollama LLM if available,
otherwise falls back to calling the `ollama` CLI.
"""
import argparse
import subprocess
import sys
import shutil
from rag import load_course_db_from_csv, load_course_db, retrieve, build_prompt_with_rag, prepare_tfidf_index, retrieve_tfidf
from prompt_manager import load_prompt_template, list_available_prompts, format_system_instruction

# Try to import langchain's Ollama wrapper if available
try:
    from langchain_ollama import OllamaLLM # type: ignore
    LANGCHAIN_AVAILABLE = True
except Exception:
    OllamaLLM = None  # type: ignore
    LANGCHAIN_AVAILABLE = False


def run_with_langchain(model: str, prompt: str) -> str:
    """Use langchain's Ollama wrapper to generate text."""
    if not LANGCHAIN_AVAILABLE:
        raise RuntimeError("langchain Ollama not available")
    llm = OllamaLLM(model=model)
    return llm.invoke(prompt)


def run_with_ollama_cli(model: str, prompt: str) -> str:
    """Fallback: call the `ollama` CLI."""
    if shutil.which("ollama") is None:
        raise RuntimeError("ollama CLI not found in PATH; please install ollama or use langchain.")
    
    cmd_with_prompt = ["ollama", "run", model, "--prompt", prompt]
    try:
        res = subprocess.run(cmd_with_prompt, capture_output=True, text=True, check=True, encoding="utf-8",errors="strict")
        return res.stdout.strip()
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or "").lower()
        stdout = (e.stdout or "").lower()
        combined = "\\n".join([stdout, stderr]).strip()
        
        if "unknown flag" in combined or "unrecognized option" in combined or "--prompt" in combined:
            cmd_stdin = ["ollama", "run", model]
            try:
                res2 = subprocess.run(cmd_stdin, input=prompt, capture_output=True, text=True, check=True,encoding="utf-8",errors="strict")
                return res2.stdout.strip()
            except subprocess.CalledProcessError as e2:
                out2 = e2.stdout.strip() if e2.stdout else e2.stderr.strip()
                raise RuntimeError(f"ollama CLI failed (fallback stdin): {out2}")
        raise RuntimeError(f"ollama CLI failed: {combined}")


def generate(model: str, prompt: str, prefer_langchain: bool = True) -> str:
    """Generate model output trying langchain first, falling back to CLI."""
    if prefer_langchain and LANGCHAIN_AVAILABLE:
        try:
            return run_with_langchain(model, prompt)
        except Exception:
            pass
    return run_with_ollama_cli(model, prompt)


def build_prompt(system: str, history: list, user_input: str) -> str:
    """Simple prompt assembly."""
    pieces = []
    if system:
        pieces.append(f"{system}\n\n")
    pieces.append("Conversation:\n")
    for role, text in history:
        if role == "user":
            pieces.append(f"User: {text}\n")
        else:
            pieces.append(f"Assistant: {text}\n")
    pieces.append(f"User: {user_input}\nAssistant:")
    return "".join(pieces)


# --- 変更点: ChatSessionクラスの追加 ---
class ChatSession:
    """
    対話の状態(履歴, 設定, RAGインデックス)を保持し、
    入力を受け取って応答を返す機能を提供するクラス
    """
    def __init__(self, args):
        self.args = args
        self.history = []  # list of (role, text)
        self.prefer_langchain = not args.no_langchain
        
        # System instruction preparation
        self.system_instruction = format_system_instruction(args.prompt_template, args.system)
        print(f"Using prompt template: {args.prompt_template}")

        # RAG Initialization
        self.course_db_wrapper = None
        self.rag_index = None

        if args.rag:
            print(f"Loading RAG DB from: {args.rag_db}")
            
            # 拡張子で分岐
            if args.rag_db.endswith('.csv'):
                print("Detected CSV format for RAG DB.")
                self.course_db_wrapper = load_course_db_from_csv(args.rag_db)
            else:
                self.course_db_wrapper = load_course_db(args.rag_db)
            
            if self.course_db_wrapper is None:
                print(f"RAG DB not found or invalid at {args.rag_db}; continuing without RAG.")
                self.args.rag = False
            else:
                if args.rag_method == "tfidf":
                    self.rag_index = prepare_tfidf_index(self.course_db_wrapper)
                    if self.rag_index is None:
                        print("TF-IDF index could not be prepared. Falling back to simple retrieval.")
                        self.args.rag_method = "simple"

    # cli_chat.py の ChatSession クラス内を変更

    def chat(self, user_input: str, history: list = None) -> str:
        """
        ユーザー入力を受け取り、AIの応答を返す関数
        
        Args:
            user_input (str): ユーザーの入力
            history (list, optional): 外部から履歴を渡す場合に使用。
                                    形式: [("user", "こんにちは"), ("assistant", "はい")]
                                    Noneの場合はクラス内部の履歴(self.history)を使用します。
        """
        # 1. 履歴の決定: 外部指定があればそれを使い、なければ内部履歴を使う
        if history is None:
            active_history = self.history
        else:
            active_history = history

        # Build prompt from system + last N history turns
        trimmed_history = active_history[-self.args.history_size:]

        user_messages = [msg for role, msg in trimmed_history if role == "user"]
        history_user_text = ", ".join(user_messages)
        combined_query = f"{history_user_text}\n{user_input}" if history_user_text else user_input
        
        # RAG or Normal Prompt Building
        if self.args.rag:
            if self.args.rag_method == "tfidf" and self.rag_index is not None:
                retrieved = retrieve_tfidf(combined_query, self.rag_index, k=self.args.rag_k)
            else:
                retrieved = retrieve(combined_query, self.course_db_wrapper, k=self.args.rag_k)
            
            prompt = build_prompt_with_rag(self.system_instruction, trimmed_history, user_input, retrieved)
        else:
            prompt = build_prompt(self.system_instruction, trimmed_history, user_input)

        # Generation
        try:
            reply = generate(self.args.model, prompt, prefer_langchain=self.prefer_langchain)
        except Exception as e:
            return f"Error generating response: {str(e)}"

        # 2. 履歴の更新: 
        # 内部履歴を使っている場合のみ、自動で追記する。
        # 外部履歴(history)を渡した場合は、呼び出し元で管理してもらうためここでは追記しない。
        if history is None:
            self.history.append(("user", user_input))
            self.history.append(("assistant", reply))
        
        return reply
# ------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Simple CLI chat using local ollama model")
    parser.add_argument("--model", default="dsasai/llama3-elyza-jp-8b", help="Ollama model name")
    parser.add_argument("--system", default="", help="Optional system prompt")
    available_prompts = " | ".join(list_available_prompts()) or "default"
    parser.add_argument("--prompt-template", default="default", help=f"Available: {available_prompts}")
    parser.add_argument("--history-size", type=int, default=6, help="History size")
    parser.add_argument("--no-langchain", action="store_true", help="Disable langchain")
    parser.add_argument("--rag", action="store_true", help="Enable RAG")
    parser.add_argument("--rag-db", default="../database/syllabus_インテリ.json", help="Path to JSON DB")
    parser.add_argument("--rag-k", type=int, default=3, help="RAG retrieval count")
    parser.add_argument("--rag-method", choices=["tfidf", "simple"], default="tfidf", help="RAG method")
    args = parser.parse_args()

    # Log environment status
    if not args.no_langchain and LANGCHAIN_AVAILABLE:
        print("Using langchain Ollama wrapper.")
    elif not args.no_langchain and not LANGCHAIN_AVAILABLE:
        print("langchain Ollama wrapper not available; will use ollama CLI fallback.")
    else:
        print("Forcing ollama CLI usage (no-langchain).")
    
    # --- 変更点: クラスのインスタンス化とチャットループ ---
    
    # チャットセッションの初期化 (DBロードなどはここで行われます)
    bot = ChatSession(args)

    print("Enter conversation. Type 'exit' or Ctrl-C to quit.")

    try:
        while True:
            try:
                user_input = input("\n> ")
            except EOFError:
                print("\nGoodbye.")
                break
            
            if not user_input.strip():
                continue
            if user_input.strip().lower() in ("exit", "quit"):
                print("Goodbye.")
                break

            # クラスのメソッド（関数）を呼び出して応答を取得
            reply = bot.chat(user_input)
            
            print("\nAssistant:", reply)

    except KeyboardInterrupt:
        print("\nInterrupted. Bye.")
    # ------------------------------------------------


if __name__ == "__main__":
    main()