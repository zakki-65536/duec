#!/usr/bin/env python3
"""
Simple CLI chat that uses langchain's Ollama LLM if available,
otherwise falls back to calling the `ollama` CLI (`ollama run <model> --prompt`).

Usage example:
  python3 cli_chat.py --model dsasai/llama3-elyza-jp-8b --system "あなたは親切なアシスタントです。"

Minimal, single-file implementation.
"""
import argparse
import subprocess
import sys
import shutil

# Try to import langchain's Ollama wrapper if available
try:
    from langchain.llms import Ollama as LangchainOllama  # type: ignore
    LANGCHAIN_AVAILABLE = True
except Exception:
    LangchainOllama = None  # type: ignore
    LANGCHAIN_AVAILABLE = False


def run_with_langchain(model: str, prompt: str) -> str:
    """Use langchain's Ollama wrapper to generate text."""
    if not LANGCHAIN_AVAILABLE:
        raise RuntimeError("langchain Ollama not available")
    llm = LangchainOllama(model=model)
    # The LLM object is callable in langchain; call returns string
    return llm(prompt)


def run_with_ollama_cli(model: str, prompt: str) -> str:
    """Fallback: call the `ollama` CLI. Requires `ollama` in PATH.
    Uses `ollama run <model> --prompt "..."` which returns the model output.
    """
    if shutil.which("ollama") is None:
        raise RuntimeError("ollama CLI not found in PATH; please install ollama or use langchain.")
    # Some versions of the ollama CLI do not accept a `--prompt` flag.
    # First try the `--prompt` invocation for compatibility, then
    # fall back to passing the prompt via stdin if the flag is unsupported.
    cmd_with_prompt = ["ollama", "run", model, "--prompt", prompt]
    try:
        res = subprocess.run(cmd_with_prompt, capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or "").lower()
        stdout = (e.stdout or "").lower()
        combined = "\\n".join([stdout, stderr]).strip()
        # Detect unknown flag or unsupported option and try stdin fallback
        if "unknown flag" in combined or "unrecognized option" in combined or "--prompt" in combined:
            # Fallback: send prompt via stdin to `ollama run <model>`
            cmd_stdin = ["ollama", "run", model]
            try:
                res2 = subprocess.run(cmd_stdin, input=prompt, capture_output=True, text=True, check=True)
                return res2.stdout.strip()
            except subprocess.CalledProcessError as e2:
                out2 = e2.stdout.strip() if e2.stdout else e2.stderr.strip()
                raise RuntimeError(f"ollama CLI failed (fallback stdin): {out2}")
        # otherwise raise with combined output for debugging
        raise RuntimeError(f"ollama CLI failed: {combined}")


def generate(model: str, prompt: str, prefer_langchain: bool = True) -> str:
    """Generate model output trying langchain first (if requested), then falling
    back to the CLI method.
    """
    if prefer_langchain and LANGCHAIN_AVAILABLE:
        try:
            return run_with_langchain(model, prompt)
        except Exception:
            # fall through to CLI
            pass
    # CLI fallback
    return run_with_ollama_cli(model, prompt)


def build_prompt(system: str, history: list, user_input: str) -> str:
    """Simple prompt assembly: include optional system prompt and a short history.
    History is list of tuples (role, text) where role is 'user' or 'assistant'.
    Output ends with 'Assistant:' so model completes the assistant reply.
    """
    pieces = []
    if system:
        pieces.append(f"System: {system}\n\n")
    pieces.append("Conversation:\n")
    for role, text in history:
        if role == "user":
            pieces.append(f"User: {text}\n")
        else:
            pieces.append(f"Assistant: {text}\n")
    pieces.append(f"User: {user_input}\nAssistant:")
    return "".join(pieces)


def main():
    parser = argparse.ArgumentParser(description="Simple CLI chat using local ollama model (langchain + fallback)")
    parser.add_argument("--model", default="dsasai/llama3-elyza-jp-8b", help="Ollama model name (default dsasai/llama3-elyza-jp-8b)")
    parser.add_argument("--system", default="", help="Optional system prompt / instruction for the assistant")
    parser.add_argument("--history-size", type=int, default=6, help="How many previous messages (counted as turns) to include in prompt")
    parser.add_argument("--no-langchain", action="store_true", help="Do not try to use langchain even if installed; use ollama CLI directly")
    args = parser.parse_args()

    prefer_langchain = not args.no_langchain
    if prefer_langchain and LANGCHAIN_AVAILABLE:
        print("Using langchain Ollama wrapper.")
    elif prefer_langchain and not LANGCHAIN_AVAILABLE:
        print("langchain Ollama wrapper not available; will use ollama CLI fallback.")
    else:
        print("Forcing ollama CLI usage (no-langchain).")

    print("Enter conversation. Type 'exit' or Ctrl-C to quit.")
    history = []  # list of (role, text)

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

            # Build prompt from system + last N history turns
            trimmed = history[-args.history_size:]
            prompt = build_prompt(args.system, trimmed, user_input)

            try:
                reply = generate(args.model, prompt, prefer_langchain=prefer_langchain)
            except Exception as e:
                print("\nError generating response:", str(e))
                continue

            print("\nAssistant:", reply)

            # Append to history
            history.append(("user", user_input))
            history.append(("assistant", reply))

    except KeyboardInterrupt:
        print("\nInterrupted. Bye.")


if __name__ == "__main__":
    main()
