import os
import warnings
from typing import List, Dict, Optional
import anthropic
import google.generativeai as genai
from openai import OpenAI
import time
import dotenv
from dataclasses import dataclass
from enum import Enum

@dataclass
class AIResponse:
    content: str
    error: Optional[str] = None

class AIModel(Enum):
    GPT = "gpt"
    CLAUDE = "claude"
    GEMINI = "gemini"

class MAGISystem:
    def __init__(self):
        dotenv.load_dotenv()
        self._initialize_clients()
        self._initialize_roles()

    def _initialize_clients(self) -> None:
        """APIクライアントの初期化"""
        try:
            # OpenAI
            self.openai_client = OpenAI(api_key=self._get_env_var("OPENAI_API_KEY"))
            self.openai_model = self._get_env_var("OPENAI_MODEL", default="gpt-4")
            
            # Anthropic
            self.anthropic_client = anthropic.Anthropic(api_key=self._get_env_var("ANTHROPIC_API_KEY"))
            self.anthropic_model = self._get_env_var("ANTHROPIC_MODEL", default="claude-3-opus-20240229")
            
            # Google
            genai.configure(api_key=self._get_env_var("GOOGLE_API_KEY"))
            self.gemini_model = self._get_env_var("GEMINI_MODEL", default="gemini-pro")
            self.gemini = genai.GenerativeModel(self.gemini_model)
        except Exception as e:
            raise RuntimeError(f"クライアントの初期化に失敗しました: {str(e)}")

    def _get_env_var(self, key: str, default: Optional[str] = None) -> str:
        """環境変数の取得と検証"""
        value = os.getenv(key, default)
        if value is None:
            raise ValueError(f"環境変数 {key} が設定されていません")
        return value

    def _initialize_roles(self) -> None:
        """AI役割の定義"""
        self.roles = {
            AIModel.GPT: "あなたは論理的な分析と批判的思を得意とする議論者です。",
            AIModel.CLAUDE: "あなたは幅広い知識と創造的な発想を持つ議論者です。",
            AIModel.GEMINI: "あなたは実践的でバランスの取れた視点を持つ議論者です。"
        }

    def get_ai_response(self, model: AIModel, prompt: str) -> AIResponse:
        """統一されたインターフェースでAIレスポンスを取得"""
        try:
            if model == AIModel.GPT:
                return self._get_gpt_response(prompt)
            elif model == AIModel.CLAUDE:
                return self._get_claude_response(prompt)
            elif model == AIModel.GEMINI:
                return self._get_gemini_response(prompt)
        except Exception as e:
            return AIResponse(content="", error=str(e))

    def _get_gpt_response(self, prompt: str) -> AIResponse:
        """GPT-4からの応答を取得"""
        response = self.openai_client.chat.completions.create(
            model=self.openai_model,
            messages=[
                {"role": "system", "content": self.roles[AIModel.GPT]},
                {"role": "user", "content": prompt}
            ],
            stream=True
        )
        
        full_response = ""
        print(f"\n🔵 {self.openai_model}の視点:")
        for chunk in response:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                full_response += content
        print()
        return AIResponse(content=full_response)

    def _get_claude_response(self, prompt: str) -> AIResponse:
        """Claudeからの応答を取得"""
        response = self.anthropic_client.messages.create(
            model=self.anthropic_model,
            max_tokens=1000,
            system=self.roles[AIModel.CLAUDE],
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )
        
        full_response = ""
        print(f"\n🟣 {self.anthropic_model}の視点:")
        for message in response:
            if message.type == "content_block_delta":
                content = message.delta.text
                print(content, end="", flush=True)
                full_response += content
        print()
        return AIResponse(content=full_response)

    def _get_gemini_response(self, prompt: str) -> AIResponse:
        """Geminiからの応答を取得"""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                response = self.gemini.generate_content(
                    f"{self.roles[AIModel.GEMINI]}\n\n{prompt}",
                    safety_settings={"HARM_CATEGORY_HARASSMENT": "BLOCK_NONE"},
                    generation_config={
                        "temperature": 0.7,
                        "top_p": 0.8,
                        "top_k": 40
                    },
                    stream=True
                )
            
            full_response = ""
            print(f"\n🟢 {self.gemini_model}の視点:")
            for chunk in response:
                if chunk.text:
                    print(chunk.text, end="", flush=True)
                    full_response += chunk.text
            print()
            return AIResponse(content=full_response)
            
        except Exception as e:
            return AIResponse(content="", error=f"Geminiでエラーが発生: {str(e)}")

    def _generate_conclusion(self, discussion_history: List[Dict]) -> Dict:
        """最終結論を生成"""
        try:
            conclusion_prompt = f"""
            これまでの議論全体を踏まえて、最終的な結論を導き出してください。
            以下が議論の履歴です：{discussion_history}
            """
            conclusion = self.get_ai_response(AIModel.CLAUDE, conclusion_prompt)
            return {
                "discussion_history": discussion_history,
                "final_conclusion": conclusion.content
            }
        except Exception as e:
            print(f"\n### ⚠️ 結論の導出時にエラーが発生しました: {str(e)}")
            return {
                "discussion_history": discussion_history,
                "final_conclusion": "結論の導出に失敗しました。"
            }

    def facilitate_discussion(self, topic: str, rounds: int = 3) -> Dict:
        """AIによる議論を進行"""
        discussion_history = []
        current_prompt = topic

        for round_num in range(rounds):
            print(f"\n## 🤖 ラウンド {round_num + 1}")
            
            try:
                responses = self._get_round_responses(current_prompt)
                discussion_history.append({
                    "round": round_num + 1,
                    **responses
                })
                current_prompt = self._prepare_next_prompt(responses)
                print("\n---")
                time.sleep(2)

            except Exception as e:
                print(f"\n### ⚠️ エラーが発生しました: {str(e)}")
                continue

        # 最終結論の生成
        conclusion_prompt = f"""
        これまでの議論を総括し、以下の点を含めた独自の結論を導き出してください：
        1. これまでの議論で見落とされていた視点
        2. 各AIの意見の中で特に重要だと思われるポイント
        3. 今後の展望や提言

        ※これまでの発言を単に要約するのではなく、新しい視点で分析してください。
        ※以下が議論の履歴です：
        {discussion_history}
        """
        
        try:
            conclusion = self.get_ai_response(AIModel.CLAUDE, conclusion_prompt)
            return {
                "discussion_history": discussion_history,
                "final_conclusion": conclusion.content
            }
        except Exception as e:
            return {
                "discussion_history": discussion_history,
                "final_conclusion": f"結論の導出に失敗しました: {str(e)}"
            }

    def _get_round_responses(self, prompt: str) -> Dict[str, str]:
        """1ラウンドの応答を取得"""
        responses = {}
        
        # GPTの応答
        gpt_response = self.get_ai_response(AIModel.GPT, prompt)
        responses["gpt"] = gpt_response.content

        # Claudeの応答
        claude_prompt = f"{prompt}\n\nGPT-4の意見: {gpt_response.content}"
        claude_response = self.get_ai_response(AIModel.CLAUDE, claude_prompt)
        responses["claude"] = claude_response.content

        # Geminiの応答
        gemini_prompt = f"{claude_prompt}\nClaudeの意見: {claude_response.content}"
        gemini_response = self.get_ai_response(AIModel.GEMINI, gemini_prompt)
        responses["gemini"] = gemini_response.content

        return responses

    def _prepare_next_prompt(self, responses: Dict[str, str]) -> str:
        """次のラウンドのプロンプトを生成"""
        return f"""
        これまでの議論を踏まえて、さらなる検討点や新しい視点を提示してください。

        議論の経緯:
        GPT: {responses["gpt"]}
        Claude: {responses["claude"]}
        Gemini: {responses["gemini"]}
        """

def display_ascii_art():
    """ASCIIアートを表示"""
    ascii_art = """
    ███╗   ███╗ ██████╗  ██████╗ ██╗      ███████╗██╗   ██╗███████╗
    ████╗ ████║██╔══██╗██╔════╝ ██║      ██╔════╝╚██╗ ██╔╝██╔════╝
    ██╔████╔██║██║  ██║██║  ███╗██║█████╗███████╗ ╚████╔╝ ███████╗
    ██║╚██╔╝██║██║  ██║██║   ██║██║╚════╝╚════██║  ╚██╔╝  ╚════██║
    ██║ ╚═╝ ██║╚██████╔╝╚██████╔╝██║      ███████║   ██║   ███████║
    ╚═╝     ╚═╝ ╚═════╝  ╚═════╝ ╚═╝      ╚══════╝   ╚═╝   ╚══════╝
    ============= MOGI-SYSTEM: Multi-AI Generative Intelligence System =============
    """
    print(ascii_art)

def main():
    """メイン実行関数"""
    try:
        display_ascii_art()
        magi = MAGISystem()

        print("\n💭 議論のテーマを入力してください：")
        user_topic = input().strip()
        
        topic = f"""
        テーマ：「{user_topic}」
        
        このテーマについて、異なる視点から議論を展開し、
        具体的な例を挙げながら検討してください。
        """

        # 議論を開始
        result = magi.facilitate_discussion(topic)

        # 結果の出力
        print("\n=== 最終結論 ===")
        print(result["final_conclusion"])
        
    except Exception as e:
        print(f"\n⚠️ エラーが発生しました: {str(e)}")
        print("プログラムを終了します。")
        return 1

    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)