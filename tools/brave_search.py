import os
import requests
from dotenv import load_dotenv
from langchain_core.tools import BaseTool, ArgsSchema
from typing import Type, Dict, List, Optional
from pydantic import BaseModel, Field

load_dotenv()

BRAVE_SEARCH_API_KEY = os.getenv("BRAVE_SEARCH_API_KEY")

class BraveSearchInput(BaseModel):
    query: str = Field(description="検索するクエリ")

class BraveSearchTool(BaseTool):
    name: str = "web_search"
    description: str = "最新情報や一般的な知識、特定のトピックについて調べる必要がある場合に使用します。検索結果はタイトル、URL、スニペットのリストとして返されます。"
    args_schema: Optional[ArgsSchema] = BraveSearchInput

    def _run(self, query: str) -> List[Dict]:
        if not BRAVE_SEARCH_API_KEY:
            return [{"error": "BRAVE_SEARCH_API_KEYが設定されていません。"}]

        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": BRAVE_SEARCH_API_KEY
        }
        params = {
            "q": query
        }
        try:
            response = requests.get("https://api.search.brave.com/res/v1/web/search", headers=headers, params=params)
            response.raise_for_status() # HTTPエラーがあれば例外を発生させる
            data = response.json()
            
            results = []
            if "web" in data and "results" in data["web"]:
                for item in data["web"]["results"]:
                    results.append({
                        "title": item.get("title"),
                        "url": item.get("url"),
                        "snippet": item.get("description") # Brave Searchでは"description"がスニペットに相当
                    })
            return results
        except requests.exceptions.RequestException as e:
            return [{"error": f"Brave Search APIリクエストエラー: {e}"}]
        except Exception as e:
            return [{"error": f"検索処理中に予期せぬエラーが発生しました: {e}"}]

    async def _arun(self, query: str) -> List[Dict]:
        # 非同期処理が必要な場合はここに実装。requestsは同期ライブラリなので、
        # aiohttpなど非同期HTTPクライアントを使用するか、ThreadPoolExecutorでラップする必要がある。
        # 今回はシンプルに同期_runを呼び出す。
        return self._run(query)

if __name__ == "__main__":
    # テストコード
    # .envファイルにBRAVE_SEARCH_API_KEYを設定してください
    # 例: BRAVE_SEARCH_API_KEY="YOUR_BRAVE_SEARCH_API_KEY"
    tool = BraveSearchTool()
    test_query = "今日の天気 東京"
    print(f"Searching for: {test_query}")
    results = tool.run(test_query)
    if results and "error" in results[0]:
        print(f"Error: {results[0]['error']}")
    else:
        for r in results[:3]: # 上位3件を表示
            print(f"Title: {r.get('title')}")
            print(f"URL: {r.get('url')}")
            print(f"Snippet: {r.get('snippet')}")
            print("-" * 20)
