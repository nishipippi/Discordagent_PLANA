import os
import requests
from langchain.tools import tool
from dotenv import load_dotenv

# Brave Search APIキーを環境変数から取得
# .envファイルのフルパスを取得 (bot.pyからの相対パスを考慮)
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(dotenv_path=dotenv_path, override=True)

BRAVE_SEARCH_API_KEY = os.getenv("BRAVE_SEARCH_API_KEY")
BRAVE_SEARCH_API_URL = "https://api.search.brave.com/res/v1/web/search"

@tool
def brave_search(query: str) -> str:
    """
    Brave Search APIを使用してウェブ検索を実行します。
    検索クエリを入力として受け取り、検索結果の要約を返します。
    """
    if not BRAVE_SEARCH_API_KEY:
        return "Brave Search APIキーが設定されていません。検索を実行できません。"

    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": BRAVE_SEARCH_API_KEY
    }
    params = {
        "q": query,
        "count": 5 # 取得する検索結果の数
    }

    try:
        response = requests.get(BRAVE_SEARCH_API_URL, headers=headers, params=params)
        response.raise_for_status() # HTTPエラーが発生した場合に例外を発生させる

        data = response.json()
        
        # 検索結果からタイトルとURL、スニペットを抽出して整形
        results = []
        if "web" in data and "results" in data["web"]:
            for item in data["web"]["results"]:
                title = item.get("title")
                url = item.get("url")
                snippet = item.get("description") # または snippet
                if title and url and snippet:
                    results.append(f"タイトル: {title}\nURL: {url}\nスニペット: {snippet}\n")
        
        if results:
            return "検索結果:\n" + "\n---\n".join(results)
        else:
            return "指定されたクエリの検索結果が見つかりませんでした。"

    except requests.exceptions.RequestException as e:
        return f"Brave Search APIへのリクエスト中にエラーが発生しました: {e}"
    except Exception as e:
        return f"検索結果の処理中に予期せぬエラーが発生しました: {e}"

if __name__ == "__main__":
    # テストコード
    # .envファイルにBRAVE_SEARCH_API_KEYを設定してから実行してください
    # このスクリプトを直接実行する場合、.envファイルのパスはbot.pyからの相対パスとは異なるため調整
    # tools/brave_search.py から見て、親ディレクトリの親ディレクトリに .env がある
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))
    
    # 環境変数が読み込まれているか確認
    if not os.getenv("BRAVE_SEARCH_API_KEY"):
        print("Warning: BRAVE_SEARCH_API_KEY is not set in .env for testing.")
        print("Please set it to run the test code.")
    else:
        test_query = "今日の天気 東京"
        print(f"Searching for: {test_query}")
        result = brave_search(test_query)
        print(result)

        test_query_no_result = "存在しない検索クエリabcdefg12345"
        print(f"\nSearching for: {test_query_no_result}")
        result_no_result = brave_search(test_query_no_result)
        print(result_no_result)
