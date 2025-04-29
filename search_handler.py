# search_handler.py
# (検索コマンドの処理、Brave Search API連携、URLテキスト抽出)

import asyncio
import re
import httpx
import discord
import json # DSRCレポート生成で使う可能性 (今回はテキスト化だが将来的に構造化データも考慮)
from typing import List, Dict, Any, Optional, Tuple, Literal, Union # Unionを追加

# aiocache をインポート
from aiocache import cached, Cache
# グローバル設定は削除

import config
import bot_constants
import llm_manager
import cache_manager # キャッシュ保存のため追加
import discord_ui # Thinking message, ボタン生成用
from llm_provider import ERROR_TYPE_UNKNOWN # エラータイプ定数

# command_handler モジュール全体ではなく、handle_mention 関数を直接インポート
# from command_handler import handle_mention # <- handle_mention 関数を直接インポート
# ただし、assess_and_respond_to_mention 関数は command_handler モジュール全体を参照しているため、
# モジュール全体のインポートを維持しつつ、Pylanceエラーが出ないようにエイリアスを使う方法を試みます。
# もしエイリアスでダメなら、handle_mention を直接インポートして呼び出し箇所を変更します。
import command_handler as ch # <- エイリアスを使用してインポート


# --- Brave Search API Call ---
# @cached デコレータを追加
# エラー修正: キャッシュタイプとTTLをキーワード引数として指定
@cached(cache=Cache.MEMORY, ttl=300, key_builder=lambda f, query: query)
async def call_brave_search_api(query: str) -> Optional[List[Dict[str, Any]]]:
    """Brave Search APIを呼び出す (キャッシュ対応)"""
    # --- 関数の中身は変更なし ---
    if not config.BRAVE_SEARCH_API_KEY:
        print("Error: BRAVE_SEARCH_API_KEY is not set.")
        return None

    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": config.BRAVE_SEARCH_API_KEY,
        "User-Agent": "PlanaBot/1.0 (Discord Bot)" # 適切なUser-Agent
    }
    params = {
        "q": query,
        "count": config.MAX_SEARCH_RESULTS,
        "search_filter": "web",
        # "country": "jp", # 必要に応じて
        # "search_lang": "ja" # 必要に応じて
    }

    async with httpx.AsyncClient() as client:
        try:
            # キャッシュされていない場合のみAPIコールが発生
            print(f"Calling Brave Search API for query: '{query}'...")
            response = await client.get(config.BRAVE_SEARCH_API_URL, headers=headers, params=params, timeout=20)
            response.raise_for_status()
            data = response.json()
            results = data.get('web', {}).get('results', [])
            print(f"Brave Search API call successful. Found {len(results)} web results.")
            return results
        except httpx.HTTPStatusError as e:
            print(f"HTTP error occurred while calling Brave Search API: {e.response.status_code}")
            # print(f"Response body: {e.response.text}") # デバッグ用
            return None
        except httpx.RequestError as e:
            print(f"An error occurred while requesting Brave Search API: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during Brave Search API call: {e}")
            return None
        finally:
            # API呼び出しごとに必ず待機 (キャッシュヒット時も待機してしまうが、大きな問題はない)
            # 頻繁なキャッシュヒットが予想される場合は、キャッシュヒット時には待機しない工夫も可能
            await asyncio.sleep(config.BRAVE_API_DELAY)


# --- URL Content Extraction ---
async def extract_text_from_url(url: str) -> Optional[str]:
    """URLからテキストコンテンツを抽出する (簡易版)"""
    if not url or not (url.startswith("http://") or url.startswith("https://")):
        print(f"Invalid URL skipped: {url}")
        return None

    print(f"Attempting to extract text from URL: {url}")
    try:
        # HEADリクエストでContent-Typeを確認
        async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
            try:
                 head_response = await client.head(url)
                 head_response.raise_for_status()
                 content_type = head_response.headers.get('Content-Type', '').lower()
                 # text/html, text/plain, application/json 以外をスキップ
                 if not any(ct in content_type for ct in ['text/html', 'text/plain', 'application/json']):
                     print(f"Skipping non-HTML/text/JSON content type: {content_type} for {url}")
                     return None
            except httpx.HTTPStatusError as e:
                 print(f"HEAD request failed for {url}: {e.response.status_code}")
                 return None # HEAD失敗はアクセス不能とみなす
            except httpx.RequestError as e:
                 print(f"HEAD request failed for {url}: {e}")
                 return None

        # GETリクエストでコンテンツ取得
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
             response = await client.get(url)
             response.raise_for_status()
             content_type = response.headers.get('Content-Type', '').lower() # 再度取得

             # Content-Typeに応じて処理を分岐
             if 'application/json' in content_type:
                  try:
                      # JSONとしてパースし、テキスト要素を結合（簡易的）
                      json_data = response.json()
                      text_parts = []
                      def extract_text_from_json(data):
                          if isinstance(data, dict):
                              for key, value in data.items():
                                  if isinstance(value, str):
                                      text_parts.append(value)
                                  else:
                                      extract_text_from_json(value)
                          elif isinstance(data, list):
                              for item in data:
                                  extract_text_from_json(item)
                          elif isinstance(data, str):
                              text_parts.append(data)
                      extract_text_from_json(json_data)
                      text_content = ' '.join(text_parts).strip()
                  except Exception as json_e:
                       print(f"Failed to parse JSON or extract text from {url}: {json_e}. Falling back to raw text.")
                       text_content = response.text # パース失敗時は生テキスト
             elif 'text/html' in content_type:
                 html_content = response.text # httpxがエンコーディングを推定
                 # HTMLタグ除去 (簡易版)
                 text_content = re.sub(r'<script.*?>.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
                 text_content = re.sub(r'<style.*?>.*?</style>', '', text_content, flags=re.DOTALL | re.IGNORECASE)
                 text_content = re.sub(r'<!--.*?-->', '', text_content, flags=re.DOTALL)
                 text_content = re.sub(r'>\s*<', '> <', text_content) # タグ間の空白
                 text_content = re.sub(r'<.*?>', '', text_content) # 全タグ除去
             else: # text/plain など
                  text_content = response.text

             text_content = re.sub(r'\s+', ' ', text_content).strip() # 連続空白をまとめる

             # 長さチェックと切り詰め
             if len(text_content) > config.MAX_CONTENT_LENGTH_PER_URL:
                 text_content = text_content[:config.MAX_CONTENT_LENGTH_PER_URL] + "..."
                 print(f"Truncated content for {url} to {config.MAX_CONTENT_LENGTH_PER_URL} characters.")

             # 短すぎるコンテンツはスキップ
             if len(text_content) < config.SEARCH_MIN_CONTENT_LENGTH:
                 print(f"Content too short ({len(text_content)} chars) for {url}. Skipping.")
                 return None

             print(f"Successfully extracted text from {url} ({len(text_content)} chars).")
             return text_content

    except httpx.HTTPStatusError as e:
        print(f"HTTP error occurred while fetching URL {url}: {e.response.status_code}")
        return None
    except httpx.RequestError as e:
        print(f"An error occurred while requesting URL {url}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while processing URL {url}: {e}")
        import traceback
        traceback.print_exc()
        return None


# --- Search Necessity Assessment ---
async def should_perform_search(question: str) -> bool:
    """与えられた質問に対して検索が必要か Lowload モデルで判断する"""
    llm_handler = llm_manager.get_current_provider()
    if not llm_handler:
        print("Warning: LLM handler not available for search necessity assessment.")
        return False # LLM利用不可なら検索できない/判断不可

    lowload_model_name = llm_handler.get_model_name('lowload')
    if not lowload_model_name:
        print(f"Warning: Lowload model unavailable ({llm_manager.get_current_provider_name()}) for search necessity assessment.")
        return False # 低負荷モデルがなければ判断不可 -> 検索しない

    try:
        assessment_prompt = config.SEARCH_NECESSITY_ASSESSMENT_PROMPT.format(question=question)
        # Lowload モデルを使用
        # search_handler.py での generate_lowload_response にはタイムアウトはまだ渡さない (llm_manager側で設定)
        assessment_response_raw = await llm_manager.generate_lowload_response(assessment_prompt)
        # ここで低負荷LLMコールが発生するが、assess_and_respond_to_mention のカウンターでは追跡しない
        assessment_response = str(assessment_response_raw).strip().lower() if assessment_response_raw else ""

        print(f"Search necessity assessment for '{question[:50]}...': Response='{assessment_response}'")
        # 応答が "必要" と完全一致する場合のみ True
        # 大文字小文字を区別しないように lower() してから比較
        return assessment_response == "必要"

    except Exception as e:
        print(f"Error during search necessity assessment: {e}")
        import traceback
        traceback.print_exc()
        return False # エラー時は安全側に倒して検索しない


# --- Mention Response with Search Assessment ---
async def assess_and_respond_to_mention(message: discord.Message, question_text: str):
    """
    メンション応答時に検索が必要か判断し、
    必要なら !src 相当の検索を実行、不要なら command_handler.handle_mention を呼び出す
    """
    # メッセージオブジェクトがあるため、Thinkingメッセージのチャンネルは明示的に渡せる
    await discord_ui.update_thinking_message(message.channel, "…検索が必要か判断中...")

    needs_search = await should_perform_search(question_text) # LLMコール発生の可能性あり

    if needs_search:
        print("Search deemed necessary by LLM. Performing simple search (!src equivalent).")
        # perform_search=True は不要だが、コードの意図を明確にするために残す
        # handle_search_command に処理を移譲し、Thinking Message は引き継がれる
        await handle_search_command(message, 'src', question_text, triggered_by_assessment=True)
    else:
        print("Search deemed unnecessary by LLM. Proceeding with standard mention response.")
        await discord_ui.update_thinking_message(message.channel, "…検索不要と判断、応答準備中...")
        # Thinking Message を削除してから通常のメンション応答を呼び出す
        await discord_ui.delete_thinking_message()
        # 通常のメンション応答 (検索なし)
        # message.guild.me が None でないことを確認してから渡す
        if message.guild and message.guild.me:
             await ch.handle_mention(message, message.guild.me, question_text=question_text, perform_search=False)
        else:
             print("Error: Cannot get bot user info (message.guild.me) in assess_and_respond_to_mention.")
             await message.reply(bot_constants.ERROR_MSG_INTERNAL + " (Bot user info not found)", mention_author=False)


# --- Deep Search (!dsrc) Core Logic ---

# 戻り値にLLMコール数を追加 -> Optional[Tuple[List[str], int]]
async def generate_dsrc_plan(question: str) -> Optional[Tuple[List[str], int]]:
    """!dsrc のための調査計画を生成する"""
    llm_calls = 0
    llm_handler = llm_manager.get_current_provider()
    primary_model_name = llm_manager.get_active_model_name('primary')
    if not llm_handler or not primary_model_name:
        print("Error: Primary model unavailable for DSRC plan generation.")
        return None # エラー時はNoneだけ返す (コール数は不明)

    plan_prompt = config.DSRC_PLAN_GENERATION_PROMPT.format(
        question=question, max_steps=config.DSRC_MAX_PLAN_STEPS
    )
    try:
        # generate_response には gemini_provider.py でデフォルトのタイムアウトが設定されている
        _used_model, plan_response_raw = await llm_manager.generate_response(
            content_parts=[{'text': plan_prompt}], chat_history=None, deep_cache_summary=None
        )
        llm_calls += 1 # LLMコールをカウント
        plan_response = str(plan_response_raw).strip() if plan_response_raw else ""

        # LLMからの応答がエラーメッセージかどうか判定
        if llm_manager.is_error_message(plan_response):
            print(f"DSRC Plan generation failed. Response: {plan_response}")
            # エラーメッセージ文字列を返すのではなく、Noneを返すことで失敗を示す
            return None

        plan_steps_raw = [line.strip() for line in plan_response.splitlines() if line.strip()]
        plan_steps = [re.sub(r"^\s*\d+\.\s*", "", step) for step in plan_steps_raw] # 先頭の空白と番号を除去
        plan_steps = [step for step in plan_steps if step] # 空のステップを除去

        if not plan_steps:
             print("DSRC Plan generation resulted in empty steps.")
             return None # エラー時はNoneだけ返す

        print(f"DSRC Plan Generated ({len(plan_steps)} steps):")
        for i, step in enumerate(plan_steps): print(f"  {i+1}. {step}")
        return plan_steps[:config.DSRC_MAX_PLAN_STEPS], llm_calls # 成功時はプランとコール数を返す

    except Exception as e:
        print(f"Error during DSRC plan generation: {e}")
        import traceback
        traceback.print_exc()
        return None # 例外時もNoneだけ返す

# 戻り値にLLMコール数を追加 -> Tuple[str, Optional[str], int]
async def assess_dsrc_step_results(question: str, step_description: str, search_results_text: str) -> Tuple[str, Optional[str], int]:
    """!dsrc の特定のステップの結果を評価する"""
    llm_calls = 0
    llm_handler = llm_manager.get_current_provider()
    primary_model_name = llm_manager.get_active_model_name('primary')
    if not llm_handler or not primary_model_name:
        print("Error: Primary model unavailable for DSRC step assessment.")
        return "ERROR", "Primary model unavailable.", llm_calls

    assessment_prompt = config.DSRC_STEP_ASSESSMENT_PROMPT.format(
        question=question,
        step_description=step_description,
        search_results_text=search_results_text
    )
    try:
        # generate_response には gemini_provider.py でデフォルトのタイムアウトが設定されている
        _used_model, assessment_response_raw = await llm_manager.generate_response(
            content_parts=[{'text': assessment_prompt}], chat_history=None, deep_cache_summary=None
        )
        llm_calls += 1 # LLMコールをカウント
        assessment_response = str(assessment_response_raw).strip() if assessment_response_raw else ""

        # LLMからの応答がエラーメッセージかどうか判定
        if llm_manager.is_error_message(assessment_response):
             print(f"DSRC Step assessment failed. Response: {assessment_response}")
             return "ERROR", f"Assessment failed: {assessment_response}", llm_calls


        # 大文字小文字を区別しないように upper() してから比較
        if assessment_response.upper() == 'COMPLETE':
            return "COMPLETE", None, llm_calls
        elif assessment_response.upper().startswith('INCOMPLETE:'):
            missing_info = assessment_response.split(':', 1)[1].strip() if ':' in assessment_response else "詳細不明"
            return "INCOMPLETE", missing_info, llm_calls
        else:
            # 予期せぬ形式 -> 不完全とみなし、応答内容を不足情報とする
            print(f"Warning: Unexpected DSRC step assessment format: '{assessment_response}'. Treating as INCOMPLETE.")
            return "INCOMPLETE", assessment_response, llm_calls

    except Exception as e:
        print(f"Error during DSRC step assessment: {e}")
        import traceback
        traceback.print_exc()
        return "ERROR", f"Exception during assessment: {e}", llm_calls

# 戻り値にLLMとBraveのコール数を追加 -> Tuple[Dict[str, str], List[Dict[str, Any]], int, int]
async def execute_dsrc_step(
    question: str,
    step_description: str,
    step_index: int,
    all_results_so_far: Dict[str, str]
    ) -> Tuple[Dict[str, str], List[Dict[str, Any]], int, int]: # llm_calls, brave_calls を返す
    """!dsrc の1ステップを実行 (最大N回の検索・評価サイクル)"""
    step_llm_calls = 0
    step_brave_calls = 0
    llm_handler = llm_manager.get_current_provider()
    primary_model_name = llm_manager.get_active_model_name('primary')
    if not llm_handler or not primary_model_name:
        print(f"Error executing DSRC Step {step_index+1}: Primary model unavailable.")
        # assessments にエラー情報を追加して返す
        assessments = [{"step": step_index + 1, "status": "ERROR", "reason": "Primary model unavailable", "queries": [], "results": {}}]
        return {}, assessments, step_llm_calls, step_brave_calls

    step_results: Dict[str, str] = {} # このステップで新たに見つかった結果 (URL -> text)
    step_assessments: List[Dict[str, Any]] = [] # このステップの評価履歴
    used_queries_for_step: List[str] = [] # このステップで使ったクエリのリスト
    missing_info: Optional[str] = None # 前回のイテレーションで不足していた情報

    for iteration in range(config.DSRC_MAX_ITERATIONS_PER_STEP):
        iteration_label = f"ステップ {step_index+1} ({iteration+1}/{config.DSRC_MAX_ITERATIONS_PER_STEP}回目)"
        # discord_ui.update_thinking_message を呼ぶ際に channel が必須でないように discord.utils.MISSING を渡す
        await discord_ui.update_thinking_message(discord.utils.MISSING, f"…考え中... ({iteration_label} クエリ生成中)")

        # 1. 検索クエリ生成
        query_gen_prompt = config.DSRC_STEP_QUERY_GENERATION_PROMPT.format(
            question=question,
            step_description=step_description,
            used_queries_for_step=", ".join(used_queries_for_step) or "なし",
            missing_info=missing_info if missing_info else "特に指定なし" # Noneの場合はデフォルト文字列
        )
        current_iteration_queries = [] # このイテレーションで使うクエリ
        try:
            # generate_response には gemini_provider.py でデフォルトのタイムアウトが設定されている
            _used_model_q, query_response_raw = await llm_manager.generate_response(
                 content_parts=[{'text': query_gen_prompt}], chat_history=None, deep_cache_summary=None
            )
            step_llm_calls += 1 # LLMコールをカウント
            query_response_text = str(query_response_raw).strip() if query_response_raw else ""
            if llm_manager.is_error_message(query_response_text): # LLMからの応答がエラーメッセージかどうか判定
                 print(f"[{iteration_label}] Query generation failed. Response: {query_response_text}")
                 # クエリ生成失敗はステップ続行不可とみなし、エラーとして終了
                 step_assessments.append({"step": step_index + 1, "iteration": iteration + 1, "status": "ERROR", "reason": f"Query generation failed: {query_response_text}", "queries": [], "results": {}})
                 return step_results, step_assessments, step_llm_calls, step_brave_calls # 失敗時は現在のカウントで終了

            queries_raw = query_response_text.replace('\n', ',')
            current_iteration_queries = [q.strip().strip('"') for q in queries_raw.split(',') if q.strip()][:3]
            current_iteration_queries = [q for q in current_iteration_queries if q]

            if not current_iteration_queries:
                 print(f"[{iteration_label}] Generated empty query list. Proceeding to assessment with existing results.")
                 # クエリが生成されなくても、既存の結果で評価フェーズに進む
                 pass # 次のステップへ
            else:
                 # 新しいクエリのみ used_queries_for_step に追加
                 new_queries = [q for q in current_iteration_queries if q not in used_queries_for_step]
                 if new_queries: used_queries_for_step.extend(new_queries)
                 print(f"[{iteration_label}] Generated queries: {current_iteration_queries}")


        except Exception as e:
            print(f"[{iteration_label}] Error during query generation: {e}")
            import traceback
            traceback.print_exc()
            step_assessments.append({"step": step_index + 1, "iteration": iteration + 1, "status": "ERROR", "reason": f"Query generation exception: {e}", "queries": [], "results": {}})
            return step_results, step_assessments, step_llm_calls, step_brave_calls # 失敗時は現在のカウントで終了


        # 2. Brave Search & 内容取得
        current_iteration_extracted: Dict[str, str] = {} # このイテレーションで取得した新しい結果
        if current_iteration_queries: # クエリがある場合のみ検索
             search_results_api: List[Dict[str, Any]] = []
             for query in current_iteration_queries:
                  await discord_ui.update_thinking_message(discord.utils.MISSING, f"…考え中... ({iteration_label} 検索中: `{query[:30]}...`)")
                  # --- call_brave_search_api 呼び出し前にキャッシュ確認 ---
                  # cache_key = query # key_builderと同じロジック
                  # is_cached = await search_cache.exists(cache_key) # キャッシュ存在確認
                  # if not is_cached:
                  #     step_brave_calls += 1 # キャッシュがなければカウント (APIが呼ばれるはず)
                  # --- シンプルにするため、呼び出しごとにカウントする（キャッシュヒット含むと注記） ---
                  step_brave_calls += 1 # Braveコールをカウント
                  results = await call_brave_search_api(query)
                  if results: search_results_api.extend(results)
                 # call_brave_search_api 内で遅延

             unique_urls = list(dict.fromkeys([r['url'] for r in search_results_api if 'url' in r]))
             # このステップの今回のイテレーションでまだ取得していないURL、かつ全ステップでもまだ取得していないURL
             urls_to_fetch = [url for url in unique_urls if url not in step_results and url not in all_results_so_far]

             if urls_to_fetch:
                  await discord_ui.update_thinking_message(discord.utils.MISSING, f"…考え中... ({iteration_label} ページ内容取得中 {len(urls_to_fetch)}件)")
                  fetch_tasks = [extract_text_from_url(url) for url in urls_to_fetch]
                  extracted_contents_list = await asyncio.gather(*fetch_tasks)
                  for url, content in zip(urls_to_fetch, extracted_contents_list):
                      if content:
                          step_results[url] = content # このステップの結果に追加
                          current_iteration_extracted[url] = content # このイテレーションで取得したもの
                  print(f"[{iteration_label}] Extracted content from {len(current_iteration_extracted)}/{len(urls_to_fetch)} new URLs.")
             else:
                 print(f"[{iteration_label}] No new unique URLs to fetch in this iteration.")

        # 3. 評価
        await discord_ui.update_thinking_message(discord.utils.MISSING, f"…考え中... ({iteration_label} 結果評価中)")
        # このステップで集めた全結果（過去のイテレーション含む）で評価
        combined_step_results_text = "\n\n".join(
            f"--- Content from {url} ---\n{text}\n--- End of {url} ---"
            for url, text in step_results.items() # このステップの全結果
        )
        # 評価用に長さを切り詰める (configの値を使用)
        if len(combined_step_results_text) > config.MAX_TOTAL_SEARCH_CONTENT_LENGTH:
            combined_step_results_text = combined_step_results_text[:config.MAX_TOTAL_SEARCH_CONTENT_LENGTH] + "\n\n... (truncated for assessment)"

        # assess_dsrc_step_results を呼び出し、返り値から LLM コール数を取得
        status, assessment_detail, assessment_llm_calls = await assess_dsrc_step_results(question, step_description, combined_step_results_text)
        step_llm_calls += assessment_llm_calls # ステップ全体のLLMコール数に加算

        # 評価結果を記録
        step_assessments.append({
            "step": step_index + 1,
            "iteration": iteration + 1,
            "status": status,
            "reason": assessment_detail,
            "queries": current_iteration_queries, # このイテレーションで使ったクエリ
            "results": current_iteration_extracted, # このイテレーションで取得した新しい結果
            "brave_calls": len(current_iteration_queries) # このイテレーションでのBraveコール数 (クエリ数に等しい)
        })
        print(f"[{iteration_label}] Assessment: {status} - {assessment_detail}")

        if status == "COMPLETE":
            print(f"Step {step_index+1} completed.")
            return step_results, step_assessments, step_llm_calls, step_brave_calls # 完了時は現在のカウントで終了
        elif status == "ERROR":
             print(f"Error during assessment in Step {step_index+1}. Stopping step.")
             return step_results, step_assessments, step_llm_calls, step_brave_calls # エラー時は現在のカウントで終了
        elif status == "INCOMPLETE":
             missing_info = assessment_detail # 次のイテレーションのために不足情報を更新
             if iteration == config.DSRC_MAX_ITERATIONS_PER_STEP - 1:
                  print(f"Max iterations reached ({config.DSRC_MAX_ITERATIONS_PER_STEP}) for Step {step_index+1}. Proceeding with incomplete results.")
                  return step_results, step_assessments, step_llm_calls, step_brave_calls # 最大回数試行しても完了せず終了
             # else: ループ続行

    # ここに到達するのは通常、最大反復回数を超えた場合
    print(f"Step {step_index+1} finished after max iterations.")
    return step_results, step_assessments, step_llm_calls, step_brave_calls # ループ終了時のカウントで終了


# 戻り値にLLMコール数を追加 -> Optional[Tuple[str, int]]
async def generate_dsrc_report(question: str, plan: List[str], all_step_results: Dict[str, str], all_assessments: List[Dict[str, Any]]) -> Optional[Tuple[str, int]]:
    """!dsrc の最終レポートを生成する"""
    llm_calls = 0
    llm_handler = llm_manager.get_current_provider()
    primary_model_name = llm_manager.get_active_model_name('primary')
    lowload_model_name = llm_manager.get_active_model_name('lowload') # Lowloadモデル名も取得

    if not llm_handler or not primary_model_name:
        print("Error: Primary model unavailable for DSRC report generation.")
        return None # エラー時はNoneだけ返す

    plan_text = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))

    # 全検索結果テキスト (URL + content) を結合
    combined_search_results_text = "\n\n".join(
        f"--- Content from {url} ---\n{text}\n--- End of {url} ---"
        for url, text in all_step_results.items()
    )

    # --- 検索結果が長すぎる場合の要約処理 ---
    report_input_results_text = combined_search_results_text # デフォルトは元のテキスト
    report_input_source = "full results" # レポート入力が元の結果か要約かを示すラベル
    source_urls_list = list(all_step_results.keys()) # 元のURLリストは常に保持
    summarization_needed = False

    # 設定された最大入力文字数を超えているかチェック
    if len(combined_search_results_text) > config.MAX_INPUT_CHARS_FOR_SUMMARY:
        print(f"DSRC report input (all_results_text) length ({len(combined_search_results_text)}) exceeds summary limit ({config.MAX_INPUT_CHARS_FOR_SUMMARY}). Attempting to summarize using lowload model.")
        summarization_needed = True

        if lowload_model_name:
            await discord_ui.update_thinking_message(discord.utils.MISSING, f"…考え中... (検索結果を要約中 using {lowload_model_name})")
            summarize_prompt = config.SUMMARIZE_SEARCH_RESULTS_PROMPT.format(
                question=question,
                search_results_text=combined_search_results_text # 長い元のテキストを渡す
            )
            try:
                # Lowload モデルで要約を試みる
                # *** 修正箇所: generate_lowload_response 呼び出しから timeout を削除 (前回の修正済み) ***
                # llm_manager.generate_lowload_response 関数に timeout 引数は渡せなくなっています。
                summarized_results_raw = await llm_manager.generate_lowload_response(summarize_prompt) # timeout 引数を削除
                # generate_lowload_response が None を返すのはエラー時と想定
                if summarized_results_raw is not None:
                    summarized_results_text = str(summarized_results_raw).strip()
                    # 要約が成功し、エラーメッセージでないか、および「要約できませんでした」でないかチェック
                    if summarized_results_text and not llm_manager.is_error_message(summarized_results_text) and "要約できませんでした。" not in summarized_results_text:
                        print(f"Successfully summarized search results ({len(summarized_results_text)} chars).")
                        report_input_results_text = f"【収集された検索結果の要約】\n{summarized_results_text}" # 要約であることを明記
                        report_input_source = "summarized results"
                    else:
                        print(f"Lowload model failed to generate valid summary. Response: {summarized_results_text}. Using truncated full results.")
                        # 要約失敗時は、元のテキストを強制的に切り詰めて使用
                        report_input_results_text = combined_search_results_text[:config.MAX_INPUT_CHARS_FOR_SUMMARY] + "\n\n... (Full results truncated for report generation due to invalid summary)"
                        report_input_source = "truncated full results (invalid summary)"
                else: # summarized_results_raw が None だった場合
                     print("Lowload model failed to summarize search results (returned None). Using truncated full results.")
                     # 要約失敗時は、元のテキストを強制的に切り詰めて使用
                     report_input_results_text = combined_search_results_text[:config.MAX_INPUT_CHARS_FOR_SUMMARY] + "\n\n... (Full results truncated for report generation due to summarization failure)"
                     report_input_source = "truncated full results (summarization failure)"

            except Exception as e:
                print(f"Error during search results summarization: {e}. Using truncated full results.")
                import traceback
                traceback.print_exc()
                # 要約中に例外発生時も、元のテキストを強制的に切り詰めて使用
                report_input_results_text = combined_search_results_text[:config.MAX_INPUT_CHARS_FOR_SUMMARY] + "\n\n... (Full results truncated for report generation due to exception)"
                report_input_source = "truncated full results (exception)"
            finally:
                # 要約試行回数に関わらずLLMコールは1回とカウント (generate_lowload_response内でカウントされるかはllm_manager実装による)
                # ここでは llm_manager.generate_lowload_response 内でカウントされることを期待し、ここではカウントしない
                pass # generate_lowload_response 自体は内部でLLMを呼ぶため、llm_manager側でカウントされるべき


        else:
             print("Lowload model not available for summarization. Using truncated full results.")
             # Lowload モデルがない場合も、元のテキストを強制的に切り詰めて使用
             report_input_results_text = combined_search_results_text[:config.MAX_INPUT_CHARS_FOR_SUMMARY] + "\n\n... (Full results truncated for report generation, lowload model unavailable)"
             report_input_source = "truncated full results (lowload missing)"
    else:
        print(f"DSRC report input length ({len(combined_search_results_text)}) is within summary limit. Skipping summarization.")
        report_input_source = "full results (within limit)"


    # --- 全評価結果テキスト ---
    assessments_summary_lines = []
    for assessment in all_assessments:
         # Step実行部分でbrave_callsをassessmentに追加したので、ここでも参照
         line = f"- Step {assessment['step']} (Iter {assessment['iteration']} / Brave Calls: {assessment.get('brave_calls', 0)}): Status={assessment['status']}"
         # reason が None でないことを確認
         if assessment.get('reason') is not None:
              line += f", Reason={str(assessment['reason'])[:100]}..." # 長すぎる理由を省略
         if assessment.get('queries'): line += f", Queries={assessment['queries']}"
         # results はテキスト量が多いので省略するか、URLだけリストアップ
         # results が None でないことを確認
         if assessment.get('results') is not None:
              # iteration ごとに取得した新しい結果のURLをリストアップ
              new_result_urls = list(assessment['results'].keys())
              if new_result_urls:
                   line += f", New Results URLs={new_result_urls}"

         assessments_summary_lines.append(line)
    all_assessments_text = "\n".join(assessments_summary_lines)


    # 最終レポート生成プロンプト
    report_prompt = config.DSRC_FINAL_REPORT_PROMPT.format(
        question=question,
        plan=plan_text,
        all_results_text=report_input_results_text, # ここで要約または切り詰めたテキストを使用
        all_assessments_text=all_assessments_text # 評価サマリーテキスト
    )

    try:
        await discord_ui.update_thinking_message(discord.utils.MISSING, f"…考え中... (最終レポート生成中 using {primary_model_name}, input: {report_input_source})")
        # Primary モデルでレポート生成
        # generate_response には gemini_provider.py でデフォルトのタイムアウト(120秒)が設定されている
        _used_model, report_response_raw = await llm_manager.generate_response(
            content_parts=[{'text': report_prompt}], chat_history=None, deep_cache_summary=None
        )
        llm_calls += 1 # Primary LLMコールをカウント
        report_response = str(report_response_raw).strip() if report_response_raw else ""

        # LLMからの応答がエラーメッセージかどうか判定
        if llm_manager.is_error_message(report_response):
            print(f"DSRC Report generation failed. Response: {report_response}")
            # レポート生成失敗時は、エラーメッセージを返す
            return f"{bot_constants.ERROR_MSG_INTERNAL} (最終レポート生成失敗)\nReason: {report_response}", llm_calls

        print("DeepResearch Final Report generated successfully.")

        # --- 最終レポートにソースリストを追加 ---
        source_header = "**参照ソース:**"
        # LLMがソースリストを含めているかチェック (簡易)
        # 応答テキストを小文字にしてから検索
        if source_header.lower() not in report_response.lower():
             print("LLM did not include sources in the final report. Appending manually.")
             if source_urls_list:
                  source_list_text = "\n".join([f"- <{url}>" for url in source_urls_list])
                  report_response += f"\n\n{source_header}\n{source_list_text}"
             else:
                  report_response += f"\n\n{source_header}\n(ソースなし)" # ソースがない場合

        return report_response, llm_calls # 成功時はレポートとコール数を返す

    except Exception as e:
        print(f"Error during DSRC report generation: {e}")
        import traceback
        traceback.print_exc() # 詳細なエラーログ
        return f"{bot_constants.ERROR_MSG_INTERNAL} (最終レポート生成中に例外発生: {e})", llm_calls


# --- Search Command Handler ---
async def handle_search_command(
        message: discord.Message,
        command_type: Literal['src', 'dsrc'],
        query_text: str, # <- question_text ではなく query_text を使用
        triggered_by_assessment: bool = False # assess_and_respond_to_mention から呼ばれたか
    ):
    """!src および !dsrc コマンド、または自動検索の処理ハンドラ"""

    # --- APIコールカウンター初期化 ---
    total_llm_calls = 0
    total_brave_calls = 0

    if not config.BRAVE_SEARCH_API_KEY:
        await message.reply("検索機能は設定されていません (APIキー不足)。", mention_author=False)
        # Thinking Message が残っているかもしれないので削除
        await discord_ui.delete_thinking_message()
        return
    llm_handler = llm_manager.get_current_provider()
    if not llm_handler:
        await message.reply(bot_constants.ERROR_MSG_INTERNAL + " (LLM Provider not available)", mention_author=False)
        # Thinking Message が残っているかもしれないので削除
        await discord_ui.delete_thinking_message()
        return

    original_question = query_text.strip() # <- query_text を使用
    if not original_question:
        command_display = f"自動検索 ({command_type})" if triggered_by_assessment else f"!{command_type}"
        await message.reply(f"検索する内容を指定してください。例: `@{message.guild.me.display_name} {command_display} 内容`", mention_author=False)
        # Thinking Message が残っているかもしれないので削除
        await discord_ui.delete_thinking_message()
        return

    provider_name = llm_manager.get_current_provider_name()
    search_source = "検索が必要と判断されました" if triggered_by_assessment else f"!{command_type.upper()}"
    print(f"[{search_source}] Search process started for: '{original_question}' by {message.author.display_name} (Provider: {provider_name})")

    # 思考中メッセージ開始 (assess_and_respond_to_mention から呼ばれた場合は既に表示済み)
    if not triggered_by_assessment:
        thinking_msg_prefix = f"…考え中... ({search_source})"
        await discord_ui.update_thinking_message(message.channel, f"{thinking_msg_prefix} 開始")

    final_sent_message: Optional[discord.Message] = None # 送信した最終メッセージ
    final_response_text = "" # 最終的なLLMの応答テキスト
    all_extracted_content: Dict[str, str] = {} # 収集した全コンテンツ (URL -> text)

    try:
        # --- !src または 自動検索 の場合 ---
        if command_type == 'src':
            model_type = 'lowload'
            query_model_name = llm_manager.get_active_model_name(model_type)
            answer_model_name = query_model_name
            if not query_model_name:
                 await discord_ui.delete_thinking_message()
                 await message.reply(bot_constants.ERROR_MSG_LOWLOAD_UNAVAILABLE + f" ({provider_name})", mention_author=False); return

            thinking_msg_prefix = f"…考え中... ({search_source})" # 再設定 (assessmentからの引継ぎも考慮)
            await discord_ui.update_thinking_message(message.channel, f"{thinking_msg_prefix} クエリ生成中 ({query_model_name})")

            # 1. クエリ生成 (Lowloadモデルを使用)
            query_gen_prompt = config.SEARCH_QUERY_GENERATION_PROMPT.format(question=original_question)
            # generate_lowload_response を使用 (search_handler.py ではタイムアウトを渡さない方針)
            query_response_raw = await llm_manager.generate_lowload_response(query_gen_prompt)
            # generate_lowload_response が None を返すのはエラー時と想定
            if query_response_raw is None:
                 await discord_ui.delete_thinking_message(); await message.reply("検索クエリ生成失敗。", mention_author=False); return

            query_response_text = str(query_response_raw).strip()
            if llm_manager.is_error_message(query_response_text): # LLMからの応答がエラーメッセージかどうか判定
                 await discord_ui.delete_thinking_message(); await message.reply(f"検索クエリ生成失敗: {query_response_text}", mention_author=False); return

            queries_raw = query_response_text.replace('\n', ',')
            search_queries = [q.strip().strip('"') for q in queries_raw.split(',') if q.strip()][:3]
            if not search_queries: await discord_ui.delete_thinking_message(); await message.reply("有効な検索クエリ生成失敗。", mention_author=False); return
            print(f"[{search_source}] Generated queries: {search_queries}")
            total_llm_calls += 1 # クエリ生成LLMコールをカウント (generate_lowload_response内でカウントされるかはllm_manager実装によるが、ここでは確実にカウント)


            # 2. Brave Search & 内容取得
            search_results_api: List[Dict[str, Any]] = []
            for query in search_queries:
                 await discord_ui.update_thinking_message(message.channel, f"{thinking_msg_prefix} 検索中: `{query[:30]}...`")
                 total_brave_calls += 1 # カウント (キャッシュヒット含む)
                 results = await call_brave_search_api(query)
                 if results: search_results_api.extend(results)
                 # call_brave_search_api 内で遅延

            unique_urls = list(dict.fromkeys([r['url'] for r in search_results_api if 'url' in r]))
            if unique_urls:
                await discord_ui.update_thinking_message(message.channel, f"{thinking_msg_prefix} ページ内容取得中 {len(unique_urls)}件")
                fetch_tasks = [extract_text_from_url(url) for url in unique_urls]
                extracted_contents_list = await asyncio.gather(*fetch_tasks)
                for url, content in zip(unique_urls, extracted_contents_list):
                    if content: all_extracted_content[url] = content # 全体結果に集約
                print(f"[{search_source}] Extracted content from {len(all_extracted_content)}/{len(unique_urls)} URLs.")

            if not all_extracted_content: await discord_ui.delete_thinking_message(); await message.reply("検索結果から有効な情報を抽出できませんでした。", mention_author=False); return

            # 3. 最終応答生成 (Lowloadモデルを使用)
            await discord_ui.update_thinking_message(message.channel, f"{thinking_msg_prefix} 応答生成中 ({answer_model_name})")
            combined_results_text_for_llm = "\n\n".join(f"--- {url} ---\n{text}\n--- End ---" for url, text in all_extracted_content.items())
            if len(combined_results_text_for_llm) > config.MAX_TOTAL_SEARCH_CONTENT_LENGTH: # configの値を再利用
                 combined_results_text_for_llm = combined_results_text_for_llm[:config.MAX_TOTAL_SEARCH_CONTENT_LENGTH] + "...(truncated)"

            answer_prompt = config.SIMPLE_SEARCH_ANSWER_PROMPT.format(question=original_question, search_results_text=combined_results_text_for_llm)
            # generate_lowload_response を使用 (search_handler.py ではタイムアウトを渡さない方針)
            final_response_raw = await llm_manager.generate_lowload_response(answer_prompt) # Lowloadモデル
            # generate_lowload_response が None を返すのはエラー時と想定
            if final_response_raw is None:
                 await discord_ui.delete_thinking_message(); await message.reply("応答生成失敗。", mention_author=False); return

            final_response_text = str(final_response_raw).strip()
            if llm_manager.is_error_message(final_response_text): # LLMからの応答がエラーメッセージかどうか判定
                 await discord_ui.delete_thinking_message(); await message.reply(f"応答生成失敗: {final_response_text}", mention_author=False); return

            total_llm_calls += 1 # 最終応答生成LLMコールをカウント (generate_lowload_response内でカウントされるかはllm_manager実装によるが、ここでは確実にカウント)


            # response_header は後で設定

        # --- !dsrc の場合 ---
        elif command_type == 'dsrc':
            primary_model_name = llm_manager.get_active_model_name('primary')
            if not primary_model_name:
                 await discord_ui.delete_thinking_message(); await message.reply(f"{bot_constants.ERROR_MSG_API_ERROR} ({provider_name})", mention_author=False); return

            all_assessments: List[Dict[str, Any]] = []
            thinking_msg_prefix = f"…考え中... ({search_source})"

            # 1. 計画生成
            await discord_ui.update_thinking_message(message.channel, f"{thinking_msg_prefix} 調査計画生成中 ({primary_model_name})")
            # generate_dsrc_plan 内で generate_response が呼ばれる。generate_response には gemini_provider.py でデフォルトのタイムアウトが設定されている。
            plan_result = await generate_dsrc_plan(original_question)
            if not plan_result: # generate_dsrc_plan が None を返すのはエラー時と想定
                 await discord_ui.delete_thinking_message(); await message.reply("調査計画の生成に失敗しました。", mention_author=False); return

            plan, plan_llm_calls = plan_result
            total_llm_calls += plan_llm_calls # カウント加算
            print(f"[{search_source}] Generated Plan: {plan}")

            # 2. 各ステップ実行
            for i, step_description in enumerate(plan):
                print(f"--- Executing DSRC Step {i+1}: {step_description} ---")
                await discord_ui.update_thinking_message(discord.utils.MISSING, f"{thinking_msg_prefix} ステップ {i+1}/{len(plan)} 実行中: {step_description[:30]}...")

                # execute_dsrc_step を呼び出し、返り値からカウントを取得
                # execute_dsrc_step 内で generate_response が呼ばれる。generate_response には gemini_provider.py でデフォルトのタイムアウトが設定されている。
                step_results, step_assessments, step_llm, step_brave = await execute_dsrc_step(
                    original_question, step_description, i, all_extracted_content
                )
                total_llm_calls += step_llm # カウント加算
                total_brave_calls += step_brave # カウント加算
                all_extracted_content.update(step_results)
                all_assessments.extend(step_assessments)

                # ステップ実行中にエラーが発生した場合 (assessment の status が ERROR)
                # all_assessments のうち、現在のステップに該当するものだけチェック
                if any(a.get('status') == 'ERROR' for a in step_assessments if a.get('step') == i+1):
                     print(f"Error occurred during Step {i+1}. Stopping DSRC process.")
                     await discord_ui.delete_thinking_message()
                     error_reason = "Unknown error"
                     # このステップのassessmentからエラー理由を探す
                     for a in step_assessments: # このステップの評価リストから探す
                         if a.get('step') == i+1 and a.get('status') == 'ERROR': error_reason = a.get('reason', 'Unknown error'); break
                     await message.reply(f"詳細検索ステップ {i+1} でエラーが発生したため処理を中断しました。\n理由: {error_reason}", mention_author=False)
                     return # 早期リターン

                # 各ステップ終了後に少し待機 (API負荷軽減)
                await asyncio.sleep(1)

            # 3. 最終レポート生成
            if not all_extracted_content:
                 await discord_ui.delete_thinking_message(); await message.reply("詳細検索の結果、有効な情報が見つかりませんでした。", mention_author=False); return

            # generate_dsrc_report には generate_response (Primaryモデル, デフォルトタイムアウト) と generate_lowload_response (Lowloadモデル, 要約用タイムアウト) が含まれる
            report_result = await generate_dsrc_report(original_question, plan, all_extracted_content, all_assessments)

            if report_result is None: # generate_dsrc_report が None を返した場合 (非常に稀な内部エラー)
                await discord_ui.delete_thinking_message(); await message.reply(f"{bot_constants.ERROR_MSG_INTERNAL} (最終レポート生成プロセスで予期せぬエラー)", mention_author=False); return

            report_text, report_llm_calls = report_result
            total_llm_calls += report_llm_calls # カウント加算
            final_response_text = report_text # レポートテキスト (エラーメッセージ含む可能性あり)

            if llm_manager.is_error_message(final_response_text): # generate_dsrc_reportがエラーメッセージを返した場合
                 await discord_ui.delete_thinking_message(); await message.reply(f"最終レポート生成失敗: {final_response_text}", mention_author=False); return

            # response_header は後で設定

        # --- 共通: 最終応答送信 ---
        await discord_ui.delete_thinking_message()

        # --- response_header に API コール数を追加 ---
        # brave calls はキャッシュヒット含むため注意書きを追加
        api_usage_info = f"(LLM Calls: {total_llm_calls}, Brave Calls: {total_brave_calls} [incl. cache hits])"
        if command_type == 'src':
            answer_model_name = llm_manager.get_active_model_name('lowload') or "Lowload"
            response_header = f"{api_usage_info}\n(🔍 **Search Result** using {answer_model_name} 🔍)\n\n"
        elif command_type == 'dsrc':
            primary_model_name = llm_manager.get_active_model_name('primary') or "Primary"
            response_header = f"{api_usage_info}\n(🔬 **DeepResearch Report** using {primary_model_name} 🔬)\n\n"
        else: # 万が一のため
            response_header = f"{api_usage_info}\n\n"

        full_response = response_header + final_response_text

        # ソースリストのフォールバック (generate_dsrc_report内で処理済み)
        # ...

        # 応答メッセージ分割送信 (message.reply を使用)
        if len(full_response) > 2000:
            print(f"[{search_source}] Final response length ({len(full_response)}) exceeds 2000. Sending in chunks.")
            response_chunks = [full_response[i:i+1990] for i in range(0, len(full_response), 1990)]
            first_chunk = True
            try:
                for chunk in response_chunks:
                    if first_chunk:
                        # 最初のチャンク送信時にメッセージオブジェクトを取得
                        final_sent_message = await message.reply(chunk, mention_author=False)
                        first_chunk = False
                    else:
                        # 2通目以降はチャンネルに直接送信 (Replyにならないが、会話の流れは維持)
                        await message.channel.send(chunk)
                    await asyncio.sleep(0.5) # 連投制限対策
            except discord.HTTPException as e:
                 print(f"[{search_source}] Error sending chunked final response: {e}")
                 if not final_sent_message: # 最初の送信で失敗した場合
                      await message.channel.send(bot_constants.ERROR_MSG_INTERNAL + " (応答送信失敗)")
        else:
            # 2000文字以下の場合は一括送信
            try:
                # メッセージオブジェクトを取得
                final_sent_message = await message.reply(full_response, mention_author=False)
            except discord.HTTPException as e:
                 print(f"[{search_source}] Error sending final response: {e}")
                 await message.channel.send(bot_constants.ERROR_MSG_INTERNAL + " (応答送信失敗)")


        # --- キャッシュ更新と追跡質問ボタン (応答成功後) ---
        # エラーメッセージでない、かつメッセージ送信に成功した場合のみ実行
        if final_sent_message and final_response_text and not llm_manager.is_error_message(final_response_text):
            try:
                print(f"[{search_source}] Updating cache for channel {message.channel.id}...")
                # mentionを除去した完全なユーザー入力テキスト (コマンド/質問含む)
                # bot.py や assess_and_respond_to_mention から渡された query_text を使用
                # query_text が None の場合は message.content から再構築 (フォールバック)
                user_input_for_cache = query_text # <- question_text ではなく query_text を使用
                if user_input_for_cache is None:
                    # このルートは handle_search_command の呼び出し元で query_text が None でない限り通らないが、念のため
                    print(f"Warning: query_text is None in handle_search_command cache update logic. Reconstructing from message.content.")
                    mention_strings = [f'<@!{message.guild.me.id}>', f'<@{message.guild.me.id}>']
                    user_input_for_cache = message.content if message.content else ""
                    for mention in mention_strings:
                        user_input_for_cache = user_input_for_cache.replace(mention, '').strip()
                    # !src, !dsrc, -nosrc, !his なども除去 ( command_handler の handle_mention に合わせる)
                    user_input_for_cache = re.sub(r'\s!-?[sS][rR][cC]\b', '', user_input_for_cache, flags=re.IGNORECASE)
                    user_input_for_cache = re.sub(r'\s-nosrc\b', '', user_input_for_cache, flags=re.IGNORECASE)
                    user_input_for_cache = re.sub(r'\b!his\b', '', user_input_for_cache, flags=re.IGNORECASE).strip()

                # 添付ファイルは cache_manager の save_cache 内で処理されるので、ここでは含めない
                # もし将来的に検索機能が添付ファイル入力に対応する場合、ここも修正が必要
                # 例: message.attachments を処理して user_entry_parts_for_cache に追加

                user_entry_parts_for_cache: List[Dict[str, Any]] = []
                if user_input_for_cache:
                    user_entry_parts_for_cache.append({'text': user_input_for_cache})


                if user_entry_parts_for_cache:
                     chat_history = await cache_manager.load_cache(message.channel.id)
                     user_entry = {'role': 'user', 'parts': user_entry_parts_for_cache}
                     model_entry = {'role': 'model', 'parts': [{'text': final_response_text}]} # LLMの応答全文
                     await cache_manager.save_cache(message.channel.id, chat_history + [user_entry, model_entry])
                     print(f"[{search_source}] Cache updated.")
                else:
                     print(f"[{search_source}] Skipping cache update because user entry parts are empty.")

            except Exception as cache_e:
                print(f"[{search_source}] Error updating cache: {cache_e}")
                import traceback
                traceback.print_exc()

            try:
                 print(f"[{search_source}] Generating follow-up buttons...")
                 # message.channel.id を渡す
                 asyncio.create_task(discord_ui.generate_and_add_followup_buttons(final_sent_message, message.channel.id))
            except Exception as btn_e:
                 print(f"[{search_source}] Error scheduling follow-up button generation: {btn_e}")
                 import traceback
                 traceback.print_exc()


    except Exception as e:
        print(f"[{search_source}] An unexpected error occurred during search process: {e}")
        import traceback
        traceback.print_exc()
        await discord_ui.delete_thinking_message()
        # message.reply の代わりに message.channel.send を使う (replyはキャッシュされたメッセージに依存する可能性)
        await message.channel.send(bot_constants.ERROR_MSG_INTERNAL + f" (検索処理中にエラー: {str(e)[:100]}...)")