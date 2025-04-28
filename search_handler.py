# search_handler.py
# (検索コマンドの処理、Brave Search API連携、URLテキスト抽出)

import asyncio
import re
import httpx
import discord
from typing import List, Dict, Any, Optional, Tuple, Literal

import config
import bot_constants
import llm_manager
import cache_manager # キャッシュ保存のため追加
import discord_ui # Thinking message, ボタン生成用
from llm_provider import ERROR_TYPE_UNKNOWN # エラータイプ定数

# --- Brave Search API Call ---
async def call_brave_search_api(query: str) -> Optional[List[Dict[str, Any]]]:
    """Brave Search APIを呼び出す"""
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
            # API呼び出しごとに必ず待機 (try/except/finallyで保証)
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
        return None


# --- Search Command Handler ---
async def handle_search_command(message: discord.Message, command_type: Literal['src', 'dsrc'], query_text: str):
    """!src および !dsrc コマンドの共通処理ハンドラ"""

    if not config.BRAVE_SEARCH_API_KEY:
        await message.reply("検索機能は設定されていません (APIキー不足)。", mention_author=False)
        return

    llm_handler = llm_manager.get_current_provider()
    if not llm_handler:
        await message.reply(bot_constants.ERROR_MSG_INTERNAL + " (LLM Provider not available)", mention_author=False)
        return

    # モデル選択と設定
    if command_type == 'src':
        model_type = 'lowload'
        query_model_name = llm_manager.get_active_model_name(model_type)
        answer_model_name = query_model_name # srcは両方lowload
        if not query_model_name:
            await message.reply(bot_constants.ERROR_MSG_LOWLOAD_UNAVAILABLE + f" ({llm_manager.get_current_provider_name()} に低負荷モデルが設定されていません)", mention_author=False)
            return
        max_iterations = 1
    elif command_type == 'dsrc':
        model_type = 'primary'
        query_model_name = llm_manager.get_active_model_name(model_type)
        answer_model_name = query_model_name # dsrcは両方primary
        if not query_model_name:
            # Primary がない場合、Secondaryで代用を試みる (llm_manager側で調整されるべきかもしれない)
            query_model_name = llm_manager.get_active_model_name('secondary')
            answer_model_name = query_model_name
            if not query_model_name:
                await message.reply(bot_constants.ERROR_MSG_API_ERROR + f" ({llm_manager.get_current_provider_name()} に利用可能なPrimary/Secondaryモデルが設定されていません)", mention_author=False)
                return
            else:
                print(f"Warning: Primary model not found for dsrc, using secondary: {query_model_name}")
        max_iterations = config.DEEP_SEARCH_MAX_ITERATIONS
    else: # command_typeが予期せぬ値の場合
         print(f"Error: Invalid command_type '{command_type}' in handle_search_command.")
         await message.reply(bot_constants.ERROR_MSG_INTERNAL, mention_author=False)
         return

    original_question = query_text.strip()
    if not original_question:
        await message.reply(f"検索する内容を指定してください。例: `@{message.guild.me.display_name} !{command_type} ChatGPTの最新情報`", mention_author=False)
        return

    provider_name = llm_manager.get_current_provider_name()
    print(f"[{command_type.upper()}] Search command received: '{original_question}' by {message.author.display_name} (Provider: {provider_name})")

    # 思考中メッセージ開始
    await discord_ui.update_thinking_message(message.channel, f"…考え中... (検索開始)") # プラナ風

    used_search_queries: List[str] = []
    all_extracted_content: Dict[str, str] = {} # URL -> text ; 全イテレーションの結果を集約
    iteration_count = 0
    should_continue_search = True
    combined_search_results_text = "" # スコープ外でも使えるように初期化
    missing_info_from_assessment: Optional[str] = None # 前回の評価で不足していた情報を保持
    final_sent_message: Optional[discord.Message] = None # 送信した最終メッセージを格納

    try:
        while should_continue_search and iteration_count < max_iterations:
            iteration_count += 1
            iteration_label = f"第{iteration_count}回" if command_type == 'dsrc' else ""

            # 1. 検索クエリ生成
            await discord_ui.update_thinking_message(message.channel, f"…考え中... ({iteration_label} 検索クエリ生成中 using {query_model_name})")

            query_gen_prompt = ""
            # dsrcの2回目以降で、かつ過去のクエリがある場合
            if command_type == 'dsrc' and iteration_count > 1 and used_search_queries:
                formatted_used_queries = "\n".join([f"- {q}" for q in used_search_queries])
                # 前回のループで得られた不足情報を利用（なければ「特に指定なし」）
                prompt_missing_info = missing_info_from_assessment if missing_info_from_assessment else "特に指定なし"
                query_gen_prompt = config.SEARCH_QUERY_GENERATION_PROMPT_WITH_HISTORY.format(
                    question=original_question,
                    used_queries=formatted_used_queries,
                    missing_info=prompt_missing_info # config.py側のテンプレートに追加
                )
            else: # 初回 または src の場合
                 query_gen_prompt = config.SEARCH_QUERY_GENERATION_PROMPT.format(question=original_question)

            # LLMでクエリ生成 (srcはlowload, dsrcはprimary/secondary=generate_response)
            query_response_raw: Optional[str] = None
            if command_type == 'src':
                query_response_raw = await llm_manager.generate_lowload_response(query_gen_prompt)
            elif command_type == 'dsrc':
                # generate_responseはモデル名と応答テキストのタプルを返す
                _used_model_q, response_text = await llm_manager.generate_response(content_parts=[{'text': query_gen_prompt}], chat_history=None, deep_cache_summary=None)
                query_response_raw = response_text # 応答テキストのみ使用

            query_response_text = str(query_response_raw).strip() if query_response_raw else "" # strip() を追加
            if not query_response_text or llm_manager.is_error_message(query_response_text):
                print(f"[{command_type.upper()}] Query generation failed (Iteration {iteration_count}). Response: {query_response_text}")
                await discord_ui.delete_thinking_message()
                error_msg = llm_handler.format_error_message(ERROR_TYPE_UNKNOWN, 'Query generation failed') if llm_handler else bot_constants.ERROR_MSG_GEMINI_UNKNOWN
                await message.reply(f"検索クエリの生成に失敗しました。{error_msg}", mention_author=False)
                return

            # クエリをパース (改行またはカンマ区切り)
            queries_raw = query_response_text.replace('\n', ',')
            current_iteration_queries = [q.strip().strip('"') for q in queries_raw.split(',') if q.strip()] # クォートも除去
            current_iteration_queries = [q for q in current_iteration_queries if q] # 空のクエリを除去
            current_iteration_queries = current_iteration_queries[:3] # 最大3つに制限

            if not current_iteration_queries:
                if iteration_count == 1:
                     await discord_ui.delete_thinking_message()
                     await message.reply("有効な検索クエリを生成できませんでした。", mention_author=False)
                     print(f"[{command_type.upper()}] Generated empty query list from '{query_response_text}'")
                     return
                else:
                     print(f"[{command_type.upper()}] Generated empty query list in iteration {iteration_count}. Ending search.")
                     should_continue_search = False
                     break # ループを抜ける

            # 新しいクエリのみを used_search_queries に追加
            newly_added_queries = [q for q in current_iteration_queries if q not in used_search_queries]
            if newly_added_queries:
                 used_search_queries.extend(newly_added_queries)
                 print(f"[{command_type.upper()}] Iteration {iteration_count} queries added: {newly_added_queries}. Total used: {used_search_queries}")
            else:
                 print(f"[{command_type.upper()}] Iteration {iteration_count} generated only duplicate queries. Using existing: {current_iteration_queries}")
                 # 重複クエリでも検索は試みる（前回失敗した可能性もあるため）


            # 2. Brave Search API呼び出し
            current_iteration_results: List[Dict[str, Any]] = []
            for i, query in enumerate(current_iteration_queries):
                 await discord_ui.update_thinking_message(message.channel, f"…考え中... ({iteration_label} 検索中: `{query[:50]}...`)")
                 results = await call_brave_search_api(query)
                 if results:
                     current_iteration_results.extend(results)
                 # call_brave_search_api内で遅延処理済み

            if not current_iteration_results:
                await discord_ui.update_thinking_message(message.channel, f"…考え中... ({iteration_label} 検索結果なし)")
                if command_type == 'src' or iteration_count == 1: # 初回dsrcで結果なし
                     # 既存の結果もない場合は終了
                     if not all_extracted_content:
                         await discord_ui.delete_thinking_message()
                         await message.reply(f"「{original_question[:50]}...」に関する検索結果が見つかりませんでした。", mention_author=False)
                         return
                     else: # 既存の結果はあるので、それを元に応答生成へ
                          print(f"[{command_type.upper()}] No results in iteration {iteration_count}, but previous results exist. Ending search.")
                          should_continue_search = False
                          break
                else: # dsrc 2回目以降で結果なし
                     print(f"[{command_type.upper()}] No results in iteration {iteration_count}. Ending search.")
                     should_continue_search = False
                     break # ループを抜ける

            # 3. ページ内容取得と集約
            unique_urls_in_iteration = list(dict.fromkeys([r['url'] for r in current_iteration_results if 'url' in r]))
            # まだ取得していないURLのみを対象とする
            urls_to_fetch = [url for url in unique_urls_in_iteration if url not in all_extracted_content]
            print(f"[{command_type.upper()}] Iteration {iteration_count}: Found {len(unique_urls_in_iteration)} unique URLs, fetching {len(urls_to_fetch)} new URLs.")

            if urls_to_fetch:
                await discord_ui.update_thinking_message(message.channel, f"…考え中... ({iteration_label} ページ内容取得中 {len(urls_to_fetch)}件)")
                fetch_tasks = [extract_text_from_url(url) for url in urls_to_fetch]
                extracted_contents_list = await asyncio.gather(*fetch_tasks)

                newly_extracted_count = 0
                for url, content in zip(urls_to_fetch, extracted_contents_list):
                    if content:
                        all_extracted_content[url] = content # 新しい内容を辞書に追加
                        newly_extracted_count += 1
                print(f"[{command_type.upper()}] Iteration {iteration_count}: Successfully extracted content from {newly_extracted_count}/{len(urls_to_fetch)} new URLs.")

            # このイテレーションで有効なコンテンツが全く取得できなかった場合 (新規URLも含む)
            # かつ、既存のコンテンツもない場合
            if not all_extracted_content:
                 await discord_ui.update_thinking_message(message.channel, f"…考え中... ({iteration_label} 有効なページ内容取得できず)")
                 await discord_ui.delete_thinking_message()
                 await message.reply(f"取得したページから有効な情報を抽出できませんでした。", mention_author=False)
                 return

            # 4. dsrcの場合、検索結果の評価 (集約された全結果を使って評価)
            missing_info_from_assessment = None # 各イテレーションの評価前にリセット
            if command_type == 'dsrc':
                # 結合と切り詰め (評価用)
                combined_search_results_text = "\n\n".join(
                    f"--- Content from {url} ---\n{text}\n--- End of {url} ---"
                    for url, text in all_extracted_content.items() # 全体の結果を使う
                )
                if len(combined_search_results_text) > config.MAX_TOTAL_SEARCH_CONTENT_LENGTH:
                    print(f"[{command_type.upper()}] Combined search content for assessment exceeds total limit. Truncating.")
                    combined_search_results_text = combined_search_results_text[:config.MAX_TOTAL_SEARCH_CONTENT_LENGTH] + "\n\n... (truncated due to length limit)"

                await discord_ui.update_thinking_message(message.channel, f"…考え中... ({iteration_label} 検索結果確認中 using {answer_model_name})")
                assessment_prompt = config.DEEP_SEARCH_ASSESSMENT_PROMPT.format(
                    question=original_question,
                    search_results_text=combined_search_results_text
                )
                # generate_responseを使用
                _used_model_a, assessment_response_raw = await llm_manager.generate_response(
                    content_parts=[{'text': assessment_prompt}], chat_history=None, deep_cache_summary=None
                )
                assessment_response_text = str(assessment_response_raw).strip() if assessment_response_raw else "" # strip() を追加

                if assessment_response_text and not llm_manager.is_error_message(assessment_response_text):
                    print(f"[{command_type.upper()}] Iteration {iteration_count} assessment: {assessment_response_text}")
                    if assessment_response_text.upper() == 'COMPLETE':
                        print(f"[{command_type.upper()}] Assessment: COMPLETE. Ending search loop.")
                        should_continue_search = False
                    elif assessment_response_text.upper().startswith('INCOMPLETE:'):
                        # ここで不足情報を抽出し、次のループで使用するために保持
                        missing_info_from_assessment = assessment_response_text.split(':', 1)[1].strip() if ':' in assessment_response_text else "詳細不明"
                        print(f"[{command_type.upper()}] Assessment: INCOMPLETE. Missing: {missing_info_from_assessment[:100]}...")
                        if iteration_count < max_iterations:
                             await discord_ui.update_thinking_message(message.channel, f"…考え中... ({iteration_label} 追加情報探索準備中)")
                             await asyncio.sleep(1) # 次の検索まで少し待つ
                        else:
                            print(f"[{command_type.upper()}] Max iterations reached ({max_iterations}). Ending search loop.")
                            should_continue_search = False
                    else:
                         # 予期せぬ形式でも、とりあえず続行してみる（内容が次のクエリ生成のヒントになるかもしれない）
                         print(f"[{command_type.upper()}] Unexpected assessment format: '{assessment_response_text}'. Continuing search if possible.")
                         if iteration_count >= max_iterations:
                             print(f"[{command_type.upper()}] Max iterations reached after unexpected assessment. Ending search loop.")
                             should_continue_search = False
                else:
                    print(f"[{command_type.upper()}] Assessment failed (Iteration {iteration_count}). Response: {assessment_response_text}. Ending search loop.")
                    should_continue_search = False # 評価失敗時はループ終了

            elif command_type == 'src':
                should_continue_search = False # src は1回で終了

        # --- ループ終了後 ---

        # 最終応答のために combined_search_results_text を再生成 (最新の all_extracted_content を使用)
        combined_search_results_text = "\n\n".join(
            f"--- Content from {url} ---\n{text}\n--- End of {url} ---"
            for url, text in all_extracted_content.items()
        )
        if len(combined_search_results_text) > config.MAX_TOTAL_SEARCH_CONTENT_LENGTH:
            print(f"[{command_type.upper()}] Final combined search content exceeds total limit. Truncating.")
            combined_search_results_text = combined_search_results_text[:config.MAX_TOTAL_SEARCH_CONTENT_LENGTH] + "\n\n... (truncated due to length limit)"


        if not combined_search_results_text: # ループを抜けた結果、有効なコンテンツが全くない場合
             await discord_ui.delete_thinking_message()
             await message.reply("検索によって質問に回答するための有効な情報が得られませんでした。", mention_author=False)
             return

        await discord_ui.update_thinking_message(message.channel, f"…考え中... (最終応答生成中 using {answer_model_name})")

        # 最終応答生成プロンプト (ペルソナ反映、ソース指示込み)
        answer_prompt = config.SEARCH_ANSWER_PROMPT.format(
            question=original_question,
            search_results_text=combined_search_results_text
        )

        # 最終応答生成
        final_response_raw: Optional[str] = None
        used_model_name_for_header = answer_model_name or "N/A"

        if command_type == 'src':
             final_response_raw = await llm_manager.generate_lowload_response(answer_prompt)
        elif command_type == 'dsrc':
             _used_model_f, response_text = await llm_manager.generate_response(
                 content_parts=[{'text': answer_prompt}], chat_history=None, deep_cache_summary=None
             )
             final_response_raw = response_text

        final_response_text = str(final_response_raw).strip() if final_response_raw else "" # strip() を追加

        if not final_response_text or llm_manager.is_error_message(final_response_text):
            print(f"[{command_type.upper()}] Final answer generation failed. Response: {final_response_text}")
            await discord_ui.delete_thinking_message()
            error_msg = llm_handler.format_error_message(ERROR_TYPE_UNKNOWN, 'Answer generation failed') if llm_handler else bot_constants.ERROR_MSG_GEMINI_UNKNOWN
            await message.reply(f"応答の生成に失敗しました。{error_msg}", mention_author=False)
            return

        # 最終応答をDiscordに送信
        await discord_ui.delete_thinking_message()
        response_header = f"(🔍 **{command_type.upper()} Search Result** using {used_model_name_for_header} 🔍)\n\n"
        full_response = response_header + final_response_text

        # --- プロンプトで指示したソースリストがLLMによって生成されなかった場合のフォールバック ---
        source_header = "**参照ソース:**"
        if source_header not in full_response and all_extracted_content:
             print(f"[{command_type.upper()}] LLM did not include sources. Appending manually.")
             source_list = "\n".join([f"- <{url}>" for url in all_extracted_content.keys()])
             full_response += f"\n\n{source_header}\n{source_list}"


        # 応答メッセージ分割送信
        if len(full_response) > 2000:
            print(f"[{command_type.upper()}] Final response length ({len(full_response)}) exceeds 2000. Sending in chunks.")
            response_chunks = [full_response[i:i+1990] for i in range(0, len(full_response), 1990)]
            first_chunk = True
            try:
                for chunk in response_chunks:
                    if first_chunk:
                        # 最初のチャンク送信時にメッセージオブジェクトを取得
                        final_sent_message = await message.reply(chunk, mention_author=False)
                        first_chunk = False
                    else:
                        await message.channel.send(chunk)
                    await asyncio.sleep(0.5)
            except discord.HTTPException as e:
                 print(f"[{command_type.upper()}] Error sending chunked final response: {e}")
                 if not final_sent_message: # 最初の送信で失敗した場合
                      await message.channel.send(bot_constants.ERROR_MSG_INTERNAL + " (応答送信失敗)")
        else:
            try:
                # メッセージオブジェクトを取得
                final_sent_message = await message.reply(full_response, mention_author=False)
            except discord.HTTPException as e:
                 print(f"[{command_type.upper()}] Error sending final response: {e}")
                 await message.channel.send(bot_constants.ERROR_MSG_INTERNAL + " (応答送信失敗)")


        # --- キャッシュ更新と追跡質問ボタンの追加 (応答成功後) ---
        if final_sent_message and final_response_text and not llm_manager.is_error_message(final_response_text):
            # 1. キャッシュ更新
            try:
                print(f"[{command_type.upper()}] Updating cache for channel {message.channel.id}...")
                # mentionを除去した完全なユーザー入力テキスト (コマンド含む)
                mention_strings = [f'<@!{message.guild.me.id}>', f'<@{message.guild.me.id}>']
                user_input_text = message.content # オリジナルのメッセージ内容を取得
                for mention in mention_strings:
                    user_input_text = user_input_text.replace(mention, '').strip()

                chat_history = await cache_manager.load_cache(message.channel.id)
                user_entry = {'role': 'user', 'parts': [{'text': user_input_text}]} # 検索コマンドとクエリ
                model_entry = {'role': 'model', 'parts': [{'text': final_response_text}]} # LLMの応答
                await cache_manager.save_cache(message.channel.id, chat_history + [user_entry, model_entry])
                print(f"[{command_type.upper()}] Cache updated.")
            except Exception as cache_e:
                print(f"[{command_type.upper()}] Error updating cache: {cache_e}")

            # 2. 追跡質問ボタン生成
            try:
                 # 非同期でボタン生成・追加を実行
                 print(f"[{command_type.upper()}] Generating follow-up buttons...")
                 asyncio.create_task(discord_ui.generate_and_add_followup_buttons(final_sent_message, message.channel.id))
            except Exception as btn_e:
                 print(f"[{command_type.upper()}] Error scheduling follow-up button generation: {btn_e}")


    except Exception as e:
        print(f"[{command_type.upper()}] An unexpected error occurred during search process: {e}")
        import traceback
        traceback.print_exc()
        await discord_ui.delete_thinking_message()
        # message.reply の代わりに message.channel.send を使う (replyはキャッシュされたメッセージに依存する可能性)
        await message.channel.send(bot_constants.ERROR_MSG_INTERNAL + f" (検索処理中にエラー: {str(e)[:100]}...)")