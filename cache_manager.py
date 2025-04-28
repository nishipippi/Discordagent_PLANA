# cache_manager.py
# (通常キャッシュとDeep Cacheの管理)

import os
import json
import base64
import aiofiles
import asyncio
from typing import List, Dict, Any, Optional, Tuple

import config
import bot_constants
import llm_manager # LLM呼び出し用

# --- キャッシュディレクトリ作成 ---
def ensure_cache_directories():
    """キャッシュ用ディレクトリが存在することを確認・作成する"""
    for cache_dir in [config.CACHE_DIR, config.DEEP_CACHE_DIR]:
        if not os.path.exists(cache_dir):
            try:
                os.makedirs(cache_dir)
                print(f"ディレクトリ '{cache_dir}' 作成。")
            except Exception as e:
                print(f"警告: '{cache_dir}' 作成失敗: {e}")

# --- 通常キャッシュ ---

async def load_cache(channel_id: int) -> List[Dict[str, Any]]:
    """指定チャンネルの通常キャッシュを読み込む"""
    cache_file = os.path.join(config.CACHE_DIR, f"{channel_id}.json")
    if not os.path.exists(cache_file): return []
    try:
        async with aiofiles.open(cache_file, mode='r', encoding='utf-8') as f:
            content = await f.read()
            if not content: return []
            data = json.loads(content)
            # Base64デコード処理
            for entry in data:
                decoded_parts = []
                if 'parts' not in entry or not isinstance(entry.get('parts'), list): continue
                for part in entry['parts']:
                    decoded_part = {}
                    if 'text' in part:
                        decoded_part['text'] = part['text']
                    elif 'inline_data' in part and isinstance(part.get('inline_data', {}).get('data'), str):
                        try:
                            decoded_part['inline_data'] = {
                                'mime_type': part['inline_data']['mime_type'],
                                'data': base64.b64decode(part['inline_data']['data'])
                            }
                        except Exception as e:
                            print(f"警告: キャッシュBase64デコード失敗: {e}, スキップします。 Part: {part}")
                            continue # エラーがあっても他のパーツは処理
                    else: pass # textでもinline_dataでもないものは無視
                    # decoded_partが空でない場合のみ追加 (textもinline_dataもなかった場合など)
                    if decoded_part:
                         decoded_parts.append(decoded_part)

                entry['parts'] = decoded_parts
            return data
    except json.JSONDecodeError:
        print(f"警告: キャッシュ {cache_file} が壊れています。リセットします。")
        await save_cache(channel_id, []) # 壊れたファイルを空にする
        return []
    except Exception as e:
        print(f"エラー: キャッシュ {cache_file} 読み込み失敗: {e}")
        return []

async def save_cache(channel_id: int, history: List[Dict[str, Any]]):
    """指定チャンネルの通常キャッシュを保存する"""
    ensure_cache_directories() # ディレクトリ確認
    cache_file = os.path.join(config.CACHE_DIR, f"{channel_id}.json")
    try:
        # キャッシュ上限処理とDeep Cacheへの連携
        num_entries_to_keep = config.CACHE_LIMIT * 2 # 保存する最大エントリ数 (ユーザー+モデル)
        history_to_save = history
        if len(history) > num_entries_to_keep:
            # 上限を超えた場合、古い部分をDeep Cache更新用に渡し、新しい部分をキャッシュに残す
            history_for_deep_cache = history[:-num_entries_to_keep]
            history_to_save = history[-num_entries_to_keep:]
            print(f"キャッシュ上限超過。古い履歴 ({len(history_for_deep_cache)}件) をDeep Cache更新に使用します。")
            # Deep Cache 更新を非同期タスクとして実行
            asyncio.create_task(update_deep_cache(channel_id, history_for_deep_cache))

        # 保存する履歴をエンコード
        encoded_history = []
        for entry in history_to_save:
            encoded_parts = []
            # 'parts' が存在し、リストであることを確認
            if 'parts' not in entry or not isinstance(entry.get('parts'), list): continue
            for part in entry['parts']:
                encoded_part = {}
                if 'text' in part:
                    encoded_part['text'] = part['text']
                elif 'inline_data' in part and isinstance(part.get('inline_data', {}).get('data'), bytes):
                    # inline_dataがあり、dataがbytes型の場合のみエンコード
                    try:
                        encoded_part['inline_data'] = {
                            'mime_type': part['inline_data']['mime_type'],
                            'data': base64.b64encode(part['inline_data']['data']).decode('utf-8')
                        }
                    except Exception as e:
                        print(f"警告: キャッシュBase64エンコード失敗: {e}, スキップします。 Part: {part}")
                        continue # エラーがあっても他のパーツは処理
                else: pass # textでもbytesのinline_dataでもないものは無視
                # encoded_partが空でない場合のみ追加
                if encoded_part:
                     encoded_parts.append(encoded_part)

            # 有効なパーツがあり、かつ 'role' が存在する場合のみ履歴に追加
            if encoded_parts and 'role' in entry:
                encoded_history.append({'role': entry['role'], 'parts': encoded_parts})

        # JSONファイルに書き込み
        async with aiofiles.open(cache_file, mode='w', encoding='utf-8') as f:
            await f.write(json.dumps(encoded_history, ensure_ascii=False, indent=2))
        # print(f"キャッシュ保存完了 (Channel: {channel_id}).") # ログ抑制

    except Exception as e:
        print(f"エラー: キャッシュ {cache_file} 書き込み失敗: {e}")


# --- Deep Cache ---

async def load_deep_cache(channel_id: int) -> Optional[str]:
    """指定チャンネルのDeep Cacheを読み込む"""
    deep_cache_file = os.path.join(config.DEEP_CACHE_DIR, f"{channel_id}.json")
    if not os.path.exists(deep_cache_file): return None
    try:
        async with aiofiles.open(deep_cache_file, mode='r', encoding='utf-8') as f:
            content = await f.read()
            if not content: return None
            data = json.loads(content)
            return data.get("summary") # summaryキーの値（文字列）を返す
    except json.JSONDecodeError:
        print(f"警告: Deep Cache {deep_cache_file} が壊れています。リセットします。")
        await save_deep_cache(channel_id, None) # 壊れたファイルを空にする
        return None
    except Exception as e:
        print(f"エラー: Deep Cache {deep_cache_file} 読み込み失敗: {e}")
        return None

async def save_deep_cache(channel_id: int, summary: Optional[str]):
    """指定チャンネルのDeep Cacheを保存する"""
    ensure_cache_directories() # ディレクトリ確認
    deep_cache_file = os.path.join(config.DEEP_CACHE_DIR, f"{channel_id}.json")
    try:
        data_to_save = {"summary": summary if summary else ""} # Noneなら空文字を保存
        async with aiofiles.open(deep_cache_file, mode='w', encoding='utf-8') as f:
            await f.write(json.dumps(data_to_save, ensure_ascii=False, indent=2))
        # print(f"Deep Cache 保存完了 (Channel: {channel_id}).") # ログ抑制
    except Exception as e:
        print(f"エラー: Deep Cache {deep_cache_file} 書き込み失敗: {e}")

def _format_history_for_prompt(history: List[Dict[str, Any]]) -> str:
    """会話履歴リストをプロンプト用のテキスト形式に変換する"""
    formatted_lines = []
    for entry in history:
        role = entry.get('role', 'unknown').capitalize()
        text_parts = []
        # 'parts' が存在し、リストであることを確認
        if 'parts' not in entry or not isinstance(entry.get('parts'), list): continue
        for part in entry['parts']:
            if 'text' in part: text_parts.append(part['text'])
            # inline_data の場合はファイル添付を示す文字列を追加
            elif 'inline_data' in part:
                 mime_type = part['inline_data'].get('mime_type', 'ファイル')
                 # data の存在チェックは不要、mime_typeがあれば添付とみなす
                 text_parts.append(f"[{mime_type} 添付]")

        content = " ".join(text_parts).strip()
        if content:
            max_len = 500 # プロンプトに含める各エントリの最大長
            if len(content) > max_len: content = content[:max_len] + "..."
            formatted_lines.append(f"{role}: {content}")
    return "\n".join(formatted_lines)

async def update_deep_cache(channel_id: int, old_history: List[Dict[str, Any]]):
    """Deep Cacheを更新する (低負荷LLMを使用)"""
    # LLMハンドラーと低負荷モデルの存在確認
    llm_handler = llm_manager.get_current_provider()
    if not llm_handler:
        print(f"Deep Cache更新スキップ (Channel: {channel_id}): LLMハンドラー未初期化。")
        return
    lowload_model_name = llm_handler.get_model_name('lowload')
    if not lowload_model_name:
        provider_name = llm_manager.get_current_provider_name()
        print(f"Deep Cache更新スキップ (Channel: {channel_id}): 低負荷モデル利用不可 ({provider_name})。")
        return

    print(f"Deep Cache更新開始 (Channel: {channel_id}, Model: {lowload_model_name})...")

    # 履歴をテキスト形式に変換
    history_text = _format_history_for_prompt(old_history)
    if not history_text.strip():
        print(f"Deep Cache更新スキップ (Channel: {channel_id}): 抽出対象テキストなし。")
        return

    # 1. 新しい情報の抽出
    extract_prompt = config.DEEP_CACHE_EXTRACT_PROMPT.format(history_text=history_text)
    extracted_summary_raw = await llm_manager.generate_lowload_response(extract_prompt)
    extracted_summary = str(extracted_summary_raw) if extracted_summary_raw else "" # Noneなら空文字

    # エラーまたは「抽出情報なし」の場合
    if not extracted_summary.strip() or "抽出情報なし" in extracted_summary or llm_manager.is_error_message(extracted_summary):
        print(f"Deep Cache更新: 新情報抽出失敗/空/エラー (Channel: {channel_id})。既存キャッシュ維持。")
        if extracted_summary: print(f"  -> Response: {extracted_summary[:100]}...")
        return
    print(f"Deep Cache: 新情報抽出:\n{extracted_summary[:300]}...")

    # 2. 既存のDeep Cacheと統合
    existing_summary = await load_deep_cache(channel_id)
    final_summary = extracted_summary # デフォルトは抽出した新情報

    if existing_summary and existing_summary.strip():
        print("Deep Cache: 既存情報と新情報を統合...")
        merge_prompt = config.DEEP_CACHE_MERGE_PROMPT.format(
            existing_summary=existing_summary, new_summary=extracted_summary
        )
        merged_summary_raw = await llm_manager.generate_lowload_response(merge_prompt)
        merged_summary = str(merged_summary_raw) if merged_summary_raw else ""

        # 統合成功時のみ final_summary を更新
        if merged_summary.strip() and not llm_manager.is_error_message(merged_summary):
            final_summary = merged_summary
            print(f"Deep Cache: 統合後情報生成:\n{final_summary[:300]}...")
        else:
            print("Deep Cache更新警告: 統合失敗/エラー。新情報のみ使用。")
            if merged_summary: print(f"  -> Response: {merged_summary[:100]}...")
            # final_summary は extracted_summary のまま
    else:
        print("Deep Cache: 既存情報なし。抽出情報をそのまま使用。")
        # final_summary は extracted_summary のまま

    # 3. Deep Cacheを保存
    await save_deep_cache(channel_id, final_summary)
    print(f"Deep Cache更新完了 (Channel: {channel_id})")

async def summarize_deep_cache(channel_id: int) -> tuple[bool, str]:
    """Deep Cacheを整理・要約する (低負荷LLMを使用)"""
    # LLMハンドラーと低負荷モデルの存在確認
    llm_handler = llm_manager.get_current_provider()
    if not llm_handler:
        return False, bot_constants.ERROR_MSG_INTERNAL + " (LLM未初期化)"
    lowload_model_name = llm_handler.get_model_name('lowload')
    provider_name = llm_manager.get_current_provider_name()
    if not lowload_model_name:
        return False, bot_constants.ERROR_MSG_LOWLOAD_UNAVAILABLE + f" ({provider_name})"

    print(f"Deep Cache 整理開始 (!csum, Channel: {channel_id}, Model: {lowload_model_name})...")

    # 既存のDeep Cacheを読み込み
    existing_summary = await load_deep_cache(channel_id)
    if not existing_summary or not existing_summary.strip():
        print("Deep Cache 整理スキップ: 対象データなし。")
        return False, "長期記憶(Deep Cache)には現在何も記録されていません。"

    # 整理プロンプトを実行
    summarize_prompt = config.DEEP_CACHE_SUMMARIZE_PROMPT.format(summary_to_clean=existing_summary)
    cleaned_summary_raw = await llm_manager.generate_lowload_response(summarize_prompt)
    cleaned_summary = str(cleaned_summary_raw) if cleaned_summary_raw else ""

    # 結果の検証と保存
    if not cleaned_summary.strip() or llm_manager.is_error_message(cleaned_summary):
        print("Deep Cache 整理失敗: 低負荷モデルから有効な整理結果が得られませんでした。")
        if cleaned_summary: print(f"  -> Response: {cleaned_summary[:100]}...")
        return False, bot_constants.ERROR_MSG_DEEP_CACHE_FAIL + " 整理に失敗しました。"

    await save_deep_cache(channel_id, cleaned_summary)
    print(f"Deep Cache 整理完了 (Channel: {channel_id})。")

    return True, f"長期記憶の整理が完了しました。({provider_name}使用)\n```\n{cleaned_summary}\n```"