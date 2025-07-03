import base64
import io
import os
import json
import requests
import time
from typing import Type

from langchain_core.messages import AIMessage, BaseMessage
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool, StructuredTool
from langchain_google_genai import ChatGoogleGenerativeAI

from llm_config import get_google_api_key, get_comfyui_url

class ImageGenerationInput(BaseModel):
    """Input for image generation tool."""
    positive_prompt: str = Field(description="The detailed positive prompt for image generation.")
    negative_prompt: str = Field(description="The detailed negative prompt for image generation.")
    workflow_file: str = Field(description="The name of the ComfyUI workflow JSON file to use (e.g., 'default_workflow.json').")

def get_image_from_comfyui(positive_prompt: str, negative_prompt: str, workflow_file: str) -> bytes:
    """
    ComfyUI APIを使用して画像を生成し、そのバイナリデータを返す。
    """
    # プリセットのネガティブプロンプト
    preset_negative_prompt = "bad anatomy, low quality, worst quality, deformed, disfigured, ugly"
    
    # ユーザー指定のネガティブプロンプトとプリセットを結合
    combined_negative_prompt = f"{preset_negative_prompt}, {negative_prompt}" if negative_prompt else preset_negative_prompt
    # --- ComfyUIサーバーが稼働している環境での実装予定 ---
    # 現状は開発環境の制約により、ComfyUIサーバーを起動できないため、
    # 以下のロジックは常にダミーデータを返すものとしています。
    # 実際のComfyUIサーバーが稼働する環境になった際には、
    # 以下のコメントアウトされたロジックを参考に実装してください。

    # comfyui_url = get_comfyui_url()
    # api_url = f"{comfyui_url}/prompt"

    # # ワークフローJSONを読み込む (workflowsディレクトリから)
    # workflow_path = os.path.join("workflows", workflow_file)
    # if not os.path.exists(workflow_path):
    #     raise FileNotFoundError(f"Workflow file not found: {workflow_path}")
    # with open(workflow_path, "r") as f:
    #     workflow_json = json.load(f)

    # # プロンプトとネガティブプロンプトをワークフローに設定
    # workflow_json["6"]["inputs"]["text"] = positive_prompt
    # workflow_json["7"]["inputs"]["text"] = combined_negative_prompt

    # # 画像生成リクエストを送信
    # response = requests.post(api_url, json={"prompt": workflow_json})
    # response.raise_for_status()
    # prompt_id = response.json()["prompt_id"]

    # # 画像生成の完了を待機
    # # ComfyUIのAPIでは、/historyエンドポイントでプロンプトIDの状態を確認できる
    # # ただし、ここでは簡略化のため、一定時間待機後に画像を直接取得する
    # # 実際の運用では、WebSocketなどを使用して生成完了を待つのが望ましい
    # time.sleep(10) # 10秒待機

    # # 生成された画像を取得 (ComfyUIのoutputフォルダから取得することを想定)
    # # 実際のAPIでは、/view?filename=... のようなエンドポイントで取得する
    # # ここでは、簡略化のためダミーの画像データを返す
    # --- ComfyUIサーバーが稼働している環境での実装予定 ---
    # 現状は開発環境の制約により、ComfyUIサーバーを起動できないため、
    # 以下のロジックはダミーデータを返すものとしています。
    # 実際のComfyUIサーバーが稼働する環境になった際には、
    # 以下のコメントアウトされたロジックを参考に実装してください。

    # # 1. ComfyUIのAPIエンドポイントにプロンプトを送信
    # response = requests.post(api_url, json={"prompt": workflow_json})
    # response.raise_for_status()
    # prompt_id = response.json()["prompt_id"]

    # # 2. 生成完了までポーリングで待機
    # # WebSocket API (ws://comfyui_url/ws?clientId=...) を使用するのが理想的ですが、
    # # ここでは簡易的にhistory APIをポーリングします。
    # max_retries = 30
    # retry_delay = 1 # seconds
    # for _ in range(max_retries):
    #     history_url = f"{comfyui_url}/history?prompt_id={prompt_id}"
    #     history_response = requests.get(history_url)
    #     history_response.raise_for_status()
    #     history_data = history_response.json()

    #     if prompt_id in history_data and history_data[prompt_id].get("outputs"):
    #         # 生成された画像情報を取得
    #         for node_id, node_output in history_data[prompt_id]["outputs"].items():
    #             if "images" in node_output:
    #                 for img_info in node_output["images"]:
    #                     filename = img_info["filename"]
    #                     file_type = img_info["type"]
    #                     subfolder = img_info["subfolder"]
                        
    #                     # 3. 画像データを取得
    #                     image_url = f"{comfyui_url}/view?filename={filename}&type={file_type}&subfolder={subfolder}"
    #                     image_response = requests.get(image_url)
    #                     image_response.raise_for_status()
    #                     return image_response.content
    #     time.sleep(retry_delay)
    # raise ValueError(f"Image generation timed out or image not found for prompt_id: {prompt_id}")

    # --- ここからダミーデータ返却ロジック ---
    # 仮のダミー画像データ (Base64デコード可能なPNGの最小データ)

    # 仮のダミー画像データ (Base64デコード可能なPNGの最小データ)
    dummy_image_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
    return base64.b64decode(dummy_image_base64)


async def _image_generation_func(positive_prompt: str, negative_prompt: str, workflow_file: str) -> str:
    """Generate an image asynchronously using ComfyUI."""
    try:
        image_data = get_image_from_comfyui(positive_prompt, negative_prompt, workflow_file)
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        return f"image_base64_data::{image_base64}"
    except Exception as e:
        return f"Error generating image with ComfyUI: {e}"

image_generation_tool = StructuredTool.from_function(
    func=_image_generation_func,
    name="image_generation_tool",
    description="Generates an image based on detailed positive and negative text prompts using ComfyUI, with a specified workflow. Returns the base64 encoded image data.",
    args_schema=ImageGenerationInput,  # type: ignore[arg-type]
    coroutine=_image_generation_func,
)
