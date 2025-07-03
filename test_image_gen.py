import asyncio
import os
from tools.image_generation_tools import _image_generation_func
from dotenv import load_dotenv

# 環境変数をロード
load_dotenv()

async def test_image_generation_comfyui():
    print("--- Running ComfyUI Image Generation Test ---")
    positive_prompt = "a cat on a sofa"
    negative_prompt = "bad anatomy, low quality"
    workflow_file = "default_workflow.json"
    print(f"Testing with positive_prompt: '{positive_prompt}', negative_prompt: '{negative_prompt}', workflow_file: '{workflow_file}'")

    # _image_generation_func は非同期関数なので await で実行
    result = await _image_generation_func(positive_prompt=positive_prompt, negative_prompt=negative_prompt, workflow_file=workflow_file)

    print(f"Result: {result[:100]}...") # 結果の先頭100文字を表示

    assert result.startswith("image_base64_data::")
    print("Test Passed: Result starts with 'image_base64_data::'")

    # ここではComfyUIが実際に動いているかまではテストしない
    # ダミーデータが返ってくることを確認
    assert "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=" in result
    print("Test Passed: Dummy image data found in result.")

if __name__ == "__main__":
    # 非同期関数を実行
    asyncio.run(test_image_generation_comfyui())
