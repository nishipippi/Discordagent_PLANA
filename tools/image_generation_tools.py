import base64
import io
import os
from typing import Type

from langchain_core.messages import AIMessage, BaseMessage
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool, StructuredTool
from langchain_google_genai import ChatGoogleGenerativeAI

from llm_config import get_google_api_key

class ImageGenerationInput(BaseModel):
    """Input for image generation tool."""
    prompt: str = Field(description="The detailed prompt for image generation.")

async def _image_generation_func(prompt: str) -> str:
    """Generate an image asynchronously."""
    try:
        api_key = get_google_api_key()
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set in environment variables.")

        image_model_name = os.getenv("GEMINI_IMAGE_MODEL", "gemini-2.0-flash-preview-image-generation")
        llm = ChatGoogleGenerativeAI(model=image_model_name, google_api_key=api_key)

        message = {
            "role": "user",
            "content": prompt,
        }

        response: BaseMessage = await llm.ainvoke(
            [message],
            generation_config=dict(response_modalities=["TEXT", "IMAGE"]),
        )

        def _get_image_base64(response: BaseMessage) -> str:
            # AIMessageのcontentはリスト形式で、辞書やAIMessageChunkを含む可能性がある
            # image_urlはAIMessageChunkのadditional_kwargsに含まれる場合がある
            if isinstance(response, AIMessage):
                for block in response.content:
                    if isinstance(block, dict) and block.get("image_url"):
                        return block["image_url"].get("url").split(",")[-1]
            # AIMessageChunkの場合の処理も考慮に入れる
            if hasattr(response, 'additional_kwargs') and 'image_url' in response.additional_kwargs:
                return response.additional_kwargs['image_url'].get('url').split(',')[-1]
            
            raise ValueError("No image URL found in the AI message response.")

        image_base64 = _get_image_base64(response)
        return f"image_base64_data::{image_base64}" # プレフィックスを付けて返す
    except Exception as e:
        return f"Error generating image: {e}" # エラー時はプレフィックスなし

image_generation_tool = StructuredTool.from_function(
    func=_image_generation_func,
    name="image_generation_tool",
    description="Generates an image based on a detailed text prompt using Google Gemini. Returns the base64 encoded image data.",
    args_schema=ImageGenerationInput,  # type: ignore[arg-type]
    coroutine=_image_generation_func,
)
