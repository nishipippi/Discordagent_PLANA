# Discord AIエージェント (LangChain & LangGraph版)

## 1. 概要

このプロジェクトは、Discord上で動作するAIエージェントです。ユーザーの自然言語による指示を理解し、情報検索、雑談、タイマー設定、画像生成、情報記憶・想起、PDF/画像ファイルの内容理解といった多様なタスクを、LangChainおよびLangGraphフレームワークを活用して自律的に実行します。Discordサーバーにおけるユーザー体験の向上とコミュニケーションの活性化を目指します。

## 2. 主な機能

本エージェントは以下の主要な機能を備えています。

*   **自然言語理解とメンションによる起動**: ユーザーからの自然言語での指示を理解し、メンションによって起動します。
*   **自律的な検索と応答**: 必要に応じてBrave Search APIを利用して情報を検索し、結果を要約して応答します。
*   **タイマー・アラーム機能**: 指定された時間に通知を行うタイマーやアラームを設定できます。
*   **画像生成機能**: 指示に基づいて画像を生成し、Discordメッセージとして投稿します。**この機能は、ローカルで動作するComfyUIサーバーと連携して提供されます。**
*   **記憶・想起機能**: 「覚えておいて」といった指示で情報を記憶し、会話の文脈に応じて想起して応答に活用します。
*   **PDF・画像の読み込みと理解**: 添付されたPDFや画像ファイルの内容を理解し、関連する質問に答えます。
*   **フォローアップ質問の提示**: 応答後、文脈に沿ったフォローアップ質問を提示し、ユーザーのインタラクションを促します。

## 3. 技術スタック

本プロジェクトで使用されている主な技術は以下の通りです。

*   **プログラミング言語**: Python
*   **主要ライブラリ・フレームワーク**:
    *   discord.py
    *   langchain
    *   langgraph
    *   langchain-google-genai
    *   google-genai (Gemini利用時)
    *   requests (Brave Search API直接コール時)
    *   ベクトルストアライブラリ (FAISS)
    *   データベース操作ライブラリ (SQLite3)
    *   langchain-discord-shikenso
*   **AIモデル**:
    *   Google Gemini
    *   **ComfyUI (ローカル画像生成)**
*   **外部API**:
    *   Brave Search API
*   **データベース**:
    *   SQLite
    *   ベクトルストア (FAISS)

## 4. セットアップと実行方法

### 4.1. 前提条件

*   Python 3.9 以降
*   各種APIキー (詳細は下記)
*   **ComfyUIのセットアップと起動** (画像生成機能を利用する場合)

### 4.2. インストール

1.  リポジトリをクローンします。
    ```bash
    git clone https://github.com/nishipippi/Discordagent_PLANA
    cd Discordagent_PLANA
    ```
2.  必要なAPIキーを設定します。`.env.template` をコピーして `.env` ファイルを作成し、以下の情報を記述してください。
    *   `DISCORD_BOT_TOKEN`: Discord Botのトークン
    *    `GOOGLE_API_KEY`: LLMのAPIキー
    *   `BRAVE_API_KEY`: Brave Search APIのキー
    *   **`COMFYUI_URL`**: ComfyUIサーバーのURL (例: `http://127.0.0.1:8188`)
    *   その他、必要なAPIキーや設定値

3.  依存ライブラリをインストールします。
    ```bash
    pip install -r requirements.txt
    ```

### 4.3. 実行

以下のコマンドでAIエージェントを起動します。

```bash
python bot.py
```

## 5. ComfyUI連携について

本エージェントの画像生成機能は、ローカルで動作する[ComfyUI](https://github.com/comfyanonymous/ComfyUI)と連携しています。

*   **ComfyUIのセットアップ**: ComfyUIをローカル環境にセットアップし、起動しておく必要があります。詳細なセットアップ手順はComfyUIの公式リポジトリを参照してください。
*   **ワークフロー**: `comfy_api_workflow.json` ファイルに、画像生成に使用するComfyUIのAPIワークフローが定義されています。必要に応じてこのワークフローをカスタマイズしてください。
*   **現在の開発環境における注意点**:
    *   このエージェントの開発フェーズでは、ComfyUIサーバーが稼働していない環境で作業を進めています。
    *   そのため、`tools/image_generation_tools.py` 内の `get_image_from_comfyui` 関数は、現在ダミーの画像データを返すように実装されており、実際のComfyUIとの連携は行っていません。
    *   ComfyUIサーバーが利用可能な環境でこの機能を動作させるためには、`get_image_from_comfyui` 関数内のコメント化された部分（実際の画像取得ロジック）を実装する必要があります。

## 7. ライセンス

このプロジェクトは Apache License 2.0 の下で公開されています。詳細については、リポジトリ内の `LICENSE` ファイル（もしあれば）または [Apache License 2.0 の公式ページ](https://www.apache.org/licenses/LICENSE-2.0) を参照してください。
