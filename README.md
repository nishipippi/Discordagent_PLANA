# LLM Discord Bot

Gemini API または OpenAI互換API (Mistralなど) を使用した、Discord向けの多機能チャットボットです。特定のペルソナ（プラナ）での応答、ファイル添付への対応、会話履歴のキャッシュ、長期記憶 (Deep Cache)、追跡質問ボタン、タイマー、投票機能などを備えています。


## 特徴 (Features)

*   **マルチLLMプロバイダー対応:** `.env` ファイルの設定により、Google Gemini API または OpenAI互換API (Mistral Pixtralなど) を切り替えて使用できます。
*   **独自のペルソナ:** タブレット上の電子生命体「プラナ」として、無機質で冷静、やや毒舌ながらもユーザーに寄り添った口調で応答します。
*   **マルチモーダル対応:** 画像ファイルやテキストファイルを添付して質問できます。（対応形式は使用するLLMモデルに依存します）
*   **会話履歴のキャッシュ:** チャンネルごとに直近の会話履歴をキャッシュし、文脈を考慮した応答を生成します。
*   **長期記憶 (Deep Cache):** キャッシュから溢れた古い会話から重要な情報を抽出し、長期記憶として保持・参照します。
*   **追跡質問ボタン:** Botの応答メッセージに、次の質問や関連情報への興味を引くボタンを動的に生成・追加します。
*   **タイマー機能:** 指定した時間後にメッセージでお知らせします。
*   **投票機能:** 簡単なコマンドでサーバーメンバー向けの投票を作成できます。

## ファイル構成 (File Structure)

```
/PLANA
├── bot.py                     # メインのBotロジック、イベントハンドラ
├── llm_provider.py            # LLM APIプロバイダーの共通インターフェース定義
├── gemini_provider.py         # Google Gemini API用のプロバイダー実装
├── openai_compatible_provider.py # OpenAI互換API (Mistral等) 用のプロバイダー実装
├── bot_constants.py           # エラーメッセージなどの定数定義
├── .env.example               # 環境変数設定の例 (コピーして.envを作成)
├── cache/                     # 会話キャッシュ保存用ディレクトリ
├── deep_cache/                # 長期記憶 (Deep Cache) 保存用ディレクトリ
└── README.md                  # このファイル
```

## セットアップ方法 (Setup)

### 必要なもの

*   Python 3.8+ (推奨)
*   Discord Bot アカウントとトークン
    *   [Discord Developer Portal](https://discord.com/developers/applications) で新しいアプリケーションを作成し、Botを追加してください。
    *   `TOKEN` をコピーしてください。
    *   OAuth2 -> URL Generator で `bot` スコープを選択し、必要な権限（最低限 `Send Messages`, `Read Message History`, `Add Reactions`, `Use External Emojis`, `Embed Links` など）を選択してURLを生成し、サーバーにBotを招待してください。
    *   Botタブの **Privileged Gateway Intents** で **`Message Content Intent`** を必ず**ON**にしてください。ONにしないとメッセージ内容を取得できません。
*   LLM API キー
    *   Google Gemini API: [Google AI Studio](https://aistudio.google.com/) または [Google Cloud](https://cloud.google.com/vertex-ai) でAPIキーを取得してください。
    *   Mistral API (OpenAI互換エンドポイント): [Mistral AI Platform](https://console.mistral.ai/) でAPIキーを取得してください。

### 手順

1.  **リポジトリをクローンします:**

    ```bash
    git clone <このリポジトリのURL>
    cd PLANA
    ```

2.  **仮想環境を作成し、アクティベートします:**
    Pythonの実行環境を分離するため、仮想環境の使用を強く推奨します。

    ```bash
    # 仮想環境作成
    python -m venv discord  # 'discord' は仮想環境名、任意に変更可

    # 仮想環境をアクティベート
    # Windows PowerShell:
    .\discord\Scripts\Activate.ps1

    # Windows Command Prompt:
    .\discord\Scripts\activate.bat

    # macOS/Linux (Bash, Zsh):
    source discord/bin/activate
    ```
    プロンプトの先頭に `(discord)` のように仮想環境名が表示されれば成功です。

3.  **必要なライブラリをインストールします:**
    仮想環境がアクティベートされた状態で、以下のコマンドを実行します。

    ```bash
    pip install aiofiles python-dotenv discord.py google-generativeai openai
    ```

   これでも可能です
   ```bash
    pip install -r requirements.txt
    ```

4.  **.env ファイルを設定します:**
    `.env.example` ファイルをコピーして `.env` という名前で保存し、エディタで開いてください。

    ```bash
    copy .env.example .env # Windows
    # or
    cp .env.example .env # macOS/Linux
    ```

    `.env` ファイルを編集し、各項目を設定します。

    ```dotenv
    # Discord Bot Token
    DISCORD_TOKEN=YOUR_DISCORD_BOT_TOKEN

    # LLM Provider の選択 ('GEMINI' または 'MISTRAL')
    LLM_PROVIDER=GEMINI # または MISTRAL

    # Google Gemini API 設定 (LLM_PROVIDER=GEMINI の場合に使用)
    GEMINI_API_KEY=YOUR_GEMINI_API_KEY
    GEMINI_PRIMARY_MODEL="gemini-1.5-pro-latest"
    GEMINI_SECONDARY_MODEL="gemini-1.5-flash-latest"
    GEMINI_LOWLOAD_MODEL="gemini-1.5-flash-latest"

    # Mistral API (OpenAI互換) 設定 (LLM_PROVIDER=MISTRAL の場合に使用)
    MISTRAL_API_KEY=YOUR_MISTRAL_API_KEY
    MISTRAL_API_BASE_URL=https://api.mistral.ai/v1 # 通常はこのままでOK
    MISTRAL_PRIMARY_MODEL=pixtral-large-latest     # Pixtralを使用する場合
    MISTRAL_SECONDARY_MODEL=mistral-large-latest   # フォールバック用
    MISTRAL_LOWLOAD_MODEL=mistral-small-latest     # 低負荷用
    ```
    **注意:** 使用するLLM Providerに対応するAPIキーとモデル名のみを設定すれば動きますが、将来的な切り替えのために両方の設定を残しておくこともできます。`.env` ファイルはgitignoreに追加し、GitHubにコミットしないように注意してください。

5.  **Bot を実行します:**
    仮想環境がアクティベートされた状態で、Botを起動します。

    ```bash
    python bot.py
    ```
    コンソールにログメッセージが表示され、`Bot is ready!` と表示されれば起動成功です。

## 使い方 (Usage)

Botが起動し、Discordサーバーに参加したら、以下の方法でBotと対話できます。

*   **会話:** Botをメンション (`@YourBotName`) してメッセージを送信します。Botが応答を生成します。
*   **ファイル添付:** メンションメッセージに画像やテキストファイルなどを添付すると、その内容も考慮して応答します。（対応形式は使用するLLMモデルに依存）
*   **追跡質問ボタン:** Botの応答メッセージの下に表示されるボタンをクリックすると、その内容を基に追加で質問や対話ができます。
*   **履歴参照:** メンションメッセージに `!his` を含めると、キャッシュを無視してチャンネルの直近履歴を再取得して応答します。（例: `@YourBotName !his このチャンネルのログについて教えて`）
*   **タイマー:** `!timer X分 [内容]` の形式でメッセージを送信すると、X分後にBotが指定した内容でお知らせします。（例: `!timer 30分 休憩終了`）
*   **投票:** `!poll "質問内容" 選択肢1 選択肢2 ...` または `!poll 質問内容 選択肢1 選択肢2 ...` の形式でメッセージを送信すると、投票が作成され、リアクションで投票できるようになります。（例: `!poll "今日のランチ" カレー ラーメン パスタ`）
*   **Deep Cache 整理:** `!csum` とメッセージを送信すると、Botがこれまでに覚えた長期記憶 (Deep Cache) を整理し、内容を表示します。
*   **Deep Cache クリア:** `!cclear` とメッセージを送信すると、Botの長期記憶 (Deep Cache) をすべて削除します。

## カスタマイズ (Customization)

*   **ペルソナ:** `bot_constants.py` ファイルの `PERSONA_TEMPLATE` を編集することで、Botの基本的な口調や設定を変更できます。
*   **モデル設定:** `.env` ファイルで `LLM_PROVIDER`, `*_PRIMARY_MODEL`, `*_SECONDARY_MODEL`, `*_LOWLOAD_MODEL` の値を変更することで、使用するLLMプロバイダーやモデルを調整できます。
*   **キャッシュ設定:** `bot.py` ファイルの `CACHE_LIMIT`, `HISTORY_LIMIT` などの定数を変更することで、会話キャッシュや履歴取得の振る舞いを調整できます。
*   **エラーメッセージ:** `bot_constants.py` ファイルの `ERROR_MSG_...` で始まる定数を編集することで、各種エラーメッセージの表現を変更できます。
*   **その他の設定:** `bot.py` 内の `MAX_FOLLOW_UP_BUTTONS` (ボタン数), ファイルサイズ制限などの定数を調整可能です。

## ライセンス (License)

[LICENSE](LICENSE) ファイルに記載してください。（MIT Licenseなどを検討してください）

## 免責事項 (Disclaimer)

*   LLM API の利用には料金が発生する場合があります。各プロバイダーの料金体系を確認してください。
*   LLMの出力内容はモデルの性質上、常に正確であるとは限りません。
*   本Botの使用により生じたいかなる損害についても、作者は責任を負いません。

## 謝辞 (Acknowledgements)

*   discord.py (Discord API Wrapper)
*   python-dotenv (環境変数管理)
*   google-generativeai (Gemini API)
*   openai (OpenAI互換 API)
*   aiofiles (非同期ファイルI/O)
*   Mr.coffin399 https://github.com/coffin399

---
```
