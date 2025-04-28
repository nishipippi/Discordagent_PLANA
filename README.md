# LLM Discord Bot (PLANA)

Gemini API または OpenAI互換API (Mistralなど) を使用した、Discord向けの多機能チャットボットです。ブルーアーカイブのユーザーアシスタント（プラナ）での応答、画像・PDF・テキストファイルへの対応、会話履歴のキャッシュ、長期記憶 (Deep Cache)、ウェブ検索機能、追跡質問ボタン、タイマー、投票機能などを備えています。

## 特徴 (Features)

*   **マルチLLMプロバイダー対応:** `.env` ファイルの設定により、Google Gemini API または OpenAI互換API (Mistral Pixtralなど) を切り替えて使用できます。
*   **ブルーアーカイブのユーザーアシスタントのペルソナ:** タブレット上の電子生命体「プラナ」として、無機質で冷静、やや毒舌ながらもユーザーに寄り添った口調で応答します。
*   **マルチモーダル対応:** 画像、PDF、テキストファイルを添付して質問できます。（対応形式は使用するLLMモデルにも依存します）
*   **会話履歴のキャッシュ:** チャンネルごとに直近の会話履歴をキャッシュし、文脈を考慮した応答を生成します。
*   **長期記憶 (Deep Cache):** キャッシュから溢れた古い会話から重要な情報を抽出し、長期記憶として保持・参照します (`!csum`で確認、`!cclear`で削除)。
*   **ウェブ検索機能:** Brave Search API と連携し、`!src` (高速検索) または `!dsrc` (詳細検索) コマンドでウェブ検索結果に基づいた応答を生成します。
*   **追跡質問ボタン:** Botの応答メッセージに、次の質問や関連情報への興味を引くボタンを動的に生成・追加します。
*   **タイマー機能:** `!timer` コマンドで指定した時間後にメッセージでお知らせします。
*   **投票機能:** `!poll` コマンドで簡単な投票を作成できます。
*   **プロバイダー切り替え:** `!gemini` / `!mistral` コマンドで実行中にLLMプロバイダーを切り替えられます。

## ファイル構成 (File Structure)

```
/PLANA
├── bot.py                        # メインのBotロジック、Discordクライアント、イベントハンドラ
├── config.py                     # 設定値（APIキー以外）、プロンプトテンプレート管理
├── llm_manager.py                # LLMプロバイダーの初期化、管理、切り替え
├── llm_provider.py               # LLM APIプロバイダーの共通インターフェース定義
├── gemini_provider.py            # Google Gemini API用のプロバイダー実装
├── openai_compatible_provider.py # OpenAI互換API (Mistral等) 用のプロバイダー実装
├── cache_manager.py              # 通常キャッシュとDeep Cacheの管理
├── command_handler.py            # 通常コマンド (!timer, !poll等) とメンション応答処理
├── search_handler.py             # 検索コマンド (!src, !dsrc) の処理、Web連携
├── discord_ui.py                 # Discord UI要素 (ボタンビュー、思考中メッセージ)
├── bot_constants.py              # ペルソナ定義、エラーメッセージなどの定数
├── requirements.txt              # 依存ライブラリリスト
├── .env.example                  # 環境変数設定の例 (コピーして.envを作成)
├── cache/                        # 会話キャッシュ保存用ディレクトリ (自動生成)
├── deep_cache/                   # 長期記憶 (Deep Cache) 保存用ディレクトリ (自動生成)
└── README.md                     # このファイル
```

## セットアップ方法 (Setup)

### 必要なもの

*   Python 3.9+ (推奨)
*   Discord Bot アカウントとトークン
    *   [Discord Developer Portal](https://discord.com/developers/applications) でアプリケーションを作成し、Botを追加してください。
    *   `TOKEN` をコピーしてください。
    *   **OAuth2 -> URL Generator:** `bot` スコープを選択し、**Permissions** で以下の権限を付与してください:
        *   `Send Messages`
        *   `Read Message History`
        *   `Add Reactions`
        *   `Use External Emojis` (Optional)
        *   `Embed Links` (Optional, for future features)
        *   `Attach Files` (Optional, for future features)
        *   `Read Messages/View Channels`
    *   生成されたURLでBotをサーバーに招待してください。
    *   **Bot タブ:** **Privileged Gateway Intents** で **`Message Content Intent`** と **`Server Members Intent`** を必ず **ON** にしてください。これらが無いとメッセージ内容やメンバー情報を取得できません。
*   LLM API キー
    *   Google Gemini API: [Google AI Studio](https://aistudio.google.com/) または [Google Cloud](https://cloud.google.com/vertex-ai) でAPIキーを取得。
    *   Mistral API (OpenAI互換エンドポイント): [Mistral AI Platform](https://console.mistral.ai/) でAPIキーを取得。
*   Brave Search API キー (検索機能を利用する場合)
    *   [Brave Search API](https://brave.com/search/api/) で無料または有料プランに登録し、APIキーを取得。

### 手順

1.  **リポジトリをクローンします:**

    ```bash
    git clone <このリポジトリのURL>
    cd PLANA
    ```

2.  **仮想環境を作成し、アクティベートします:** (強く推奨)

    ```bash
    # 仮想環境作成 (例: venv という名前で作成)
    python -m venv venv

    # 仮想環境をアクティベート
    # Windows PowerShell:
    .\venv\Scripts\Activate.ps1
    # Windows Command Prompt:
    .\venv\Scripts\activate.bat
    # macOS/Linux (Bash, Zsh):
    source venv/bin/activate
    ```
    プロンプトの先頭に `(venv)` のように仮想環境名が表示されれば成功です。

3.  **必要なライブラリをインストールします:**
    仮想環境がアクティベートされた状態で、以下のコマンドを実行します。

    ```bash
    pip install -r requirements.txt
    ```
    これにより、`discord.py`, `google-generativeai`, `openai`, `python-dotenv`, `aiofiles`, `httpx`, `PyPDF2` など、必要なライブラリがすべてインストールされます。

4.  **.env ファイルを設定します:**
    `.env.example` ファイルをコピーして `.env` という名前で保存し、エディタで開いてください。

    ```bash
    copy .env.example .env # Windows
    # or
    cp .env.example .env # macOS/Linux
    ```

    `.env` ファイルを編集し、各項目を設定します。

    ```dotenv
    # Discord Bot Token (必須)
    DISCORD_TOKEN=YOUR_DISCORD_BOT_TOKEN

    # --- LLM プロバイダー設定 ---
    # 初期状態で使用するLLMプロバイダー ('GEMINI' または 'MISTRAL')
    LLM_PROVIDER=GEMINI

    # Google Gemini API 設定 (LLM_PROVIDER=GEMINI を初期選択する場合、または !gemini で使用する場合)
    GEMINI_API_KEY=YOUR_GEMINI_API_KEY
    # 使用するモデル名を指定 (例)
    GEMINI_PRIMARY_MODEL=gemini-1.5-pro-latest
    GEMINI_SECONDARY_MODEL=gemini-1.5-flash-latest # Primaryがレートリミット時に使用
    GEMINI_LOWLOAD_MODEL=gemini-1.5-flash-latest   # DeepCache更新、ボタン生成、!src検索に使用

    # Mistral API (OpenAI互換) 設定 (LLM_PROVIDER=MISTRAL を初期選択する場合、または !mistral で使用する場合)
    MISTRAL_API_KEY=YOUR_MISTRAL_API_KEY
    MISTRAL_API_BASE_URL=https://api.mistral.ai/v1 # 通常はこのままでOK
    # 使用するモデル名を指定 (例)
    MISTRAL_PRIMARY_MODEL=mistral-large-latest     # または 画像対応の pixtral-large-latest など
    MISTRAL_SECONDARY_MODEL=mistral-large-latest   # Primaryがレートリミット時に使用 (Primaryと同じでも可)
    MISTRAL_LOWLOAD_MODEL=mistral-small-latest     # DeepCache更新、ボタン生成、!src検索に使用

    # --- 検索機能設定 ---
    # Brave Search API キー (!src, !dsrc を利用する場合に必要)
    BRAVE_SEARCH_API_KEY=YOUR_BRAVE_SEARCH_API_KEY
    ```
    **注意:**
    *   最低限 `DISCORD_TOKEN` と、`LLM_PROVIDER` で選択したプロバイダーの `API_KEY` は設定 **必須** です。
    *   検索機能を使わない場合は `BRAVE_SEARCH_API_KEY` は空でもBotは起動しますが、`!src`/`!dsrc` コマンドは使用できません。
    *   モデル名は、利用可能な最新のモデルや、用途に合わせて適宜変更してください。
    *   `.env` ファイルは `.gitignore` に含まれていることを確認し、GitHubなどにコミットしないようにしてください。

5.  **Bot を実行します:**
    仮想環境がアクティベートされた状態で、Botを起動します。

    ```bash
    python bot.py
    ```
    コンソールにログメッセージが表示され、`Bot is ready!` と表示されれば起動成功です。初回起動時に `cache` と `deep_cache` ディレクトリが自動生成されます。

## 使い方 (Usage)

Botが起動し、Discordサーバーに参加したら、以下の方法でBotと対話できます。

*   **会話:** Botをメンション (`@YourBotName`) してメッセージを送信します。Botが応答を生成します。
*   **ファイル添付:** メンションメッセージに画像、PDF、テキストファイルなどを添付すると、その内容も考慮して応答します。（対応形式は使用するLLMモデルにも依存。PDFはテキスト抽出されます）
*   **検索:**
    *   `@YourBotName !src [検索キーワード]` : 高速検索。ウェブ検索結果（上位数件）を基に低負荷モデルで応答します。
    *   `@YourBotName !dsrc [検索キーワード]` : 詳細検索。ウェブ検索と評価を繰り返し（最大3回）、より深く情報を掘り下げて高負荷モデルで応答します。Brave APIとLLM APIの消費が多くなります。
*   **追跡質問ボタン:** Botの応答メッセージの下に表示されるボタンをクリックすると、その内容を基に追加で質問や対話ができます。
*   **履歴参照:** メンションメッセージに `!his` を含めると、キャッシュを無視してチャンネルの直近履歴を再取得して応答します。（例: `@YourBotName !his このチャンネルのログについて教えて`）
*   **プロバイダー切替:**
    *   `!gemini`: LLMプロバイダーを Gemini に切り替えます。
    *   `!mistral`: LLMプロバイダーを Mistral (OpenAI互換) に切り替えます。
*   **タイマー:** `!timer X分 [内容]` の形式でメッセージを送信すると、X分後にBotが指定した内容でお知らせします。（例: `!timer 30分 休憩終了`）
*   **投票:** `!poll "質問内容" 選択肢1 選択肢2 ...` または `!poll 質問内容 選択肢1 選択肢2 ...` の形式でメッセージを送信すると、投票が作成され、リアクションで投票できるようになります。（例: `!poll "今日のランチ" カレー ラーメン パスタ`）
*   **長期記憶 (Deep Cache) 操作:**
    *   `!csum`: Botが覚えている長期記憶の内容を整理し、表示します。
    *   `!cclear`: Botの長期記憶をすべて削除します。

## カスタマイズ (Customization)

*   **ペルソナ:** `bot_constants.py` ファイルの `PERSONA_TEMPLATE` を編集することで、Botの基本的な口調や設定を変更できます。
*   **LLMモデル:** `.env` ファイルで `*_PRIMARY_MODEL`, `*_SECONDARY_MODEL`, `*_LOWLOAD_MODEL` の値を変更することで、使用するLLMモデルを調整できます。
*   **検索設定:** `config.py` ファイル内の定数を変更することで、検索関連の挙動を調整できます。
    *   `MAX_SEARCH_RESULTS`: 一度の検索で取得する最大結果数。
    *   `MAX_CONTENT_LENGTH_PER_URL`: 各Webページから抽出する最大文字数。
    *   `MAX_TOTAL_SEARCH_CONTENT_LENGTH`: LLMに渡す検索コンテンツ合計の最大文字数。
    *   `DEEP_SEARCH_MAX_ITERATIONS`: `!dsrc` での最大検索繰り返し回数。
    *   `SEARCH_MIN_CONTENT_LENGTH`: Webページから抽出する最小文字数閾値。
*   **キャッシュ設定:** `config.py` の `CACHE_LIMIT`, `HISTORY_LIMIT` を変更。
*   **ボタン設定:** `config.py` の `MAX_FOLLOW_UP_BUTTONS`, `FOLLOW_UP_BUTTON_TIMEOUT` を変更。
*   **プロンプトテンプレート:** `config.py` 内の各種 `*_PROMPT` 定数を編集することで、検索クエリ生成、最終応答生成、Deep Cache処理などの指示内容を調整できます。
*   **エラーメッセージ:** `bot_constants.py` ファイルの `ERROR_MSG_...` 定数を編集。
*   **ファイルサイズ制限:** `command_handler.py` 内の `FILE_LIMIT_MB` を変更。

## ライセンス (License)

MIT License (このリポジトリに含まれるコードに適用されます)

## 免責事項 (Disclaimer)

*   LLM APIおよびBrave Search APIの利用には料金が発生する場合があります。各サービスの料金体系を確認してください。特に `!dsrc` はAPIコールが多くなる可能性があります。
*   LLMの出力内容はモデルの性質上、常に正確であるとは限りません。
*   本Botの使用により生じたいかなる損害についても、作者は責任を負いません。自己責任でご利用ください。

## 謝辞 (Acknowledgements)

*   discord.py (Discord API Wrapper)
*   google-generativeai (Gemini API SDK)
*   openai (OpenAI API SDK)
*   python-dotenv (環境変数管理)
*   aiofiles (非同期ファイルI/O)
*   httpx (HTTPクライアント)
*   PyPDF2 (PDFテキスト抽出)
*   貢献者: Mr.coffin399 (https://github.com/coffin399)

---
