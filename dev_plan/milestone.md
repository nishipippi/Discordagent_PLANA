マイルストーン (LangChain & LangGraph版)Discord AIエージェント開発マイルストーンフェーズ０：準備・基盤構築
M0.1: 開発環境構築 - 完了

Python環境設定 (venv を使用) - 完了
必要なライブラリのインストール (discord.py, langchain, langgraph, langchain-openai (または langchain-google-genai 等のLLM連携ライブラリ), google-genai (Gemini利用時), python-dotenv, requests など。brave-search は numpy のコンパイル問題で代替実装) - 完了
Gitリポジトリ作成とバージョン管理設定 (既存リポジトリ使用、.gitignore 更新) - 完了


M0.2: APIキーと認証情報の設定 - 完了

Discord Botトークンの取得と設定 (.env ファイル使用) - 完了
LLM APIキー (OpenAI, Google GenAI等) の設定 (.env ファイル使用) - 完了
Brave Search APIキーの設定 (.env ファイル使用) - 完了


M0.3: 基本的なDiscord Botの疎通確認 - 完了

BotがDiscordサーバーに接続し、メンションに簡単な固定メッセージで応答できることを確認 (bot.py 作成) - 完了
LangChainの基本的なセットアップと動作確認（例: ChatOpenAI または ChatGoogleGenerativeAI を使用し、簡単な応答生成を確認） - 完了
LangGraphの基本的なセットアップと簡単なグラフ実行確認 - 完了


フェーズ１：中核機能の実装 (MVP - Minimum Viable Product) - LangGraphによるステートフルエージェント化
M1.1: 自然言語による指示の理解と雑談応答 (LangChain & LangGraph) - 完了

メンションされた内容をLangChainのChatPromptTemplateとLLM (ChatOpenAI等) を用いて処理し、基本的な雑談応答をLangGraphのノードとして実装。 - 完了
AIのペルソナ（プラナ）を設定ファイルから読み込み、LangChainのプロンプトに反映。 - 完了
LangGraphで基本的な会話状態 (State) を定義。 - 完了


M1.2: 会話履歴の取得とコンテキスト利用 (LangChain Memory & LangGraph State) - 完了

メンションされたチャンネルの直近の会話履歴をlangchain-discordのDiscordReadMessagesツール等で取得し、LangChainのConversationBufferWindowMemory (または同等のメモリコンポーネント) をLangGraphのステートの一部としてユーザーIDごとに管理・永続化する仕組みを実装。 - 完了
履歴と現在の指示の区別を明確化し、LLMへの入力コンテキストに含める。 - 完了

**達成内容:**
*   会話履歴の永続化のために `tools/db_utils.py` を新規作成し、SQLiteデータベース（`data/memory.db`）に履歴を保存・ロードする機能を実装しました。
*   `bot.py` の `fetch_chat_history` ノードと `on_message` イベントを修正し、データベース連携と、ユーザー入力・AI応答の履歴への適切な追加を行いました。
*   `prompts/system_instruction.txt` を更新し、LLMが過去の会話履歴を考慮して応答するように指示を追加しました。
*   `nodes.py` の `call_llm` ノードを修正し、LLMの応答が `AgentState` の `chat_history` に適切に反映されるようにしました。


M1.3: 自律的な検索機能の実装 (LangChain Custom Tool & LangGraph Agent Logic) - 完了

**達成内容:**
*   `tools/brave_search.py` に `requests` を使用したカスタム検索ツール (`BraveSearchTool`) を実装しました。
*   `state.py` の `AgentState` に `search_query`, `search_results`, `should_search_decision` フィールドを追加しました。
*   `nodes.py` に検索要否を判断する `should_search_node` と検索を実行する `execute_search_node` を追加し、`call_llm` ノードを検索結果を利用できるように修正しました。
*   `bot.py` の LangGraph ワークフローを更新し、`fetch_chat_history` -> `should_search` -> (条件分岐) -> `execute_search` -> `call_llm` のフローを組み込みました。
*   `prompts/system_instruction.txt` に検索機能に関する指示を追加しました。
*   Pylanceによる型エラーを解消しました。


M1.4: 記憶・想起機能の実装 (LangChain Tools, VectorStore & LangGraph State/Agent Logic)

M1.4.1: データベース基盤の準備 (SQLite & VectorStore) - 完了

**達成内容:**
*   `tools/db_utils.py` を更新し、構造化された記憶を保存するための `memories` テーブルを SQLite データベース (`data/memory.db`) に追加しました。
*   `tools/vector_store_utils.py` を新規作成し、FAISS と Google Generative AI Embeddings (`models/embedding-001`) を使用したベクトルストア管理クラス (`VectorStoreManager`) を実装しました。ベクトルストアのデータは `data/vector_store/faiss_index_gemini` に永続化されます。
*   `llm_config.py` に `get_google_api_key` 関数を追加し、APIキーの取得方法を統一しました。
*   `bot.py` を更新し、Bot起動時に `setup_hook` を使用して SQLite データベース (`memories` テーブルを含む) と `VectorStoreManager` が初期化されるようにしました。
*   `tools/vector_store_utils.py` で発生していた `ImportError` を、インポートパスを修正することで解決しました。


M1.4.2: 記憶ツール (remember_information) の実装 (LangChain Tool & LangGraph)

tools/memory_tools.py にユーザーの入力テキストをLLM (ChatOpenAI等) を用いてJSON形式に構造化し、SQLiteデータベースに保存する remember_information_func 関数と対応するPydanticモデル (RememberInput) を定義。これをLangChainのToolとしてラップ。
ユーザーが指定した情報をベクトルストアにも埋め込みベクトルとして保存するロジックを追加。
server_id, channel_id, user_id をツールに渡すため、LangGraphのステートまたはエージェント呼び出し時の引数として管理。


M1.4.3: 想起ツール (recall_information) の実装 (LangChain Tool & LangGraph)

tools/memory_tools.py に recall_information_func 関数と対応するPydanticモデル (RecallInput) を定義。
構造化データはSQLiteから、意味的に関連する情報はベクトルストアから検索し、LLMを用いてユーザーの質問に回答を生成するロジックを実装。これをLangChainのToolとしてラップ。


M1.4.4: エージェントへのツール組み込みと動作確認 (LangGraph Agent & LangChain Tools) - **完了**

**達成内容:**
*   `state.py` の `AgentState` を更新し、ツール呼び出しと結果を管理するためのフィールド (`tool_name`, `tool_args`, `tool_output`, `llm_direct_response`) を追加しました。
*   `nodes.py` に以下の主要なノードを実装・更新しました。
    *   `fetch_chat_history`: 会話履歴を取得するノード (botインスタンスへのアクセス方法を調整)。
    *   `decide_tool_or_direct_response_node`: LLMがツール使用または直接応答を判断するノード。LLMの応答（JSON形式を期待）をパースしてツール呼び出し情報や直接応答を `AgentState` に設定します。
    *   `execute_tool_node`: 汎用的なツール実行ノード。`AgentState` からツール名と引数を取得し、対応するツールを実行します。
    *   `generate_final_response_node`: ツール実行結果またはLLMの直接判断に基づいて最終的なユーザーへの応答を生成するノード。
*   `prompts/system_instruction.txt` を更新し、LLMに対して各ツールの機能説明と、どのような場合にどのツールをどのような引数で呼び出すべきかの具体的な指示（JSON形式での出力指示を含む）を明確化しました。
*   `bot.py` のLangGraphグラフ構造を更新し、新しいノード (`fetch_chat_history`, `decide_action`, `execute_tool`, `generate_response`) を組み込み、エッジと条件分岐を設定しました。
*   `tools/memory_tools.py` の `remember_tool` と `recall_tool` を `StructuredTool` を使用するように修正し、非同期関数を `coroutine` 引数に指定しました。また、LLMからのJSON応答をパースする処理を強化しました。
*   `llm_config.py` の `llm_chain` のプロンプトテンプレートを修正し、`system_instruction` を動的に受け取れるようにしました。
*   **一連のエラー修正を実施:**
    *   **LLM出力形式の厳密化:** `prompts/system_instruction.txt` を修正し、LLMが `tool_call` または `direct_response` を確実に出力するように指示を強化しました。
    *   **Pydanticモデルのバリデーション強化:**
        *   `state.py` の `ToolCall` モデルで、`args` フィールドが文字列で渡された場合にJSONパースして辞書に変換する `@field_validator` を追加しました。
        *   `state.py` の `LLMDecisionOutput` モデルで、`tool_call` と `direct_response` の排他性とどちらか一方の存在を強制する `@root_validator` を追加しました。
    *   **LLMチェーン出力の取り扱い修正:** `nodes.py` の `generate_final_response_node` で、`StrOutputParser` を使用した `llm_chain` の出力（文字列）を正しく扱うように修正しました。
    *   **これらの修正により、当初発生していた `LLMDecisionOutput did not contain tool_call or direct_response` エラー、Pydanticバリデーションエラー (`tool_call.args Input should be a valid dictionary`)、および `'str' object has no attribute 'content'` エラーが解消され、`remember_information` および `recall_information` ツールを含むエージェントが正常に動作するようになりました。**


フェーズ２：マルチモーダル機能と高度なインタラクションの実装 (LangChain & LangGraph)
M2.1: PDF・画像の読み込みと理解 (LangChain Multimodal LLM & LangGraph) - **完了**

**達成内容:**
*   `state.py` に `attachments` フィールドを追加し、添付ファイル情報を保持できるようにしました。
*   `bot.py` の `on_message` イベントハンドラを修正し、Discordメッセージに添付された画像ファイルとPDFファイルをダウンロードし、Base64エンコードして `AgentState` の `attachments` フィールドに格納するようにしました。
*   `nodes.py` に `process_attachments_node` を新規追加しました。このノードは `AgentState` の `attachments` を処理し、LLMがマルチモーダル入力として解釈できる形式（画像はBase64エンコードされたimage_url、PDFはBase64エンコードされたmediaデータ）に変換して `chat_history` に追加します。
*   `bot.py` の LangGraph ワークフローを更新し、`fetch_chat_history` の後に `process_attachments_node` を実行し、その後に `decide_action` に進むようにエッジを設定しました。
*   `prompts/system_instruction.txt` を更新し、LLMが添付ファイル（画像とPDF）の内容を理解し、それに基づいて応答を生成するよう指示を追加しました。

M2.2: 画像生成機能の実装 (LangChain Image Generation Tool & LangGraph) - **一部完了**

「〇〇の画像を生成して」という指示をLangGraphのルーティングロジックで認識し、LangChainの画像生成ツール (例: OpenAIDALLEImageGenerationTool やカスタムツール) を呼び出す。
生成された画像をDiscordメッセージとして投稿する処理を実装 (langchain-discord の DiscordSendMessageTool または discord.py 直接呼び出し)。

**達成内容:**
*   ユーザーの指示に基づいて `image_generation_tool` を呼び出し、画像を生成してDiscordに投稿する基本的なフローを実装しました。
*   **LLMへの入力トークン数超過問題への対処:**
    *   `nodes.py` の `decide_tool_or_direct_response_node`（行動判断ノード）および `generate_final_response_node`（最終応答生成ノード）において、LLMに渡す会話履歴 (`chat_history`) に含まれる過去の添付ファイルのBase64エンコードデータを簡略化（テキスト部分のみを抽出するか、固定の代替文字列に置換）する処理を実装しました。
    *   特に `generate_final_response_node` では、画像生成ツールが成功した場合、新たに生成された画像のBase64データをLLMに渡すことなく、ユーザーへのテキスト応答は「画像を生成しました！」のような固定メッセージとし、LLMの呼び出し自体をスキップするように修正しました。この判断は、`tool_output` がエラーメッセージで始まらないことを基準に行います。
*   **LLMの出力形式に関する問題への対処:**
    *   LLMがツール呼び出しの引数を誤った形式（例：シングルクォート使用）で出力し、Pydanticモデルのバリデーションエラーが発生していた問題に対し、`prompts/system_instruction.txt` 内の指示をより明確にし、引数を厳密なJSON形式（全てのキーと文字列値をダブルクォートで囲む）で出力するよう強く促す記述に変更しました。
    *   LLMがツール呼び出し (`tool_call`) も直接応答 (`direct_response`) も返さないというPydanticバリデーションエラーに対し、`prompts/system_instruction.txt` に、判断に迷う場合や指示が曖昧な場合でも、必ず `direct_response` を用いて何らかの応答（例：明確化のための質問）をするようフォールバック指示を追加しました。

**今後の課題 (ログより抽出):**
*   ~~**Pydantic v1非推奨警告 (`LangChainDeprecationWarning`):**~~ - **対応完了** (`tools/image_generation_tools.py` のインポートを修正)
    *   ~~内容: `langchain_core.pydantic_v1` の使用に関する警告。LangChainが内部的にPydantic v2を使用するようになったため、互換シムであるv1モジュールは非推奨となっています。~~
    *   ~~対応: プロジェクト全体でPydantic v2への移行を計画し、`from langchain_core.pydantic_v1 import BaseModel` のようなインポートを `from pydantic import BaseModel` （またはPydantic v2環境でv1互換のコードを扱う場合は `from pydantic.v1 import BaseModel`）に更新する必要があります。~~


M2.3: タイマー・アラーム機能の実装 (LangChain Custom Tool, LangGraph & Scheduling) - **完了**

「〇分後に教えて」「〇時にアラーム」といった指示を認識し、タイマー/アラームを設定するカスタムツールをLangChainのToolとして実装 (Pythonのasyncio.sleepやスケジューリングライブラリAPSchedulerなどを利用)。
LangGraphのエージェントがこのツールを呼び出し、設定時刻になったら指定されたチャンネルにメンションで通知する機能を実装。

**達成内容:**
*   `tools/timer_tools.py` に `set_timer` ツールを実装しました。このツールは、指定された分数後にDiscordチャンネルへ通知メッセージを送信します。
    *   タイマー設定時には「タイマーを設定しました。」という確認メッセージを即座に返します。
    *   実際の通知は、`asyncio.create_task` を用いてバックグラウンドで実行されます。
    *   `create_timer_tool` ファクトリ関数を導入し、`bot` インスタンスをツールに渡すようにしました。
*   `bot.py` を修正し、`create_timer_tool` を使用して `timer_tool` を初期化し、`nodes.py` に `tool_map` を渡すようにしました。また、`on_message` からタイマー完了通知のロジックを削除しました。
*   `nodes.py` の `generate_final_response_node` を修正し、タイマー設定時の確認メッセージを正しく処理するようにしました。また、ツール管理方法を `bot.py` と統一しました。
*   `prompts/system_instruction.txt` に `set_timer` ツールの説明を追加しました。
*   LLMが生成するツール引数のパースエラーに対応するため、`state.py` の `ToolCall` モデルのバリデーションを強化しました。
*   画像生成ツール以外のツール出力が画像生成成功と誤認される問題を修正するため、`tools/image_generation_tools.py` で出力にプレフィックスを追加し、`nodes.py` の `generate_final_response_node` の判定ロジックを修正しました。


フェーズ３：ユーザーエクスペリエンス向上と洗練化 (LangChain & LangGraph)
M3.1: フォローアップ質問選択肢の提供 (LangChain & Discord UI via LangGraph)

AIの応答後、LangChain (またはLLMの直接的な機能) を利用して文脈に沿ったフォローアップ質問を3つ生成する機能をLangGraphのノードとして実装。
生成された質問をDiscordのボタンコンポーネントとして表示し、ユーザーが選択できるようにする (LangGraphから discord.py のUI機能を呼び出す)。


M3.2: 意図解釈とツール使用判断の高度化 (LangChain Prompting, LangGraph Routing)

LangChainエージェントのプロンプトチューニングや、LangGraphの条件付きエッジ、ルーターノードをより詳細に活用し、ツール使用の判断精度を向上させる。
複数のツールを組み合わせた複雑なリクエストへの対応をLangGraphのグラフ構造で強化。


M3.3: エラーハンドリングとロギングの強化 (LangGraph & LangSmith)

LangGraphの各ノードで発生しうるエラーを網羅的に洗い出し、ユーザーフレンドリーなエラーメッセージを表示する。
LangSmithなどのLangChainエコシステムツールを活用し、詳細なログやトレース情報を出力し、問題発生時の原因究明を容易にする。


M3.4: パフォーマンスチューニングとリソース監視 (LangGraph & LangChain Caching)

応答速度の計測と改善 (LangGraphの非同期実行の活用など)。
APIコール数やトークン使用量の監視、LangChainのキャッシュ機能の導入による最適化。


フェーズ４：テスト・デプロイ・運用保守 (LangChain & LangGraph)
M4.1: 総合テスト

各機能の単体テスト、LangGraphの各ノード・エッジのテスト、結合テストを徹底。
様々な利用シナリオを想定したユーザー受入テスト (UAT) の実施。


M4.2: ドキュメント作成

ユーザー向け利用ガイド、管理者向け運用マニュアルの作成。
LangGraphのグラフ構造や状態遷移に関する開発者向けドキュメントの整備。


M4.3: デプロイ (LangServe検討)

本番環境へのデプロイ（VPS, クラウドサーバー, Dockerなどを検討）。
LangServeを用いたAPIとしてのデプロイも検討。


M4.4: 運用とフィードバック収集

Botの稼働監視 (LangSmithの活用)。
ユーザーからのフィードバックを収集し、改善点を洗い出す。
定期的なメンテナンスとLangChain/LangGraphライブラリのアップデート対応。


各マイルストーンのポイント：
反復的な開発: 各マイルストーン完了後、実際に動作させてみて、改善点や新たな発見があれば次のマイルストーン計画に反映させることが重要です。
LangChainとLangGraphの活用: LangChainのAgent, Tools, Chains, Memoryなどのコンポーネントと、LangGraphによる状態管理、フロー制御、エージェントオーケストレーションを最大限に活用することで、複雑な処理をモジュール化し、開発効率と堅牢性を高めます。
プロンプトエンジニアリング: LLMの能力を最大限に引き出すために、各機能におけるプロンプト (LangChainのChatPromptTemplate等) の設計とチューニングが鍵となります。
テストの重要性: 特に自然言語処理と外部API連携、LangGraphによる状態遷移が複雑に絡むため、予期せぬ挙動を防ぐために十分なテストが必要です。


プロジェクト改善のためのオプション修正(優先度低)
*   **PyNaCl未インストール警告:**
    *   内容: `PyNaCl is not installed, voice will NOT be supported`。音声関連機能（Discordのボイスチャットなど）が利用できない状態です。
    *   対応: 将来的に音声機能の実装を検討する場合は、PyNaClライブラリをインストールする必要があります。現状、テキストベースの機能のみであれば影響は限定的です。
*   **FAISS GPU非対応情報:**
    *   内容: `Failed to load GPU Faiss: name 'GpuIndexIVFFlat' is not defined.`。GPU版FAISSのロードに失敗し、CPU版で動作していることを示しています。
    *   対応: 現在のベクトルストアのデータ量や検索パフォーマンスに問題がなければ、CPU版のままでも問題ありません。将来的にパフォーマンスがボトルネックになるようであれば、GPU環境の整備とGPU版FAISSのセットアップを検討する価値があります。
