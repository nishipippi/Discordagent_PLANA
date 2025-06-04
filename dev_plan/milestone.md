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

M1.4.1: データベース基盤の準備 (SQLite & VectorStore)

tools/db_utils.py を作成し、SQLiteデータベース (data/memory.db) の初期化と構造化データ用テーブル (memories) の作成を実装。
長期記憶・意味的想起のために、LangChainがサポートするベクトルストア (例: Chroma, FAISS をSQLite等で永続化) のセットアップを検討・実装。
bot.py (またはLangGraphの初期化処理) でデータベースとベクトルストアが初期化されるように連携。


M1.4.2: 記憶ツール (remember_information) の実装 (LangChain Tool & LangGraph)

tools/memory_tools.py にユーザーの入力テキストをLLM (ChatOpenAI等) を用いてJSON形式に構造化し、SQLiteデータベースに保存する remember_information_func 関数と対応するPydanticモデル (RememberInput) を定義。これをLangChainのToolとしてラップ。
ユーザーが指定した情報をベクトルストアにも埋め込みベクトルとして保存するロジックを追加。
server_id, channel_id, user_id をツールに渡すため、LangGraphのステートまたはエージェント呼び出し時の引数として管理。


M1.4.3: 想起ツール (recall_information) の実装 (LangChain Tool & LangGraph)

tools/memory_tools.py に recall_information_func 関数と対応するPydanticモデル (RecallInput) を定義。
構造化データはSQLiteから、意味的に関連する情報はベクトルストアから検索し、LLMを用いてユーザーの質問に回答を生成するロジックを実装。これをLangChainのToolとしてラップ。


M1.4.4: エージェントへのツール組み込みと動作確認 (LangGraph Agent & LangChain Tools)

bot.py (またはLangGraphのメイン処理) で、定義した記憶・想起ツール、検索ツールをLangGraphで構築するエージェントのツールリストに含める。
prompts/structure_memory_prompt.txt と prompts/answer_from_memory_prompt.txt を作成・更新し、LLMによる構造化と回答生成の精度向上を図る (LangChainのChatPromptTemplateとして利用)。
prompts/system_instruction.txt を更新し、プラナのペルソナ設定と各ツールの利用に関する指示をエージェントのシステムプロンプトに明確化 (LangChainのChatPromptTemplateとして利用)。
LangGraphのステートとエッジを調整し、記憶・想起フローを確立。


現在の課題:

LangGraphを用いたより複雑な状態遷移と、ユーザーごとのメモリ永続化の堅牢な実装。
ベクトルストアを利用した高度な想起機能のチューニング。
M1.2: 会話履歴のコンテキスト利用において、LLMが過去の会話内容を長期記憶として活用できていない。これは、プロンプトの指示不足、またはLangGraphのState管理とLLMへのコンテキスト渡し方の改善が必要である可能性を示唆している。


フェーズ２：マルチモーダル機能と高度なインタラクションの実装 (LangChain & LangGraph)
M2.1: PDF・画像の読み込みと理解 (LangChain Multimodal LLM & LangGraph)

Discordメッセージに添付されたPDFや画像をLangChain経由でマルチモーダル対応LLM (例: Gemini, GPT-4o) に渡し、内容を理解させる機能をLangGraphのノードとして実装。
理解した情報を基にユーザーの質問に答えたり、要約したりする機能をLangGraphのフローとして実装。


M2.2: 画像生成機能の実装 (LangChain Image Generation Tool & LangGraph)

「〇〇の画像を生成して」という指示をLangGraphのルーティングロジックで認識し、LangChainの画像生成ツール (例: OpenAIDALLEImageGenerationTool やカスタムツール) を呼び出す。
生成された画像をDiscordメッセージとして投稿する処理を実装 (langchain-discord の DiscordSendMessageTool または discord.py 直接呼び出し)。


M2.3: タイマー・アラーム機能の実装 (LangChain Custom Tool, LangGraph & Scheduling)

「〇分後に教えて」「〇時にアラーム」といった指示を認識し、タイマー/アラームを設定するカスタムツールをLangChainのToolとして実装 (Pythonのasyncio.sleepやスケジューリングライブラリAPSchedulerなどを利用)。
LangGraphのエージェントがこのツールを呼び出し、設定時刻になったら指定されたチャンネルにメンションで通知する機能を実装。


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
