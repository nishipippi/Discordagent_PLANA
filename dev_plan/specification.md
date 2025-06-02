Discord AIエージェント 要件定義書 (LangChain & LangGraph版)1. はじめに
1.1. プロジェクト概要

本ドキュメントは、Discord上で動作するAIエージェント（以下、本エージェント）の開発に関する要件を定義するものです。
本エージェントは、ユーザーの自然言語による指示を理解し、情報検索、雑談、タイマー設定、画像生成、情報記憶・想起、PDF/画像ファイルの内容理解といった多様なタスクを、LangChainおよびLangGraphフレームワークを活用して自律的に実行することで、Discordサーバーにおけるユーザー体験の向上とコミュニケーションの活性化を目指します。


1.2. 背景と目的

Discordサーバーにおける情報共有の効率化、ユーザーエンゲージメントの向上、および各種タスクの自動化。
高度なAI技術（特にGoogle Gemini等のLLM）と、LangChainおよびLangGraphフレームワークを活用し、多機能かつ自然でステートフルな対話が可能なAIエージェントを実現する。


1.3. 適用範囲

本ドキュメントは、本エージェントの機能要件、非機能要件、技術要件、および開発マイルストーンの概要を記述します。


2. 機能要件
2.1. 基本動作 (LangGraphによる状態管理)

2.1.1. メンションによる起動:

ユーザーが本エージェントにメンションすることで、本エージェントは指示の受付を開始し、LangGraphのステートマシンが適切な状態に遷移する。


2.1.2. 自然言語理解 (LangChain LLMChain & LangGraph Routing):

ユーザーからの指示は、特定のコマンド形式ではなく、自然言語で行われるものとする。
本エージェントは、LangChainのLLMChainを用いて指示内容を解析し、LangGraphのルーティングロジックにより実行すべきタスク（ツール呼び出しや内部状態遷移）を判断する。


2.1.3. 会話履歴のコンテキスト利用 (LangChain Memory & LangGraph State):

メンションされたチャンネルの直近10分間の会話履歴をlangchain-discordのDiscordReadMessagesツール等で取得し、LangChainのConversationBufferWindowMemory（または同等のメモリコンポーネント）をLangGraphのステートの一部としてユーザーIDごとに管理・永続化する 1。メッセージ数の上限は設けない（トークン制限の範囲内で最大限活用）。




2.2. コア機能 (LangChain Tools & LangGraph Agent Logic)

2.2.1. 自律的な検索と応答 (LangChain Custom Tool & LangGraph):

ユーザーの指示内容に基づき、LangGraph内のLLM判断ノードが検索の必要性を判断する。
必要と判断した場合、Brave Search APIを利用するLangChainカスタムツールを実行する 5。
検索結果をLangChainのLLMChain等で適切に処理（要約、関連部分の引用など）し、ユーザーに応答する。


2.2.2. ユーザーとの雑談 (LangChain LLMChain & LangGraph):

検索や特定タスクに該当しない場合、LangChainのLLMChainを用いてユーザーの雑談に応じる。
（任意）特定のペルソナ（性格・口調）をLangChainのプロンプトテンプレートで設定し、LangGraphのステートで管理することを検討する 6。


2.2.3. タイマー・アラーム機能 (LangChain Custom Tool & LangGraph & Scheduling):

ユーザーの指示に基づき、LangGraph内のLLM判断ノードがタイマーまたはアラームの設定が必要かを判断する。
設定が必要な場合、PythonのasyncioやAPSchedulerを利用したLangChainカスタムツールを実行し、指定された時間または時刻に、メンション元のチャンネルで通知（メンション）を行う。


2.2.4. 画像生成機能 (LangChain Image Generation Tool & LangGraph):

ユーザーの指示に基づき、LangGraph内のLLM判断ノードが画像生成の必要性を判断する。
必要と判断した場合、Google Gemini API (gemini-2.0-flash-preview-image-generation) またはLangChainのOpenAIDALLEImageGenerationTool等を利用するLangChainツールを実行し、画像を生成し、Discordメッセージとして投稿する 8。


2.2.5. 記憶・想起機能 (LangChain Tools, VectorStore & LangGraph):

「覚えておいて」等の指示に基づき、指定された情報をLangChainのカスタムツール経由でデータベース（SQLite等の構造化データ用）およびベクトルストア（意味的検索用、例: Chroma, FAISS）に保存する 10。情報はサーバー単位およびユーザー単位で管理し、永続的に保持する。
会話の文脈やユーザーからの質問に応じて、LangGraphのロジックとLangChainツール（ベクトルストア検索等）を用いて記憶した情報を自律的に想起し、応答に活用する。




2.3. ファイル処理機能 (LangChain Multimodal LLM & LangGraph)

2.3.1. PDF・画像の読み込みと理解:

Discordメッセージに添付されたPDFファイルおよび画像ファイルを認識する。
LangChain経由でGoogle Gemini (Flash) のようなマルチモーダル対応LLMを利用し、ファイルの内容（テキスト、画像情報）を理解する。
理解した内容に基づき、ユーザーの質問応答や情報提供を行う処理をLangGraphのノードとして実装する。




2.4. ユーザーインターフェース・インタラクション (LangChain & Discord UI via LangGraph)

2.4.1. フォローアップ質問の提示:

本エージェントの応答後、LangChain (またはLLMの直接的な機能) を利用して文脈に沿ったフォローアップ質問を3つ自動生成する機能をLangGraphのノードとして実装する。
生成された質問をDiscordのUIボタンとして表示し、ユーザーが選択できるようにする (LangGraphから discord.py のUI機能を呼び出す)。




3. 非機能要件
3.1. パフォーマンス

3.1.1. 応答速度:

通常の雑談や簡単な情報提供：数秒以内を目指す。
検索、画像生成、ファイル処理を伴う場合：処理内容に応じて許容範囲を設定するが、過度にユーザーを待たせないよう配慮する。LangGraphの非同期実行を活用する。




3.2. 信頼性・可用性

3.2.1. 安定稼働:

長時間の安定稼働を目指す。LangGraphの永続化機能を利用して状態を保持する 3。


3.2.2. エラーハンドリング:

APIエラー、機能実行エラーなどが発生した場合、ユーザーに分かりやすいエラーメッセージを表示する。
LangSmithや適切なロギングライブラリを用いて、LangChainおよびLangGraphの実行トレースを含む詳細なログを記録し、問題解決に役立てる 16。




3.3. セキュリティ

3.3.1. APIキー管理:

Discord Botトークン、LLM APIキー、Brave Search APIキー等は、環境変数や設定ファイルなど、セキュアな方法で管理する。


3.3.2. データ保護:

記憶機能で保存される情報について、不必要な外部漏洩がないよう配慮する。




3.4. 保守性・拡張性

3.4.1. コード品質:

可読性が高く、メンテナンスしやすいコードを記述する。
LangChainおよびLangGraphフレームワークの思想に沿ったモジュール化を意識する 17。


3.4.2. 機能追加の容易性:

将来的な機能追加や変更が比較的容易に行えるような設計を心がける。LangGraphのノードやエッジの追加・変更の容易性を活用する。




3.5. 使用言語

3.5.1. 対応言語:

当面は日本語での指示と応答を対象とする。




4. 技術要件
4.1. プログラミング言語:

Python (非同期処理 asyncio を活用)


4.2. 主要ライブラリ・フレームワーク:

discord.py: Discord Bot開発用ライブラリ 18。
langchain: AIエージェント開発フレームワーク 5。
langgraph: ステートフルなマルチエージェントアプリケーション構築用ライブラリ 3。
langchain-openai (または langchain-google-genai 等): LangChain用LLMインテグレーション 16。
google-generativeai (Gemini利用時): Google GenAI SDK。
requests (Brave Search API直接コール時): HTTPリクエスト用ライブラリ。
ベクトルストアライブラリ (例: chromadb, faiss-cpu, pinecone-client) 10。
データベース操作ライブラリ (例: sqlite3, SQLAlchemy など)。
(任意) APScheduler 等のスケジューリングライブラリ (タイマー機能用)。
langchain-discord-shikenso: Discordメッセージ送受信ツール連携用 4。


4.3. AIモデル:

Google Gemini (Flash) または OpenAI GPTシリーズ (テキスト生成、マルチモーダル処理、ツール呼び出し/Function Callingなど)。
Google Gemini gemini-2.0-flash-preview-image-generation または DALL-E (画像生成)。


4.4. 外部API:

LLM API (Google GenAI API, OpenAI APIなど)。
Brave Search API。


4.5. データベース:

SQLite (初期)、または必要に応じてPostgreSQL, MySQLなど (構造化データ用)。
ベクトルストア (Chroma, FAISS, Pineconeなど) (非構造化データ・意味的検索用) 24。


4.6. 開発環境:

Gitによるバージョン管理。


4.7. デプロイ環境:

未定 (VPS, クラウドサーバー, Dockerコンテナなどを想定)。LangServeの利用も検討。


5. 開発マイルストーン概要 (詳細は別紙マイルストーン計画を参照)
フェーズ０：準備・基盤構築

開発環境構築、APIキー設定、基本的なBot疎通確認、LangChain/LangGraph基本セットアップ。


フェーズ１：中核機能の実装 (MVP) - LangGraphによるステートフルエージェント化

自然言語指示理解と雑談応答 (LangGraphノード)、会話履歴利用 (LangGraphステートとLangChainメモリ)、自律検索 (LangChainカスタムツールとLangGraphロジック)、記憶・想起機能 (LangChainツール、VectorStore、LangGraphステート/ロジック)。


フェーズ２：マルチモーダル機能と高度なインタラクションの実装 (LangChain & LangGraph)

PDF/画像読み込み・理解 (LangChainマルチモーダルLLMとLangGraphノード)、画像生成機能 (LangChain画像生成ツールとLangGraphロジック)、タイマー・アラーム機能 (LangChainカスタムツール、LangGraph、スケジューリング)。


フェーズ３：ユーザーエクスペリエンス向上と洗練化 (LangChain & LangGraph)

フォローアップ質問提示 (LangChainとDiscord UI via LangGraph)、意図解釈精度向上 (LangChainプロンプティングとLangGraphルーティング)、エラーハンドリング強化 (LangGraphとLangSmith)、パフォーマンスチューニング (LangGraphとLangChainキャッシング)。


フェーズ４：テスト・デプロイ・運用保守 (LangChain & LangGraph)

総合テスト、ドキュメント作成、デプロイ (LangServe検討)、運用とフィードバック収集。


6. 除外事項
本定義書に記載のない機能。
現時点での多言語対応 (日本語のみ)。
ユーザーごとの詳細な権限管理機能 (サーバー単位での利用を想定)。
GUIによる設定画面。
7. その他・備考
LLM APIのトークン制限は設計時に考慮し、LangChainのテキスト分割機能やLangGraphの状態管理、メモリ戦略によって適切に処理する。
AIの意図解釈やツール使用判断の精度向上は継続的な課題とし、プロンプトエンジニアリング、LangChainのツール定義の最適化、LangGraphのルーティングロジックの改善を通じて向上を図る 5。
利用するLLM APIの仕様変更等があった場合は、適宜対応を行う。
この要件定義書が、LangChainとLangGraphを前提としたプロジェクトの明確な指針となれば幸いです。内容についてご確認いただき、修正点や追加事項があればお気軽にお知らせください。