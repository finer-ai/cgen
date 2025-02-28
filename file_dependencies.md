# SotaiGenerator - ファイル依存関係図

本ドキュメントでは、SotaiGeneratorプロジェクトのファイル間の依存関係をグラフ形式で示します。これにより、コードベースの構造と各コンポーネント間の関係を視覚的に理解できます。

## 全体依存関係図

以下の図は、主要ファイル間の依存関係を示しています。矢印は「依存する方向」を示しており、例えば A → B は「AがBに依存している」ことを表します。

```mermaid
graph TB
    %% アプリケーションのメインファイル
    main[app/main.py] --> routes[app/api/routes.py]
    main --> config[app/core/config.py]
    main --> errors[app/core/errors.py]
    
    %% APIルート
    routes --> schemas[app/api/schemas.py]
    routes --> rag_service[app/services/rag_service.py]
    routes --> dart_service[app/services/dart_service.py]
    routes --> image_service[app/services/image_service.py]
    routes --> errors
    
    %% サービス
    rag_service --> config
    rag_service --> errors
    rag_service --> tag_utils[app/utils/tag_utils.py]
    
    dart_service --> config
    dart_service --> errors
    dart_service --> tag_utils
    
    image_service --> config
    image_service --> errors
    
    %% データストア
    rag_service --> faiss[app/data/faiss]
    
    %% モデル
    dart_service --> dart_model[app/models/dart]
    image_service --> sd_model[app/models/animagine]
    
    %% 設定
    config --> env[.env]
    
    %% スクリプト
    download_dataset[scripts/download_dataset.py]
    download_models[scripts/download_models.sh]
    prepare_vector_store[scripts/prepare_vector_store.py] --> faiss
    test_api[scripts/test_api.py] -.-> routes
    call_py[scripts/call.py] -.-> routes
    
    %% クラスハイライト
    class main,routes,config,schemas focus
    class rag_service,dart_service,image_service emphasis
```

## レイヤー別依存関係図

プロジェクトのレイヤーアーキテクチャを示す依存関係図です。

```mermaid
graph TB
    subgraph "APIレイヤー"
        main[app/main.py]
        routes[app/api/routes.py]
        schemas[app/api/schemas.py]
    end
    
    subgraph "サービスレイヤー"
        rag_service[app/services/rag_service.py]
        dart_service[app/services/dart_service.py]
        image_service[app/services/image_service.py]
    end
    
    subgraph "ユーティリティレイヤー"
        config[app/core/config.py]
        errors[app/core/errors.py]
        tag_utils[app/utils/tag_utils.py]
    end
    
    subgraph "データレイヤー"
        faiss[app/data/faiss]
        dart_model[app/models/dart]
        sd_model[app/models/animagine]
        env[.env]
    end
    
    %% レイヤー間の依存関係
    main --> routes
    main --> config
    main --> errors
    
    routes --> schemas
    routes --> rag_service
    routes --> dart_service
    routes --> image_service
    
    rag_service --> config
    rag_service --> errors
    rag_service --> tag_utils
    rag_service --> faiss
    
    dart_service --> config
    dart_service --> errors
    dart_service --> tag_utils
    dart_service --> dart_model
    
    image_service --> config
    image_service --> errors
    image_service --> sd_model
    
    config --> env
```

## 主要処理フローの依存関係

画像生成プロセスの主要フローに関わるファイルの依存関係を示します。

```mermaid
flowchart TD
    %% ユーザーからのリクエストの流れ
    user[ユーザー] --> main[app/main.py]
    main --> routes[app/api/routes.py]
    
    %% タグ生成フロー
    routes --> rag_service[app/services/rag_service.py]
    rag_service --> faiss[app/data/faiss]
    rag_service --> tag_utils[app/utils/tag_utils.py]
    
    routes --> dart_service[app/services/dart_service.py]
    dart_service --> dart_model[app/models/dart]
    dart_service --> tag_utils
    
    %% 画像生成フロー
    routes --> image_service[app/services/image_service.py]
    image_service --> sd_model[app/models/animagine]
    
    %% レスポンスの流れ
    image_service --> routes
    routes --> main
    main --> user
```

## サービスコンポーネントの内部依存関係

各サービスコンポーネント内部の依存関係を示します。

### RAGサービス

```mermaid
graph LR
    rag_service[RAGService] --> embeddings[OpenAIEmbeddings]
    rag_service --> vector_store[FAISS]
    rag_service --> llm[OpenAI LLM]
    rag_service --> keyword_chain[LLMChain]
    rag_service --> extract_keywords[extract_keywords]
    rag_service --> retrieve_tags[retrieve_tags]
    rag_service --> generate_tag_candidates[generate_tag_candidates]
    
    extract_keywords --> keyword_chain
    retrieve_tags --> vector_store
    generate_tag_candidates --> extract_keywords
    generate_tag_candidates --> retrieve_tags
```

### Dartサービス

```mermaid
graph LR
    dart_service[DartService] --> tokenizer[AutoTokenizer]
    dart_service --> model[AutoModelForCausalLM]
    dart_service --> format_input[_format_input_for_dart]
    dart_service --> generate_final_tags[generate_final_tags]
    
    generate_final_tags --> format_input
    generate_final_tags --> tokenizer
    generate_final_tags --> model
```

### 画像生成サービス

```mermaid
graph LR
    image_service[ImageService] --> pipe[StableDiffusionPipeline]
    image_service --> scheduler[DPMSolverMultistepScheduler]
    image_service --> generate_image[generate_image]
    
    generate_image --> pipe
```

## スクリプトと初期化関連

初期化スクリプトとデータ準備に関連するコンポーネントの依存関係です。

```mermaid
graph TB
    download_dataset[scripts/download_dataset.py] --> data[data/]
    download_models[scripts/download_models.sh] --> models[app/models/]
    prepare_vector_store[scripts/prepare_vector_store.py] --> faiss[app/data/faiss]
    
    data --> prepare_vector_store
```

## 注意事項

- 上記の図はファイル間の主要な依存関係を示しています。実際のコードベースでは、ここに示されていない副次的な依存関係が存在する場合があります。
- モジュールレベルではなく、関数やクラスレベルの詳細な依存関係は含まれていません。
- 外部ライブラリへの依存関係は、簡潔さのために図には含まれていません。

## 依存関係分析の活用方法

このファイル依存関係図は以下のような場面で活用できます：

1. **新規開発者のオンボーディング** - コードベースの全体像をすばやく理解
2. **リファクタリング計画** - 変更の影響範囲を把握
3. **テスト戦略** - 依存関係に基づくテスト対象の特定
4. **モジュール分割** - 高凝集・低結合を実現するためのモジュール境界の検討
5. **技術的負債の特定** - 過剰な依存関係や循環依存などの問題箇所の発見 