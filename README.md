# Enterprise RAG Platform

Production-ready RAG (Retrieval-Augmented Generation) system with hybrid search, semantic caching, and full observability.

## 🏛️ Архитектура (C4 Model)

### Уровень 1: Системный контекст

graph TB
    User[Пользователь]
    RAG[RAG System]
    Admin[Администратор]
    VectorDB[(Qdrant)]
    Cache[(Redis)]
    LLM[Ollama]
    Observability[Prometheus/Grafana/Jaeger]

    User -->|API запросы| RAG
    RAG -->|Ответы| User
    Admin -->|Мониторинг| RAG
    RAG -->|Векторы| VectorDB
    RAG -->|Кэш| Cache
    RAG -->|Генерация| LLM
    RAG -->|Метрики/Трейсы| Observability

graph TB
    Client[Клиент]
    NGINX[NGINX Reverse Proxy]
    API[FastAPI Application]
    Pipeline[RAG Pipeline]
    Qdrant[(Qdrant Vector Store)]
    Redis[(Redis Cache)]
    Ollama[Ollama Local LLM]
    Embedder[Embedding Factory]
    Prometheus[Prometheus Metrics]
    Grafana[Grafana Dashboards]
    Jaeger[Jaeger Tracing]

    Client --> NGINX
    NGINX --> API
    API --> Pipeline
    Pipeline --> Qdrant
    Pipeline --> Redis
    Pipeline --> Ollama
    Pipeline --> Embedder
    API --> Prometheus
    API --> Jaeger
    Grafana --> Prometheus
    
graph TB
    QueryHandler[Query Handler]
    HealthHandler[Health Handler]
    MetricsHandler[Metrics Handler]
    Pipeline[RAG Pipeline]
    Config[Config Manager]
    Middleware[Middleware]
    Logger[Structured Logger]
    Metrics[Prometheus Metrics]
    Tracer[OpenTelemetry Tracer]
    Retriever[Retriever]
    LLMClient[LLM Client]
    PromptBuilder[Prompt Builder]
    EmbeddingFactory[Embedding Factory]

    QueryHandler --> Pipeline
    Pipeline --> Retriever
    Pipeline --> LLMClient
    Pipeline --> PromptBuilder
    Pipeline --> EmbeddingFactory
    QueryHandler --> Logger
    QueryHandler --> Metrics
    QueryHandler --> Tracer

sequenceDiagram
    participant Client
    participant API as FastAPI
    participant Pipeline as RAG Pipeline
    participant Retriever
    participant Qdrant
    participant Prompt as PromptBuilder
    participant LLM as LocalLLM
    participant Obs as Observability
    
    Client->>API: POST /query
    API->>Obs: Start trace
    API->>Pipeline: query()
    Pipeline->>Retriever: search()
    Retriever->>Qdrant: hybrid_search()
    Qdrant-->>Retriever: results
    Retriever-->>Pipeline: contexts
    Pipeline->>Prompt: build()
    Prompt-->>Pipeline: prompt
    Pipeline->>LLM: generate()
    LLM-->>Pipeline: answer
    Pipeline-->>API: response
    API->>Obs: Record metrics
    API-->>Client: JSON response

graph TB
    App[FastAPI App]
    Prometheus[Prometheus]
    Grafana[Grafana]
    Console[Console]
    LogFile[logs/app.log]
    Jaeger[Jaeger]
    JaegerUI[Jaeger UI]

    App -->|/metrics| Prometheus
    Prometheus --> Grafana
    App -->|JSON| Console
    App -->|File| LogFile
    App -->|OTLP| Jaeger
    Jaeger --> JaegerUI
