# Enterprise RAG Platform

Production-ready RAG (Retrieval-Augmented Generation) system with hybrid search, semantic caching, and full observability.

## 🏛️ Архитектура (C4 Model)

### Уровень 1: Системный контекст

```mermaid
graph TB
    subgraph "Enterprise RAG Platform"
        RAG[RAG System]
    end
    
    User[Пользователь] -->|API запросы| RAG
    RAG -->|Ответы| User
    
    Admin[Администратор] -->|Мониторинг| RAG
    
    subgraph "Внешние системы"
        VectorDB[(Qdrant)]
        Cache[(Redis)]
        LLM[Ollama]
        Observability[Prometheus/Grafana/Jaeger]
    end
    
    RAG -->|Векторы| VectorDB
    RAG -->|Кэш| Cache
    RAG -->|Генерация| LLM
    RAG -->|Метрики/Трейсы| Observability

graph TB
    subgraph "Enterprise RAG Platform"
        subgraph "Frontend"
            NGINX[NGINX Reverse Proxy]
        end
        
        subgraph "Application"
            API[FastAPI Application]
            Pipeline[RAG Pipeline]
        end
        
        subgraph "Data Layer"
            Qdrant[(Qdrant Vector Store)]
            Redis[(Redis Cache)]
        end
        
        subgraph "ML Layer"
            Ollama[Ollama Local LLM]
            Embedder[Embedding Factory]
        end
        
        subgraph "Observability"
            Prometheus[Prometheus Metrics]
            Grafana[Grafana Dashboards]
            Jaeger[Jaeger Tracing]
        end
    end
    
    Client[Клиент] --> NGINX
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
    subgraph "FastAPI Application"
        QueryHandler[Query Handler]
        HealthHandler[Health Handler]
        MetricsHandler[Metrics Handler]
        Pipeline[RAG Pipeline]
        Config[Config Manager]
        Middleware[Middleware]
        Logger[Structured Logger]
        Metrics[Prometheus Metrics]
        Tracer[OpenTelemetry Tracer]
    end
    
    subgraph "RAG Pipeline Components"
        Retriever[Retriever]
        LLMClient[LLM Client]
        PromptBuilder[Prompt Builder]
        EmbeddingFactory[Embedding Factory]
    end
    
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
    
    subgraph "Metrics"
        App -->|/metrics| Prometheus[Prometheus]
        Prometheus --> Grafana[Grafana]
    end
    
    subgraph "Logging"
        App -->|JSON| Console[Console]
        App -->|File| LogFile[logs/app.log]
    end
    
    subgraph "Tracing"
        App -->|OTLP| Jaeger[Jaeger]
        Jaeger --> JaegerUI[Jaeger UI]
    end