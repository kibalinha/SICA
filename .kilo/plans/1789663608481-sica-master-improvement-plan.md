# SICA - Plano Mestre de Melhorias (3 Fases)

**Objetivo**: Evoluir o SICA de MVP funcional para sistema de produção robusto, observável e performático.

---

## Visão Geral das Fases

| Fase | Foco | Entregável Principal | Dependência |
|------|------|---------------------|-------------|
| **1** | Infraestrutura | Docker Compose + Redis + Prometheus/Grafana | Nenhuma |
| **2** | Modelo/Performance | PANNs → ONNX INT8 (2-3x mais rápido, 4x menor) | Fase 1 (para cache Redis) |
| **3** | Frontend/UX | Dashboard histórico + exportação PDF | Fase 1 (API de histórico) |

---

## FASE 1: Infraestrutura Docker + Observabilidade

### 1.1 Docker Compose Unificado
**Arquivos a criar/modificar:**
- `docker-compose.yml` (raiz) - Orquestra: backend, frontend, redis, prometheus, grafana
- `sica-backend/Dockerfile` - Multi-stage: builder → runtime (Python 3.11 slim)
- `sica-app/Dockerfile` - Multi-stage: builder (Node 20) → nginx runtime
- `.dockerignore` (backend e frontend)
- `docker-compose.override.yml.example` - Para desenvolvimento local

**Serviços:**
```yaml
services:
  redis:
    image: redis:7-alpine
    volumes: [redis_data:/data]
    healthcheck: redis-cli ping

  backend:
    build: ./sica-backend
    ports: ["8001:8000"]
    env_file: ./sica-backend/.env
    depends_on: [redis]
    environment:
      - REDIS_URL=redis://redis:6379
      - PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus

  frontend:
    build: ./sica-app
    ports: ["5173:80"]
    depends_on: [backend]

  prometheus:
    image: prom/prometheus:v2.52
    ports: ["9090:9090"]
    volumes: [./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml]
    command: ["--config.file=/etc/prometheus/prometheus.yml", "--storage.tsdb.path=/prometheus"]

  grafana:
    image: grafana/grafana:10.4
    ports: ["3000:3000"]
    volumes: [grafana_data:/var/lib/grafana, ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards]
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
```

### 1.2 Backend: Integração Redis + Métricas Prometheus
**Modificações em `sica-backend/`:**

| Arquivo | Mudança |
|---------|---------|
| `requirements.txt` | + `redis>=5.0`, `prometheus-client>=0.19`, `gunicorn>=21.0` |
| `server.py` | Substituir `_audio_cache` (dict) por Redis client com TTL; adicionar `/metrics` endpoint |
| `audio_processor.py` | Instrumentar latência por etapa (STFT, HPSS, IA, total) com `Histogram` |
| `ai_engine.py` | Métricas de inferência: `model_inference_duration_seconds`, `model_load_duration` |
| `pyproject.toml` | Adicionar `[tool.prometheus]` config se necessário |

**Estratégia de Cache Redis:**
```python
# Chave: sha256(audio_bytes) -> valor: JSON result + timestamp
# TTL: 3600s (configurável via SICA_CACHE_TTL)
# Max entradas: LRU via Redis maxmemory-policy allkeys-lru
```

### 1.3 Configuração Prometheus + Grafana
**Arquivos a criar:**
- `monitoring/prometheus.yml` - Scrape configs para backend, node-exporter, cadvisor
- `monitoring/grafana/dashboards/sica-overview.json` - Dashboard principal:
  - Latência P50/P95/P99 `/analyze`
  - Throughput (req/s)
  - Cache hit/miss rate
  - Erros por tipo (4xx, 5xx)
  - Uso CPU/Memória/GPU
  - Fila de requisições (se houver)

### 1.4 Variáveis de Ambiente Novas
**Backend (`.env`):**
```env
REDIS_URL=redis://localhost:6379
SICA_CACHE_TTL=3600
SICA_REDIS_MAX_MEMORY=256mb
PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus
```

---

## FASE 2: PANNs → ONNX INT8

### 2.1 Script de Exportação ONNX
**Novo arquivo:** `sica-backend/export_onnx.py`
```python
# 1. Carregar PANNs Cnn14 (panns_inference)
# 2. Exportar para ONNX opset 17 com dynamic axes (batch, time)
# 3. Validar paridade numérica (FP32 vs original PyTorch)
# 4. Aplicar quantização dinâmica INT8 via onnxruntime.quantization
# 5. Salvar: cnn14_fp32.onnx, cnn14_int8.onnx
# 6. Benchmark: latência, memória, acurácia (subset AudioSet)
```

**Dependências novas:** `requirements-export.txt` → `onnx>=1.16`, `onnxruntime>=1.18`, `onnxruntime-tools`

### 2.2 Novo Motor de Inferência ONNX
**Novo arquivo:** `sica-backend/onnx_engine.py`
```python
class OnnxEngine:
    def __init__(self, model_path: str, providers: List[str] = ["CPUExecutionProvider"]):
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
    
    def analyze_soundscape(self, waveform: np.ndarray, sr: int) -> Dict:
        # Preprocess idêntico ao PANNs (librosa.resample, mel-spectrogram)
        # Run inference
        # Postprocess: mapear 527 classes → 3 macro (music/speech/noise)
        # Retornar formato compatível com PannsEngine
```

### 2.3 Integração no `ai_engine.py`
- Factory `get_ai_engine()` seleciona backend via env `SICA_INFERENCE_BACKEND=panns|onnx_fp32|onnx_int8`
- Fallback automático: ONNX INT8 → FP32 → PANNs PyTorch
- Manter API idêntica: `analyze_soundscape()`, `apply_neural_speech_mask()`, `categorize_panns_probs()`

### 2.4 Benchmarks & Validação
**Script:** `sica-backend/benchmark_onnx.py`
- Dataset: 100 amostras variadas (música, voz, ruído, misto)
- Métricas: latência P50/P95/P99, throughput, acurácia macro-F1 vs PANNs original
- Critério de aceitação: INT8 < 5% degradação F1, latência < 400ms CPU

---

## FASE 3: Dashboard Histórico Frontend

### 3.1 API de Histórico (Backend)
**Novos endpoints em `server.py`:**
```python
GET  /api/v1/history?limit=50&offset=0&start_date=&end_date=
GET  /api/v1/history/{session_id}
GET  /api/v1/history/export?format=pdf|csv&start_date=&end_date=
DELETE /api/v1/history/{session_id}
```

**Armazenamento:** Redis sorted set (`sica:history:{user_id}`) com score = timestamp, value = JSON resumo
- Retenção: 90 dias (configurável)
- Tamanho máximo: 10k entradas por usuário

### 3.2 Frontend: Nova Página `/history`
**Componentes novos em `sica-app/src/`:**
```
components/
  history/
    HistoryPage.tsx          # Página principal
    SessionList.tsx          # Tabela paginada + filtros
    SessionCard.tsx          # Card resumo (dBA, SNR, recomendação, timestamp)
    SessionDetailModal.tsx   # Detalhes + gráfico espectro + replay áudio
    ExportButton.tsx         # PDF/CSV download
    DateRangePicker.tsx      # Filtro de data
```

**Bibliotecas novas:**
- `recharts` ou `chart.js` - Gráficos timeline (dBA, SNR ao longo do tempo)
- `jspdf` + `jspdf-autotable` - Exportação PDF client-side
- `date-fns` - Manipulação de datas

### 3.3 Visualização de Sessão Individual
**Modal com:**
- Gráfico de linha: total_dBA, harmonic_dBA, noise_dBA, SNR (eixo X = tempo relativo)
- Espectrograma estático (imagem gerada no backend ou canvas frontend)
- Player de áudio (se armazenado) ou link para download
- Métricas: duração, confiança IA, cena dominante, recomendação aplicada

### 3.4 Navegação & Integração
- Adicionar aba "Histórico" no Header/Nav
- Botão "Ver histórico" no ResultCard após análise
- Persistir preferências de filtro (localStorage)

---

## Riscos & Mitigações

| Risco | Probabilidade | Impacto | Mitigação |
|-------|---------------|---------|-----------|
| ONNX INT8 degrada acurácia >5% | Média | Alto | Manter fallback PANNs; validar com dataset representativo |
| Redis indisponível derruba backend | Baixa | Alto | Circuit breaker + fallback para cache em memória |
| Dashboard lentidão com 10k sessões | Média | Médio | Paginação server-side + virtualização lista |
| Docker compose complexo para dev | Baixa | Médio | `docker-compose.override.yml` para bind mounts + hot reload |

---

## Ordem de Execução Sugerida

```
Semana 1: Fase 1.1-1.3 (Docker + Redis + Prometheus)
Semana 2: Fase 1.4 + Testes integração + CI/CD update
Semana 3: Fase 2.1-2.2 (Export ONNX + Engine)
Semana 4: Fase 2.3-2.4 (Integração + Benchmarks + Validação)
Semana 5: Fase 3.1-3.2 (API Histórico + Página base)
Semana 6: Fase 3.3-3.4 (Detalhes + Export + Polish)
```

---

## Validação por Fase

### Fase 1 - Critérios de Aceite
- [ ] `docker-compose up -d` sobe todos os 5 serviços saudáveis
- [ ] Backend responde em `localhost:8001/api/v1/health` com `cache: {enabled: true, backend: "redis"}`
- [ ] `/metrics` expõe métricas Prometheus (latência, cache hit/miss)
- [ ] Grafana mostra dashboard SICA com dados reais
- [ ] Cache Redis persiste entre reinícios do backend

### Fase 2 - Critérios de Aceite
- [ ] `onnx_engine.py` passa todos os testes de `test_ai_engine.py` (adaptados)
- [ ] Benchmark: ONNX INT8 < 400ms P95 CPU, < 5% F1 drop vs PANNs
- [ ] Fallback automático funciona (INT8 → FP32 → PyTorch)
- [ ] Modelo INT8 < 100MB (vs 327MB original)

### Fase 3 - Critérios de Aceite
- [ ] `/history` carrega lista paginada (20/sessão) em < 500ms
- [ ] Filtros de data e busca por termo funcionam
- [ ] Modal detalhe mostra gráfico timeline + espectro
- [ ] Export PDF gera relatório formatado com métricas + gráficos
- [ ] Navegação integrada: análise → "Ver no histórico" → detalhe

---

## Fora do Escopo (Futuro)
- Autenticação JWT/OAuth multi-usuário
- WebSocket streaming tempo real
- Fine-tuning PANNs com dados próprios
- App mobile (React Native / Capacitor)
- Multi-idioma (i18n)

---

## Próximos Passos Imediatos

1. **Aprovar este plano** → `plan_exit`
2. **Iniciar Fase 1.1**: Criar `docker-compose.yml` + `Dockerfile`s
3. **Atualizar CI/CD** (`.github/workflows/`) para build/push images