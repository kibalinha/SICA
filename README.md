# SICA — Sistema de Inteligência Comercial Acústica

Aplicação full-stack para análise de som ambiente em espaços comerciais (shoppings, lojas). Captura áudio via microfone, classifica a cena acústica (música, vozes, ruído) e recomenda ajustes de volume em dB.

## Arquitetura

```
sica-app/       → Frontend React + Vite + Tailwind + TypeScript
sica-backend/   → API FastAPI + librosa + PyTorch
```

## Características

### Backend
- ✅ API FastAPI com documentação automática (/api/v1/docs)
- ✅ Motor de IA CNN PyTorch para classificação de áudio (PANNs Cnn14 / AudioSet)
- ✅ Processamento espectral (STFT, HPSS)
- ✅ Loudness LUFS (EBU R128, K-weighting) via pyloudnorm
- ✅ Score de musicalidade (onsets, BPM, spectral rolloff)
- ✅ Suavização temporal EMA entre janelas
- ✅ Validação de arquivos e sanitização
- ✅ Rate limiting com slowapi
- ✅ Logging estruturado em JSON
- ✅ Type hints + MyPy type checking
- ✅ Cache de resultados em memória com TTL
- ✅ Autenticação via API Key (opcional)
- ✅ API versionada (/api/v1/)
- ✅ Health check detalhado (sistema, modelo, cache)
- ✅ Testes unitários com pytest
- ✅ Ruff para linting/formatting

### Frontend
- ✅ Interface React 19 com Vite
- ✅ Captura de áudio via Web Audio API
- ✅ Visualização em tempo real (FFT)
- ✅ Componentes reutilizáveis e modulares
- ✅ Error Boundary
- ✅ Melhorias de acessibilidade (ARIA, suporte a teclado)
- ✅ Tratamento de erros com retry e timeout
- ✅ Design responsivo com Tailwind CSS v4
- ✅ TypeScript strict mode
- ✅ ESLint + Prettier
- ✅ Vitest + React Testing Library
- ✅ Bundle analyzer
- ✅ Husky pre-commit hooks

## Pré-requisitos

- **Node.js** 20+
- **Python** 3.10+
- Microfone no dispositivo

## Instalação e execução

### 1. Backend (Python)

```bash
cd sica-backend
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
pip install -r requirements-dev.txt  # Para desenvolvimento
python server.py
```

API disponível em `http://localhost:8002`
Documentação interativa: `http://localhost:8002/api/v1/docs`

### 2. Frontend (React)

```bash
cd sica-app
npm install
npm run dev
```

Interface em `http://localhost:5173`

O Vite faz proxy de `/api/*` para o backend automaticamente em desenvolvimento.

## Variáveis de ambiente

### Backend (`sica-backend/.env`)

| Variável | Padrão | Descrição |
|----------|--------|-----------|
| `SICA_HOST` | `0.0.0.0` | Host do servidor |
| `SICA_PORT` | `8002` | Porta do servidor |
| `SICA_CORS_ORIGINS` | `http://localhost:5173,...` | Origens permitidas (separadas por vírgula) |
| `SICA_MAX_UPLOAD_MB` | `10` | Tamanho máximo de upload |
| `SICA_RATE_LIMIT_REQUESTS` | `10` | Limite de requisições por período |
| `SICA_RATE_LIMIT_PERIOD` | `60` | Período em segundos para rate limiting |
| `SICA_ENABLE_CACHE` | `true` | Habilitar cache de resultados |
| `SICA_CACHE_SIZE` | `100` | Tamanho máximo do cache |
| `SICA_CACHE_TTL` | `3600` | Tempo de vida do cache em segundos |
| `SICA_API_KEY` | *(vazio)* | API Key para autenticação (opcional) |

### Frontend (`sica-app/.env`)

| Variável | Padrão | Descrição |
|----------|--------|-----------|
| `VITE_API_URL` | *(vazio)* | URL base da API. Vazio usa proxy relativo `/api` |

## Endpoints

| Método | Rota | Descrição |
|--------|------|-----------|
| GET | `/` | Status da API |
| GET | `/api/v1/health` | Health check detalhado |
| POST | `/api/v1/analyze` | Análise de áudio (multipart `file`) |

## Autenticação

Se `SICA_API_KEY` estiver definido no backend, inclua o header `X-API-Key` nas requisições:

```bash
curl -X POST http://localhost:8002/api/v1/analyze \
  -H "X-API-Key: your-api-key" \
  -F "file=@audio.webm"
```

## Pipeline de análise

1. **Captura** — 4,5s de áudio via MediaRecorder (WebM/Opus)
2. **STFT** — Transformada de Fourier de tempo curto
3. **HPSS** — Separação harmônica vs percussiva/ruído
4. **IA** — PANNs Cnn14 (AudioSet, 527 classes) em PyTorch, com fallback heurístico (música / voz / ruído)
5. **Métricas** — dB(A), LUFS (EBU R128, K-weighting), SNR, planicidade espectral, score de musicalidade (onsets/BPM/rolloff)
6. **Recomendação** — Ajuste de volume ±6 dB com base no SNR alvo, LUFS e classificação semântica

## Testes

### Backend

```bash
cd sica-backend
pip install -r requirements-dev.txt
pytest tests/ -v
# Com coverage
pytest tests/ -v --cov=. --cov-report=html
```

### Frontend

```bash
cd sica-app
npm install
npm run test        # Modo watch
npm run test:run    # Execução única
npm run test:ui     # Interface visual
npm run test:coverage
```

## Scripts úteis

```bash
# Frontend
npm run build          # Build de produção
npm run lint           # ESLint
npm run lint:fix       # ESLint auto-fix
npm run format         # Prettier format
npm run format:check   # Prettier check
npm run preview        # Preview do build
npm run analyze        # Bundle analyzer
npm run test:run       # Testes CI

# Backend
python server.py       # Servidor com hot-reload
ruff check .           # Linting
ruff check . --fix     # Auto-fix
ruff format .          # Formatar código
ruff format . --check  # Verificar formatação
mypy .                 # Type checking
pytest tests/ -v       # Executar testes
```

## CI/CD

GitHub Actions workflows:
- **Backend CI**: lint, type-check, tests, coverage
- **Frontend CI**: lint, format, type-check, tests, build, bundle analysis
- **Full CI**: Combined pipeline with integration test placeholder

## Limitações conhecidas

- Níveis dB(A) são **estimativas** (não substituem sonômetro calibrado)
- Modelo CNN calibrado com arquétipos sintéticos — retreinar com dados reais melhora precisão
- Microfone requer contexto seguro (HTTPS ou localhost)

## Licença

Projeto privado — uso interno.