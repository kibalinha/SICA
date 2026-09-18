# SICA - Sistema de Inteligência Comercial Acústica

## Comandos Úteis para Desenvolvimento

### Backend (Python)

```bash
cd sica-backend

# Ativar ambiente virtual
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/macOS

# Instalar dependências
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Para desenvolvimento

# Executar servidor em desenvolvimento
python server.py

# Executar testes
pytest tests/ -v

# Executar testes com coverage
pytest tests/ -v --cov=. --cov-report=html

# Verificar estilo de código (ruff)
ruff check .
ruff check . --fix

# Formatar código
ruff format .
ruff format . --check

# Type checking (mypy)
mypy .
```

### Frontend (React + Vite)

```bash
cd sica-app

# Instalar dependências
npm install

# Executar em desenvolvimento
npm run dev

# Build de produção
npm run build

# Preview do build
npm run preview

# Lint
npm run lint
npm run lint:fix

# Format
npm run format
npm run format:check

# Type checking
npx tsc --noEmit --project tsconfig.app.json

# Testes
npm run test          # Modo watch
npm run test:run      # Execução única (CI)
npm run test:ui       # Interface visual
npm run test:coverage

# Bundle analyzer
npm run analyze
```

### Estrutura do Projeto

```
delightful-galileo/
├── .github/
│   └── workflows/       # GitHub Actions CI/CD
├── .husky/              # Husky pre-commit hooks
├── sica-backend/        # API FastAPI + PyTorch
│   ├── .husky/          # Backend pre-commit hooks
│   ├── ai_engine.py     # Motor de IA CNN
│   ├── audio_processor.py # Processamento de áudio
│   ├── server.py        # API FastAPI
│   ├── tests/           # Testes unitários
│   ├── pyproject.toml   # Configuração moderna Python
│   ├── requirements.txt # Dependências
│   └── requirements-dev.txt
├── sica-app/            # Frontend React
│   ├── .husky/          # Frontend pre-commit hooks
│   ├── src/
│   │   ├── components/  # Componentes React
│   │   ├── utils/       # Utilitários (API, etc)
│   │   ├── test/        # Configuração de testes
│   │   ├── App.tsx      # Componente principal
│   │   ├── types.ts     # TypeScript types
│   │   └── config.ts    # Configuração da API
│   ├── package.json     # Dependências
│   ├── tsconfig.json    # TypeScript config
│   ├── vite.config.ts   # Vite config
│   ├── eslint.config.js # ESLint config
│   └── .prettierrc      # Prettier config
└── README.md            # Documentação principal
```

### Variáveis de Ambiente

#### Backend (.env)
- `SICA_HOST`: Host do servidor (padrão: 0.0.0.0)
- `SICA_PORT`: Porta do servidor (padrão: 8002)
- `SICA_CORS_ORIGINS`: Origens CORS permitidas (padrão: http://localhost:5173,http://127.0.0.1:5173)
- `SICA_MAX_UPLOAD_MB`: Tamanho máximo de upload em MB (padrão: 10)
- `SICA_RATE_LIMIT_REQUESTS`: Limite de requisições por período (padrão: 10)
- `SICA_RATE_LIMIT_PERIOD`: Período em segundos para rate limiting (padrão: 60)
- `SICA_ENABLE_CACHE`: Habilitar cache (padrão: true)
- `SICA_CACHE_SIZE`: Tamanho máximo do cache (padrão: 100)
- `SICA_CACHE_TTL`: Tempo de vida do cache em segundos (padrão: 3600)
- `SICA_API_KEY`: API Key para autenticação (opcional)

#### Frontend (.env)
- `VITE_API_URL`: URL base da API (vazio usa proxy do Vite em dev)

### Funcionalidades Implementadas

#### Backend
- ✅ API FastAPI com documentação automática (/api/v1/docs)
- ✅ Motor de IA CNN PyTorch para classificação de áudio
- ✅ Processamento espectral (STFT, HPSS)
- ✅ Validação de arquivos e sanitização
- ✅ Rate limiting com slowapi
- ✅ Logging estruturado em JSON
- ✅ Type hints + MyPy type checking
- ✅ Cache de resultados em memória com TTL e limpeza automática
- ✅ Autenticação via API Key (opcional)
- ✅ API versionada (/api/v1/)
- ✅ Health check detalhado (sistema, modelo, cache, disco, memória)
- ✅ Testes unitários com pytest

#### Frontend
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
- ✅ Bundle analyzer (vite-bundle-analyzer)
- ✅ Husky pre-commit hooks
- ✅ Testes de componentes e utilitários

### Testes

Para executar todos os testes:

```bash
# Backend
cd sica-backend
pytest tests/ -v
pytest tests/ -v --cov=. --cov-report=html

# Frontend
cd sica-app
npm run test:run
npm run test:coverage
```

### Troubleshooting

**Backend não inicia:**
- Verifique se a porta 8000 está disponível
- Verifique se as dependências estão instaladas (`pip install -r requirements.txt`)
- Verifique se há um arquivo `.env` configurado (copie de `.env.example`)
- Verifique se o modelo `.pt` existe ou será gerado na primeira execução

**Frontend não conecta ao backend:**
- Verifique se o backend está rodando na porta 8000
- Verifique se CORS está configurado corretamente (`SICA_CORS_ORIGINS`)
- Use `VITE_API_URL=http://localhost:8000` se necessário
- Verifique se o proxy do Vite está funcionando (dev server)

**Microfone não funciona:**
- Use HTTPS ou localhost (necessário para Web Audio API)
- Verifique permissões do navegador
- Verifique se o microfone está conectado

**Erros de lint/type check:**
- Backend: `ruff check . --fix` e `mypy .`
- Frontend: `npm run lint:fix` e `npm run format`

### Melhorias Futuras

- [ ] Implementar cache distribuído (Redis) no backend
- [ ] Adicionar autenticação JWT/OAuth completa
- [ ] Implementar histórico de análises no banco de dados
- [ ] Adicionar testes E2E com Playwright
- [ ] Otimizar modelo CNN para inferência mais rápida (ONNX/TensorRT)
- [ ] Adicionar suporte para upload de arquivos pré-gravados
- [ ] Implementar exportação de relatórios em PDF
- [ ] Adicionar métricas Prometheus/Grafana
- [ ] Implementar rate limiting por API Key
- [ ] Adicionar suporte a WebSocket para streaming em tempo real