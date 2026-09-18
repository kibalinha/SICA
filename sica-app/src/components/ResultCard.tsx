import type { AnalysisResult } from '../types';

interface ResultCardProps {
  analysis: AnalysisResult;
  onReset: () => void;
}

export function ResultCard({ analysis, onReset }: ResultCardProps) {
  return (
    <div className="sica-fade-in-slow flex flex-1 flex-col space-y-4 overflow-y-auto pr-1">
      <RecommendationCard recommendation={analysis.recommendation} />

      {analysis.ai_diagnostics && <AIDiagnosticsCard diagnostics={analysis.ai_diagnostics} />}

      <MetricsGrid metrics={analysis.metrics} />

      {analysis.metrics.band_dba && <FrequencyBandsGrid bandDba={analysis.metrics.band_dba} />}

      <button
        onClick={onReset}
        aria-label="Iniciar nova análise"
        className="w-full rounded-xl border border-slate-700 bg-slate-800 px-6 py-3 font-bold text-white transition-all hover:bg-slate-700 focus:ring-2 focus:ring-blue-400 focus:ring-offset-2 focus:ring-offset-slate-900 focus:outline-none"
      >
        Nova Análise
      </button>
    </div>
  );
}

function RecommendationCard({
  recommendation,
}: {
  recommendation: AnalysisResult['recommendation'];
}) {
  return (
    <div className="relative overflow-hidden rounded-2xl border border-slate-800 bg-slate-900 p-4 text-center shadow-lg">
      <div className="mb-1 font-mono text-xs tracking-wider text-slate-400 uppercase">
        Recomendação Inteligente SICA
      </div>

      <div className="my-2 flex items-center justify-center space-x-2">
        <span
          className={`text-5xl font-black tracking-tighter ${
            recommendation.adjustment_db < 0
              ? 'text-red-400'
              : recommendation.adjustment_db > 0
                ? 'text-emerald-400'
                : 'text-blue-400'
          }`}
        >
          {recommendation.adjustment_db > 0 ? '+' : ''}
          {recommendation.adjustment_db}
        </span>
        <span className="text-xl font-bold text-slate-400">dB</span>
      </div>

      <p className="rounded-lg border border-slate-800 bg-slate-950/60 p-2 text-xs font-medium text-slate-300">
        {recommendation.status}
      </p>
    </div>
  );
}

function AIDiagnosticsCard({ diagnostics }: { diagnostics: AnalysisResult['ai_diagnostics'] }) {
  return (
    <div className="space-y-3 rounded-2xl border border-slate-800 bg-slate-900 p-4 shadow-lg">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-1.5">
          <span className="h-2 w-2 animate-pulse rounded-full bg-indigo-400" />
          <span className="font-mono text-[10px] font-bold tracking-wider text-indigo-300 uppercase">
            {diagnostics.model_name}
          </span>
        </div>
        <span className="rounded-full border border-indigo-800 bg-indigo-950 px-2 py-0.5 font-mono text-[10px] text-indigo-300">
          {Math.round(diagnostics.confidence * 100)}% Confiança
        </span>
      </div>

      <div className="flex items-center justify-between rounded-xl border border-slate-800 bg-slate-950/80 p-2.5">
        <div>
          <span className="block font-mono text-[10px] text-slate-400">
            CENA ACÚSTICA DETECTADA
          </span>
          <span className="text-sm font-bold text-slate-100">{diagnostics.scene_description}</span>
        </div>
        <span
          className={`rounded-lg px-2.5 py-1 text-xs font-bold ${
            diagnostics.dominant_scene === 'Música'
              ? 'border border-emerald-800 bg-emerald-950 text-emerald-300'
              : diagnostics.dominant_scene.includes('Voz')
                ? 'border border-blue-800 bg-blue-950 text-blue-300'
                : 'border border-amber-800 bg-amber-950 text-amber-300'
          }`}
        >
          {diagnostics.dominant_scene}
        </span>
      </div>

      <ProbabilityBars probabilities={diagnostics.probabilities} />

      {diagnostics.neural_mask_applied && (
        <div className="flex items-center space-x-1.5 rounded-lg border border-indigo-900/60 bg-indigo-950/40 p-2 text-[10px] text-indigo-300">
          <span>⚡</span>
          <span>
            Máscara Neural Ativa: Harmônicos vocais humanos atenuados para isolar a música real.
          </span>
        </div>
      )}
    </div>
  );
}

function ProbabilityBars({
  probabilities,
}: {
  probabilities: AnalysisResult['ai_diagnostics']['probabilities'];
}) {
  return (
    <div className="space-y-2 pt-1 font-mono text-xs">
      <ProbabilityBar label="🎵 Música Ambiente" value={probabilities.music} color="emerald" />
      <ProbabilityBar label="🗣️ Conversação / Vozes" value={probabilities.speech} color="blue" />
      <ProbabilityBar label="🏢 Ruído Residual" value={probabilities.ambient_noise} color="amber" />
    </div>
  );
}

function ProbabilityBar({
  label,
  value,
  color,
}: {
  label: string;
  value: number;
  color: 'emerald' | 'blue' | 'amber';
}) {
  const colorClasses = {
    emerald: 'from-emerald-600 to-emerald-400 text-emerald-400 text-emerald-300',
    blue: 'from-blue-600 to-indigo-400 text-blue-400 text-blue-300',
    amber: 'from-amber-600 to-amber-400 text-amber-400 text-amber-300',
  };

  const [from, to, textColor, textBoldColor] = colorClasses[color].split(' ');

  return (
    <div>
      <div className="mb-1 flex justify-between text-[11px]">
        <span className={`${textColor} font-medium`}>{label}</span>
        <span className={`${textBoldColor} font-bold`}>{value}%</span>
      </div>
      <div className="h-2 w-full overflow-hidden rounded-full border border-slate-800 bg-slate-950">
        <div
          className={`h-full bg-gradient-to-r ${from} ${to} rounded-full transition-all duration-500`}
          style={{ width: `${Math.min(100, Math.max(2, value))}%` }}
        />
      </div>
    </div>
  );
}

function MetricsGrid({ metrics }: { metrics: AnalysisResult['metrics'] }) {
  return (
    <div className="grid grid-cols-2 gap-2 font-mono text-xs">
      <MetricCard label="NÍVEL TOTAL dB(A)" value={`${metrics.total_dba} dBA`} />
      <MetricCard
        label="RELAÇÃO SINAL/RUÍDO (SNR)"
        value={`${metrics.snr_db} dB`}
        valueColor={metrics.snr_db > 0 ? 'text-emerald-400' : 'text-orange-400'}
      />
      <MetricCard
        label="MÚSICA HARMÔNICA (ESTIMADA)"
        value={`${metrics.harmonic_music_dba} dBA`}
        labelColor="text-emerald-400"
      />
      <MetricCard
        label="RUÍDO PERCUSSIVO"
        value={`${metrics.noise_dba} dBA`}
        labelColor="text-amber-400"
      />
    </div>
  );
}

function MetricCard({
  label,
  value,
  labelColor = 'text-slate-400',
  valueColor = 'text-white',
}: {
  label: string;
  value: string;
  labelColor?: string;
  valueColor?: string;
}) {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900 p-3">
      <span className={`${labelColor} block text-[10px]`}>{label}</span>
      <span className={`text-base font-bold ${valueColor}`}>{value}</span>
    </div>
  );
}

function FrequencyBandsGrid({ bandDba }: { bandDba: Record<string, number> }) {
  // Definir as bandas na ordem desejada para exibição
  const bandOrder = ['sub-bass', 'bass', 'low-mid', 'mid', 'upper-mid', 'high', 'air'];
  const bandLabels: Record<string, string> = {
    'sub-bass': 'Sub-Bass (20-100Hz)',
    'bass': 'Bass (100-300Hz)',
    'low-mid': 'Low-Mid (300-800Hz)',
    'mid': 'Mid (800-2kHz)',
    'upper-mid': 'Upper-Mid (2k-4kHz)',
    'high': 'High (4k-8kHz)',
    'air': 'Air (8k-20kHz)'
  };
  const bandColors: Record<string, string> = {
    'sub-bass': 'bg-blue-900',
    'bass': 'bg-blue-700',
    'low-mid': 'bg-blue-500',
    'mid': 'bg-indigo-500',
    'upper-mid': 'bg-indigo-400',
    'high': 'bg-violet-500',
    'air': 'bg-violet-300'
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between mb-2">
        <span className="font-mono text-xs text-slate-400 uppercase">
          ANÁLISE POR BANDAS DE FREQUÊNCIA (dBA)
        </span>
        <span className="text-xs text-slate-500">
          {Object.keys(bandDba).length} bandas analisadas
        </span>
      </div>
      <div className="space-y-2">
        {bandOrder.map(band => {
          const value = bandDba[band];
          // Ignorar bandas sem dados (-Infinity) ou valores undefined
          if (value === -Infinity || value === undefined) return null;
          
          const label = bandLabels[band] || band;
          const color = bandColors[band] || 'bg-slate-600';
          
          // Determinar a cor do texto baseado no valor
          let textColor = 'text-white';
          if (value < 40) textColor = 'text-blue-300';
          else if (value < 50) textColor = 'text-blue-200';
          else if (value < 60) textColor = 'text-slate-200';
          else if (value < 70) textColor = 'text-slate-100';
          else textColor = 'text-red-200';
          
          return (
            <div key={band} className="flex items-center justify-between">
              <div className="flex-1 space-x-2">
                <span className="block text-xs font-mono">{label}</span>
                <span className={`block text-[10px] font-bold ${textColor}`}>
                  {value.toFixed(1)} dBA
                </span>
              </div>
              <div className="w-1/2">
                <div className="h-2 w-full rounded-full border border-slate-800 bg-slate-950">
                  <div
                    className={`h-full ${color} rounded-full transition-all duration-500`}
                    style={{ width: `${Math.min(100, Math.max(0, value))}%` }}
                  />
                </div>
              </div>
            </div>
          );
        }).filter(Boolean)}
      </div>
    </div>
  );
}
