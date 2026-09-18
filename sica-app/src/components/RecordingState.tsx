import { AudioVisualizer } from './AudioVisualizer';
import { RECORDING_DURATION_MS } from '../config';

interface RecordingStateProps {
  recordingProgress: number;
  activeAnalyser: AnalyserNode | null;
  onCancel: () => void;
}

export function RecordingState({
  recordingProgress,
  activeAnalyser,
  onCancel,
}: RecordingStateProps) {
  const recordingSeconds = Math.ceil(
    (RECORDING_DURATION_MS / 1000) * (1 - recordingProgress / 100)
  );

  return (
    <div className="sica-fade-in flex flex-1 flex-col justify-between py-4">
      <div className="space-y-3 text-center">
        <span className="inline-block animate-pulse rounded-full border border-red-800/60 bg-red-950 px-3 py-1 font-mono text-xs font-bold text-red-400">
          REC ● CAPTURANDO ({recordingSeconds}s restantes)
        </span>
        <h2 className="text-xl font-bold text-slate-200">Amostrando Ruído &amp; Música...</h2>
        <div className="h-1.5 w-full overflow-hidden rounded-full bg-slate-800">
          <div
            className="h-full bg-gradient-to-r from-red-500 to-orange-400 transition-all duration-100"
            style={{ width: `${recordingProgress}%` }}
          />
        </div>
      </div>

      <AudioVisualizer analyser={activeAnalyser} isRecording={true} />

      <div className="space-y-3 text-center">
        <p className="animate-pulse text-xs text-slate-400">
          Alimente o microfone com som do ambiente para a IA Neural e Fourier.
        </p>
        <button
          onClick={onCancel}
          aria-label="Cancelar gravação de áudio"
          className="rounded text-xs text-slate-400 underline underline-offset-2 transition-colors hover:text-red-300 focus:ring-1 focus:ring-red-400 focus:outline-none"
        >
          Cancelar gravação
        </button>
      </div>
    </div>
  );
}
