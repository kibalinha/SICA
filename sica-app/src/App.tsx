import { useState, useRef, useEffect, useCallback } from 'react';
import { Header } from './components/Header';
import { ErrorMessage } from './components/ErrorMessage';
import { IdleState } from './components/IdleState';
import { RecordingState } from './components/RecordingState';
import { AnalyzingState } from './components/AnalyzingState';
import { ResultCard } from './components/ResultCard';
import {
  ANALYZE_ENDPOINT,
  HEALTH_ENDPOINT,
  RECORDING_DURATION_MS,
  API_PORT,
  API_KEY,
} from './config';
import { fetchWithRetry } from './utils/api';
import type { AnalysisResult, AppState } from './types';

function App() {
  const [appState, setAppState] = useState<AppState>('IDLE');
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [activeAnalyser, setActiveAnalyser] = useState<AnalyserNode | null>(null);
  const [recordingProgress, setRecordingProgress] = useState(0);

  const audioContextRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const recordingTimeoutRef = useRef<number | null>(null);
  const progressIntervalRef = useRef<number | null>(null);
  const cancelledRef = useRef(false);

  const clearRecordingTimers = useCallback(() => {
    if (recordingTimeoutRef.current !== null) {
      window.clearTimeout(recordingTimeoutRef.current);
      recordingTimeoutRef.current = null;
    }
    if (progressIntervalRef.current !== null) {
      window.clearInterval(progressIntervalRef.current);
      progressIntervalRef.current = null;
    }
  }, []);

  const stopAudioTracks = useCallback(() => {
    clearRecordingTimers();
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      void audioContextRef.current.close();
      audioContextRef.current = null;
    }
    setActiveAnalyser(null);
    setRecordingProgress(0);
  }, [clearRecordingTimers]);

  useEffect(() => {
    return () => {
      stopAudioTracks();
    };
  }, [stopAudioTracks]);

  const cancelRecording = useCallback(() => {
    cancelledRef.current = true;
    clearRecordingTimers();
    if (mediaRecorderRef.current?.state === 'recording') {
      mediaRecorderRef.current.stop();
    } else {
      stopAudioTracks();
      setAppState('IDLE');
    }
  }, [clearRecordingTimers, stopAudioTracks]);

  const startRealRecording = async () => {
    setErrorMsg(null);
    setAnalysis(null);
    audioChunksRef.current = [];
    cancelledRef.current = false;

    try {
      const healthResponse = await fetchWithRetry(HEALTH_ENDPOINT, {
        method: 'GET',
      });

      if (!healthResponse.ok) {
        throw new Error(`Servidor indisponível: ${healthResponse.status}`);
      }
    } catch {
      setErrorMsg(
        `Não foi possível conectar ao backend. Verifique se o servidor Python está rodando na porta ${API_PORT}.`
      );
      setAppState('IDLE');
      return;
    }

    if (!navigator.mediaDevices?.getUserMedia) {
      setErrorMsg('Seu navegador não suporta captura de microfone. Use HTTPS ou localhost.');
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        },
        video: false,
      });
      streamRef.current = stream;

      const AudioContextClass =
        window.AudioContext ||
        (window as Window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
      if (!AudioContextClass) {
        throw new Error('Web Audio API indisponível neste navegador.');
      }

      const audioCtx = new AudioContextClass();
      audioContextRef.current = audioCtx;

      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 256;
      source.connect(analyser);
      setActiveAnalyser(analyser);

      const options = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
        ? { mimeType: 'audio/webm;codecs=opus' }
        : MediaRecorder.isTypeSupported('audio/webm')
          ? { mimeType: 'audio/webm' }
          : {};

      const mediaRecorder = new MediaRecorder(stream, options);
      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        clearRecordingTimers();

        if (cancelledRef.current) {
          stopAudioTracks();
          setAppState('IDLE');
          return;
        }

        setAppState('ANALYZING');
        const audioBlob = new Blob(audioChunksRef.current, {
          type: mediaRecorder.mimeType || 'audio/webm',
        });

        try {
          const formData = new FormData();
          formData.append('file', audioBlob, 'sample.webm');

          const headers: Record<string, string> = {};
          if (API_KEY) {
            headers['X-API-Key'] = API_KEY;
          }

          const response = await fetchWithRetry(ANALYZE_ENDPOINT, {
            method: 'POST',
            headers,
            body: formData,
          });

          if (!response.ok) {
            let detail = 'Erro no processamento do servidor.';
            try {
              const errData = (await response.json()) as { detail?: string };
              detail = errData.detail || detail;
            } catch {
              // resposta não-JSON
            }
            throw new Error(detail);
          }

          const data = (await response.json()) as AnalysisResult;
          setAnalysis(data);
          setAppState('RESULT');
        } catch (err) {
          console.error('Erro na requisição para o backend:', err);
          const message =
            err instanceof Error ? err.message : 'Erro ao conectar ao servidor de processamento.';
          setErrorMsg(
            message.includes('Failed to fetch') || message.includes('Tempo limite excedido')
              ? `Não foi possível conectar ao backend. Verifique se o servidor Python está rodando na porta ${API_PORT}.`
              : message
          );
          setAppState('IDLE');
        } finally {
          stopAudioTracks();
        }
      };

      mediaRecorder.start(100);
      setAppState('RECORDING');
      setRecordingProgress(0);

      const startedAt = Date.now();
      progressIntervalRef.current = window.setInterval(() => {
        const elapsed = Date.now() - startedAt;
        setRecordingProgress(Math.min(100, (elapsed / RECORDING_DURATION_MS) * 100));
      }, 50);

      recordingTimeoutRef.current = window.setTimeout(() => {
        if (mediaRecorder.state === 'recording') {
          mediaRecorder.stop();
        }
      }, RECORDING_DURATION_MS);
    } catch (err) {
      console.error('Erro ao acessar o microfone:', err);
      stopAudioTracks();
      setErrorMsg(
        'Acesso ao microfone negado ou indisponível. Permita o uso do microfone e tente novamente.'
      );
      setAppState('IDLE');
    }
  };

  const resetAnalysis = () => {
    setAppState('IDLE');
    setAnalysis(null);
    setErrorMsg(null);
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-slate-950 p-4 font-sans text-slate-100">
      <div className="relative flex h-[880px] max-h-[92vh] w-full max-w-md flex-col overflow-hidden rounded-[2.5rem] border-4 border-slate-800 bg-slate-900 shadow-2xl sm:h-auto sm:max-h-[95vh]">
        <Header />

        <main className="flex flex-1 flex-col space-y-4 overflow-y-auto bg-slate-950 p-4 sm:p-6">
          {errorMsg && <ErrorMessage message={errorMsg} />}

          {appState === 'IDLE' && <IdleState onStartRecording={() => void startRealRecording()} />}

          {appState === 'RECORDING' && (
            <RecordingState
              recordingProgress={recordingProgress}
              activeAnalyser={activeAnalyser}
              onCancel={cancelRecording}
            />
          )}

          {appState === 'ANALYZING' && <AnalyzingState />}

          {appState === 'RESULT' && analysis && (
            <ResultCard analysis={analysis} onReset={resetAnalysis} />
          )}
        </main>
      </div>
    </div>
  );
}

export default App;
