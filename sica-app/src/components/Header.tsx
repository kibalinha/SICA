export function Header() {
  return (
    <header className="z-10 rounded-b-3xl border-b border-blue-700/50 bg-gradient-to-r from-blue-900 to-indigo-900 p-6 pt-10 text-center shadow-lg">
      <div className="flex items-center justify-center space-x-2">
        <span className="h-3 w-3 animate-ping rounded-full bg-emerald-400" />
        <h1 className="text-3xl font-black tracking-wider text-white">SICA</h1>
      </div>
      <p className="mt-1 text-xs font-medium tracking-wide text-blue-200">
        Deep Learning (PyTorch CNN) &amp; Fourier (STFT)
      </p>
    </header>
  );
}
