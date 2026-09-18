import react from '@vitejs/plugin-react';
import { defineConfig, loadEnv } from 'vite';
import tailwindcss from '@tailwindcss/vite';
import { analyzer } from 'vite-bundle-analyzer';

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');

  return {
    plugins: [
      react(),
      tailwindcss(),
      mode === 'analyze' && analyzer({
        analyzerMode: 'static',
        openAnalyzer: false,
        fileName: 'bundle-report.html',
      }),
    ].filter(Boolean),
    server: {
      proxy: {
        '/api': {
          target: env['VITE_API_URL'] || 'http://localhost:8002',
          changeOrigin: true,
        },
      },
    },
    build: {
      rollupOptions: {
        output: {
          manualChunks(id: string) {
            if (id.includes('node_modules/react') || id.includes('node_modules/react-dom')) {
              return 'vendor';
            }
            return undefined;
          },
        },
      },
      chunkSizeWarningLimit: 500,
    },
  };
});