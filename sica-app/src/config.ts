export const DEFAULT_API_BASE_URL = 'http://localhost:8002';

const env = import.meta.env as Record<string, unknown>;
const envApiUrl = typeof env['VITE_API_URL'] === 'string' ? env['VITE_API_URL'].trim() : '';

export const API_BASE_URL = envApiUrl || DEFAULT_API_BASE_URL;
export const API_PORT = Number.parseInt(new URL(API_BASE_URL).port || '8002', 10);
const apiVersion = 'v1';

export const ANALYZE_ENDPOINT = `${API_BASE_URL}/api/${apiVersion}/analyze`;
export const HEALTH_ENDPOINT = `${API_BASE_URL}/api/${apiVersion}/health`;
export const RECORDING_DURATION_MS = 4500;
