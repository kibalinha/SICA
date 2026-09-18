import { describe, it, expect, vi, beforeEach, afterEach, type Mock } from 'vitest';
import { fetchWithTimeout, fetchWithRetry, API_TIMEOUT_MS, MAX_RETRIES } from '../api';

describe('API Utilities', () => {
  beforeEach(() => {
    global.fetch = vi.fn();
  });

  afterEach(() => {
    vi.resetAllMocks();
  });

  describe('fetchWithTimeout', () => {
    it('resolves with response when fetch succeeds', async () => {
      const mockResponse = { ok: true, json: vi.fn().mockResolvedValue({ data: 'test' }) };
      (global.fetch as Mock).mockResolvedValue(mockResponse);

      const response = await fetchWithTimeout('/test');

      expect(response).toBe(mockResponse);
      const [, init] = (global.fetch as Mock).mock.calls[0] as [string, RequestInit];
      expect(init.signal).toBeInstanceOf(AbortSignal);
    });
  });

  describe('fetchWithRetry', () => {
    it('resolves on first success', async () => {
      const mockResponse = { ok: true };
      (global.fetch as Mock).mockResolvedValue(mockResponse);

      const response = await fetchWithRetry('/test');
      expect(response).toBe(mockResponse);
      expect(global.fetch).toHaveBeenCalledTimes(1);
    });

    it('retries on failure and succeeds', async () => {
      const mockResponse = { ok: true };
      (global.fetch as Mock)
        .mockRejectedValueOnce(new Error('Network error'))
        .mockResolvedValueOnce(mockResponse);

      const response = await fetchWithRetry('/test', {}, 1);
      expect(response).toBe(mockResponse);
      expect(global.fetch).toHaveBeenCalledTimes(2);
    });

    it('throws after max retries exceeded', async () => {
      (global.fetch as Mock).mockRejectedValue(new Error('Network error'));

      await expect(fetchWithRetry('/test', {}, 2)).rejects.toThrow('Network error');
      expect(global.fetch).toHaveBeenCalledTimes(3);
    });
  });

  describe('constants', () => {
    it('exports API_TIMEOUT_MS', () => {
      expect(API_TIMEOUT_MS).toBe(30000);
    });

    it('exports MAX_RETRIES', () => {
      expect(MAX_RETRIES).toBe(2);
    });
  });
});
