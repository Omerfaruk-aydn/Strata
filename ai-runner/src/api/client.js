/** Central API configuration, authentication, and error handling. */

const DEFAULT_HOST = import.meta.env.VITE_API_HOST || '127.0.0.1';
const DEFAULT_PORT = Number(import.meta.env.VITE_API_PORT || 8420);
const STORAGE_KEY = 'ai-runner.api-config.v1';
const PENDING_ENDPOINT_KEY = 'ai-runner.pending-api-endpoint.v1';
const SESSION_MARKER_KEY = 'ai-runner.api-session.v1';

export class ApiError extends Error {
  constructor(message, status = 0, data = null) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.data = data;
  }
}

function readConfig() {
  try {
    const stored = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}');
    return {
      host: stored.host || DEFAULT_HOST,
      port: Number(stored.port) || DEFAULT_PORT,
      apiKey: stored.apiKey || null,
    };
  } catch {
    return { host: DEFAULT_HOST, port: DEFAULT_PORT, apiKey: null };
  }
}

function promotePendingEndpointOnNewSession() {
  try {
    if (sessionStorage.getItem(SESSION_MARKER_KEY)) return;
    sessionStorage.setItem(SESSION_MARKER_KEY, 'active');

    const pending = JSON.parse(localStorage.getItem(PENDING_ENDPOINT_KEY) || 'null');
    if (!pending) return;

    const current = readConfig();
    localStorage.setItem(STORAGE_KEY, JSON.stringify({
      ...current,
      host: pending.host || current.host,
      port: Number(pending.port) || current.port,
    }));
    localStorage.removeItem(PENDING_ENDPOINT_KEY);
  } catch {
    // Storage can be unavailable in hardened browser contexts; defaults remain usable.
  }
}

promotePendingEndpointOnNewSession();

function clientHost(host) {
  const value = String(host || DEFAULT_HOST).trim();
  const connectable = value === '0.0.0.0' || value === '::' || value === '[::]'
    ? '127.0.0.1'
    : value;
  return connectable.includes(':') && !connectable.startsWith('[')
    ? `[${connectable}]`
    : connectable;
}

export function getApiConfig() {
  return readConfig();
}

export function configureApi({ host, port, apiKey } = {}) {
  const current = readConfig();
  const next = {
    host: host ?? current.host,
    port: Number(port ?? current.port) || DEFAULT_PORT,
    apiKey: apiKey === undefined ? current.apiKey : (apiKey || null),
  };
  localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
  if (JSON.stringify(current) !== JSON.stringify(next)) {
    window.dispatchEvent(new CustomEvent('ai-runner:api-config-changed', { detail: next }));
  }
  return next;
}

/** Save a host/port change for the next application launch. */
export function scheduleApiEndpoint({ host, port }) {
  const current = readConfig();
  const pending = {
    host: host || current.host,
    port: Number(port) || current.port,
  };
  localStorage.setItem(PENDING_ENDPOINT_KEY, JSON.stringify(pending));
  return pending;
}

export function getApiBase() {
  const { host, port } = readConfig();
  return `http://${clientHost(host)}:${port}`;
}

export function getWebSocketUrl(path = '/ws/inference') {
  const { host, port } = readConfig();
  return new URL(`ws://${clientHost(host)}:${port}${path}`).toString();
}

function base64UrlEncode(value) {
  const bytes = new TextEncoder().encode(value);
  let binary = '';
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/g, '');
}

/** Keep WebSocket credentials out of URLs and access logs. */
export function getWebSocketConnection(path = '/ws/inference') {
  const { apiKey } = readConfig();
  return {
    url: getWebSocketUrl(path),
    protocols: apiKey
      ? ['ai-runner', `ai-runner-key.${base64UrlEncode(apiKey)}`]
      : [],
  };
}

/** Wait for the packaged backend to finish extracting and start listening. */
export async function waitForApi({ timeoutMs = 90_000, intervalMs = 500 } = {}) {
  const deadline = Date.now() + timeoutMs;
  let lastError = null;

  while (Date.now() < deadline) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 2_000);
    try {
      const response = await fetch(`${getApiBase()}/`, {
        headers: { Accept: 'application/json' },
        signal: controller.signal,
      });
      if (response.ok) return true;
      lastError = new ApiError(`Backend hazır değil (${response.status}).`, response.status);
    } catch (error) {
      lastError = error;
    } finally {
      clearTimeout(timer);
    }
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }

  throw new ApiError(
    'AI Runner backend başlatılamadı. API adresini ve uygulama günlüklerini kontrol edin.',
    0,
    lastError,
  );
}

async function errorFromResponse(response) {
  let data = null;
  try {
    data = await response.clone().json();
  } catch {
    try {
      data = await response.clone().text();
    } catch {
      data = null;
    }
  }

  const detail = typeof data === 'object' && data
    ? data.detail || data.error || data.message
    : data;
  const detailMessage = typeof detail === 'string'
    ? detail
    : Array.isArray(detail?.blockers)
      ? detail.blockers.join(' ')
      : (detail?.message || null);
  const fallback = response.status === 401
    ? 'API anahtarı geçersiz veya eksik.'
    : `API isteği başarısız (${response.status}).`;
  return new ApiError(detailMessage || fallback, response.status, data);
}

export async function apiFetch(path, options = {}) {
  const config = readConfig();
  const headers = new Headers(options.headers || {});
  headers.set('Accept', headers.get('Accept') || 'application/json');
  if (config.apiKey) headers.set('Authorization', `Bearer ${config.apiKey}`);

  const response = await fetch(`${getApiBase()}${path}`, { ...options, headers });
  if (!response.ok) throw await errorFromResponse(response);
  return response;
}

export async function* readSse(response) {
  if (!response.body) throw new ApiError('Sunucu boş bir akış döndürdü.', response.status);

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
      const events = buffer.split(/\r?\n\r?\n/);
      buffer = events.pop() || '';

      for (const event of events) {
        const payload = event
          .split(/\r?\n/)
          .filter((line) => line.startsWith('data:'))
          .map((line) => line.slice(5).trimStart())
          .join('\n');
        if (!payload) continue;
        if (payload === '[DONE]') return;
        try {
          yield JSON.parse(payload);
        } catch {
          throw new ApiError('Sunucudan geçersiz bir SSE olayı alındı.', response.status, payload);
        }
      }

      if (done) break;
    }
  } finally {
    reader.releaseLock();
  }
}
