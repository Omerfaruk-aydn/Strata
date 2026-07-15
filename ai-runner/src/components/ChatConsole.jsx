/**
 * AI Runner — ChatConsole
 * Center panel: active chat message stream with input area.
 * Implements FR-302, FR-303, FR-306, FR-401.
 */

import { useState, useRef, useEffect, useCallback } from 'react';
import useSessionStore from '../store/useSessionStore';
import useModelStore from '../store/useModelStore';
import useSettingsStore from '../store/useSettingsStore';
import { useTranslation } from '../i18n/useTranslation';
import MessageBubble from './MessageBubble';
import './ChatConsole.css';

export default function ChatConsole() {
  const t = useTranslation();
  const {
    messages, activeSessionId, isGenerating,
    streamingContent, streamingSpeed,
    sendMessage, stopGeneration, createSession,
  } = useSessionStore();
  const activeModel = useModelStore((s) => s.activeModel);
  const defaultSystemPrompt = useSettingsStore((s) => s.defaultSystemPrompt);
  const maxContextLength = useSettingsStore((s) => s.maxContextLength) || 4096;

  const [input, setInput] = useState('');
  const [showParams, setShowParams] = useState(false);
  const [params, setParams] = useState({
    temperature: 0.7,
    topP: 0.9,
    topK: 40,
    repeatPenalty: 1.1,
    maxTokens: 2048,
    systemPrompt: '',
  });

  const [tokenUsage, setTokenUsage] = useState({
    total_used: 0,
    utilization_pct: 0,
    is_warning: false,
    is_critical: false,
  });

  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  // Fetch token budget usage from backend
  useEffect(() => {
    if (!activeSessionId || messages.length === 0) {
      setTokenUsage({ total_used: 0, utilization_pct: 0, is_warning: false, is_critical: false });
      return;
    }
    const fetchBudget = async () => {
      try {
        const systemPromptVal = params.systemPrompt || defaultSystemPrompt || '';
        const res = await fetch(`http://127.0.0.1:8420/api/optimizer/prompt-budget?context_length=${maxContextLength}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(messages.map(m => ({ role: m.role, content: m.content }))),
        });
        if (res.ok) {
          const data = await res.json();
          setTokenUsage(data);
        }
      } catch (err) {
        console.error("Budget fetch error:", err);
      }
    };
    fetchBudget();
  }, [messages, activeSessionId, maxContextLength, params.systemPrompt, defaultSystemPrompt]);

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, streamingContent]);

  // Focus input
  useEffect(() => {
    if (!isGenerating) inputRef.current?.focus();
  }, [isGenerating, activeSessionId]);

  const handleSend = useCallback(async () => {
    if (!input.trim() || !activeModel || isGenerating) return;

    // Create session if none active
    if (!activeSessionId) {
      await createSession(input.slice(0, 40) + (input.length > 40 ? '...' : ''));
    }

    const message = input.trim();
    setInput('');
    sendMessage(message, {
      ...params,
      systemPrompt: params.systemPrompt || defaultSystemPrompt,
    });
  }, [input, activeModel, isGenerating, activeSessionId, params, defaultSystemPrompt, createSession, sendMessage]);

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
    if (e.key === 'Escape' && isGenerating) {
      stopGeneration();
    }
  };

  return (
    <main className="chat-console" id="chat-console">
      {/* Messages Area */}
      <div className="chat-messages">
        {messages.length === 0 && !streamingContent ? (
          <div className="chat-empty">
            <div className="chat-empty-icon">💬</div>
            <h3 className="text-heading">{t('app.title')}</h3>
            <p className="text-body" style={{ color: 'var(--text-secondary)' }}>
              {activeModel ? t('chat.empty') : t('chat.no_model')}
            </p>
            {activeModel && (
              <div className="chat-suggestions">
                {['Merhaba, kendini tanıtır mısın?', 'Python ile quicksort yaz', 'Yapay zeka nedir kısaca açıkla'].map((suggestion, i) => (
                  <button
                    key={i}
                    className="suggestion-chip"
                    onClick={() => {
                      setInput(suggestion);
                      inputRef.current?.focus();
                    }}
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            )}
          </div>
        ) : (
          <>
            {messages.map((msg) => (
              <MessageBubble
                key={msg.id}
                role={msg.role}
                content={msg.content}
                tokensGenerated={msg.tokens_generated || 0}
              />
            ))}

            {/* Streaming message */}
            {streamingContent && (
              <MessageBubble
                role="assistant"
                content={streamingContent}
                streaming={true}
              />
            )}
          </>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Generation Status Bar */}
      {isGenerating && (
        <div className="generation-bar">
          <div className="generation-status">
            <div className="generation-pulse" />
            <span className="text-small">{t('chat.generating')}</span>
          </div>
          {streamingSpeed > 0 && (
            <span className="text-mono generation-speed">
              {streamingSpeed.toFixed(1)} {t('chat.tokens_per_sec')}
            </span>
          )}
        </div>
      )}

      {/* Input Area */}
      <div className="chat-input-area">
        {/* Parameter Toggle */}
        <div className="input-toolbar">
          <button
            className={`btn btn-ghost btn-sm ${showParams ? 'active' : ''}`}
            onClick={() => setShowParams(!showParams)}
            title="Parametreler"
          >
            ⚙️
          </button>
          {activeModel && (
            <>
              <span className="text-small active-model-badge">
                🟢 {activeModel.model_id?.split('/').pop() || 'Model'}
              </span>
              <div className="context-indicator" title={`Bağlam Kullanımı: %${tokenUsage.utilization_pct || 0} (${tokenUsage.total_used || 0}/${maxContextLength} Token)`}>
                <span className="text-small" style={{ fontSize: '11px', opacity: 0.8 }}>
                  🧠 {tokenUsage.total_used || 0} / {maxContextLength} Token
                </span>
                <div className="context-bar-bg">
                  <div
                    className={`context-bar-fg ${tokenUsage.is_warning ? 'warning' : ''} ${tokenUsage.is_critical ? 'critical' : ''}`}
                    style={{ width: `${Math.min(100, tokenUsage.utilization_pct || 0)}%` }}
                  />
                </div>
              </div>
            </>
          )}
        </div>

        {/* Parameters Panel (FR-306) */}
        {showParams && (
          <div className="params-panel animate-fade-in">
            <div className="param-row">
              <label className="text-small">Temperature</label>
              <input
                type="range"
                min="0"
                max="2"
                step="0.1"
                value={params.temperature}
                onChange={(e) => setParams({ ...params, temperature: parseFloat(e.target.value) })}
              />
              <span className="text-mono param-value">{params.temperature}</span>
            </div>
            <div className="param-row">
              <label className="text-small">Top-P</label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={params.topP}
                onChange={(e) => setParams({ ...params, topP: parseFloat(e.target.value) })}
              />
              <span className="text-mono param-value">{params.topP}</span>
            </div>
            <div className="param-row">
              <label className="text-small">Top-K</label>
              <input
                type="number"
                min="1"
                max="100"
                value={params.topK}
                onChange={(e) => setParams({ ...params, topK: parseInt(e.target.value) || 40 })}
                className="param-number-input"
              />
            </div>
            <div className="param-row">
              <label className="text-small">Repeat Penalty</label>
              <input
                type="range"
                min="1"
                max="2"
                step="0.05"
                value={params.repeatPenalty}
                onChange={(e) => setParams({ ...params, repeatPenalty: parseFloat(e.target.value) })}
              />
              <span className="text-mono param-value">{params.repeatPenalty}</span>
            </div>
            <div className="param-row">
              <label className="text-small">Max Tokens</label>
              <input
                type="number"
                min="64"
                max="8192"
                step="64"
                value={params.maxTokens}
                onChange={(e) => setParams({ ...params, maxTokens: parseInt(e.target.value) || 2048 })}
                className="param-number-input"
              />
            </div>
            <div className="param-row param-row-full">
              <label className="text-small">System Prompt</label>
              <textarea
                rows={2}
                value={params.systemPrompt}
                onChange={(e) => setParams({ ...params, systemPrompt: e.target.value })}
                placeholder={defaultSystemPrompt || 'Sistem promptu (opsiyonel)...'}
                className="param-textarea"
              />
            </div>
          </div>
        )}

        {/* Message Input */}
        <div className="input-row">
          <textarea
            ref={inputRef}
            className="chat-input"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={activeModel ? t('chat.placeholder') : t('chat.no_model')}
            disabled={!activeModel}
            rows={1}
            id="chat-input"
          />
          {isGenerating ? (
            <button
              className="btn btn-danger send-btn"
              onClick={stopGeneration}
              title={t('chat.stop') + ' (Esc)'}
            >
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                <rect x="3" y="3" width="10" height="10" rx="2" fill="currentColor"/>
              </svg>
            </button>
          ) : (
            <button
              className="btn btn-primary send-btn"
              onClick={handleSend}
              disabled={!input.trim() || !activeModel}
              title={t('chat.send') + ' (Enter)'}
            >
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                <path d="M14.5 1.5L7 9M14.5 1.5L10 14.5L7 9M14.5 1.5L1.5 6L7 9" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            </button>
          )}
        </div>
      </div>
    </main>
  );
}
