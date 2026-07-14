/**
 * AI Runner — MessageBubble
 * Individual message rendering with Markdown support, code highlighting, and copy button.
 */

import { useState, useMemo } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { useTranslation } from '../i18n/useTranslation';
import './MessageBubble.css';

export default function MessageBubble({ role, content, streaming = false, tokensGenerated = 0 }) {
  const t = useTranslation();
  const [copied, setCopied] = useState(false);

  const isUser = role === 'user';

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(content);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (e) { /* fallback */ }
  };

  const markdownComponents = useMemo(() => ({
    code({ node, inline, className, children, ...props }) {
      const match = /language-(\w+)/.exec(className || '');
      const codeString = String(children).replace(/\n$/, '');

      if (!inline && match) {
        return (
          <div className="code-block-wrapper">
            <div className="code-block-header">
              <span className="code-lang">{match[1]}</span>
              <button
                className="code-copy-btn"
                onClick={() => navigator.clipboard.writeText(codeString)}
              >
                Kopyala
              </button>
            </div>
            <SyntaxHighlighter
              style={oneDark}
              language={match[1]}
              PreTag="div"
              customStyle={{
                margin: 0,
                borderRadius: '0 0 8px 8px',
                fontSize: '13px',
                background: '#1a1b26',
              }}
              {...props}
            >
              {codeString}
            </SyntaxHighlighter>
          </div>
        );
      }

      return (
        <code className="inline-code" {...props}>
          {children}
        </code>
      );
    },
    p({ children }) {
      return <p className="message-paragraph">{children}</p>;
    },
    ul({ children }) {
      return <ul className="message-list">{children}</ul>;
    },
    ol({ children }) {
      return <ol className="message-list message-list-ordered">{children}</ol>;
    },
    blockquote({ children }) {
      return <blockquote className="message-blockquote">{children}</blockquote>;
    },
    table({ children }) {
      return (
        <div className="message-table-wrapper">
          <table className="message-table">{children}</table>
        </div>
      );
    },
  }), []);

  return (
    <div className={`message-bubble ${isUser ? 'message-user' : 'message-assistant'}`}>
      {/* Avatar */}
      <div className={`message-avatar ${isUser ? 'avatar-user' : 'avatar-assistant'}`}>
        {isUser ? '👤' : '🤖'}
      </div>

      {/* Content */}
      <div className="message-content">
        <div className="message-body">
          {isUser ? (
            <p className="message-paragraph">{content}</p>
          ) : (
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={markdownComponents}
            >
              {content}
            </ReactMarkdown>
          )}

          {/* Streaming cursor */}
          {streaming && (
            <span className="streaming-cursor">
              <span className="cursor-dot" />
              <span className="cursor-dot" />
              <span className="cursor-dot" />
            </span>
          )}
        </div>

        {/* Actions */}
        <div className="message-actions">
          {!isUser && !streaming && (
            <button
              className="btn btn-ghost btn-sm message-action-btn"
              onClick={handleCopy}
              title={t('chat.copy')}
            >
              {copied ? '✓ ' + t('chat.copied') : '📋 ' + t('chat.copy')}
            </button>
          )}
          {tokensGenerated > 0 && (
            <span className="message-meta text-small">
              {tokensGenerated} {t('chat.tokens')}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
