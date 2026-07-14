/**
 * AI Runner — SessionList
 * Chat session list in sidebar with rename, delete, pin operations.
 * Implements FR-701–FR-703.
 */

import { useState } from 'react';
import useSessionStore from '../store/useSessionStore';
import { useTranslation } from '../i18n/useTranslation';
import './SessionList.css';

export default function SessionList() {
  const t = useTranslation();
  const {
    sessions, activeSessionId, createSession,
    selectSession, renameSession, deleteSession, togglePin,
    exportSession,
  } = useSessionStore();

  const [editingId, setEditingId] = useState(null);
  const [editTitle, setEditTitle] = useState('');
  const [contextMenu, setContextMenu] = useState(null);

  const handleNew = () => {
    createSession();
  };

  const handleStartRename = (session) => {
    setEditingId(session.id);
    setEditTitle(session.title);
    setContextMenu(null);
  };

  const handleFinishRename = (id) => {
    if (editTitle.trim()) {
      renameSession(id, editTitle.trim());
    }
    setEditingId(null);
  };

  const handleContextMenu = (e, session) => {
    e.preventDefault();
    e.stopPropagation();
    setContextMenu({
      x: e.clientX,
      y: e.clientY,
      session,
    });
  };

  const closeContextMenu = () => setContextMenu(null);

  const pinnedSessions = sessions.filter(s => s.pinned);
  const unpinnedSessions = sessions.filter(s => !s.pinned);

  return (
    <div className="session-list" onClick={closeContextMenu}>
      {/* New Chat Button */}
      <button
        className="btn btn-primary new-chat-btn"
        onClick={handleNew}
        id="new-chat-btn"
      >
        <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
          <path d="M7 1v12M1 7h12" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
        </svg>
        {t('chat.new_chat')}
      </button>

      {/* Pinned Sessions */}
      {pinnedSessions.length > 0 && (
        <div className="session-group">
          <span className="session-group-label text-small">📌 {t('chat.pin')}</span>
          {pinnedSessions.map(session => (
            <SessionItem
              key={session.id}
              session={session}
              isActive={session.id === activeSessionId}
              isEditing={editingId === session.id}
              editTitle={editTitle}
              onSelect={() => selectSession(session.id)}
              onContextMenu={(e) => handleContextMenu(e, session)}
              onEditChange={setEditTitle}
              onEditFinish={() => handleFinishRename(session.id)}
            />
          ))}
        </div>
      )}

      {/* Regular Sessions */}
      <div className="session-group">
        {pinnedSessions.length > 0 && (
          <span className="session-group-label text-small">{t('chat.sessions')}</span>
        )}
        {unpinnedSessions.map(session => (
          <SessionItem
            key={session.id}
            session={session}
            isActive={session.id === activeSessionId}
            isEditing={editingId === session.id}
            editTitle={editTitle}
            onSelect={() => selectSession(session.id)}
            onContextMenu={(e) => handleContextMenu(e, session)}
            onEditChange={setEditTitle}
            onEditFinish={() => handleFinishRename(session.id)}
          />
        ))}
      </div>

      {sessions.length === 0 && (
        <div className="session-empty">
          <span className="text-small" style={{ color: 'var(--text-tertiary)' }}>
            {t('chat.empty')}
          </span>
        </div>
      )}

      {/* Context Menu */}
      {contextMenu && (
        <div
          className="session-context-menu"
          style={{ top: contextMenu.y, left: contextMenu.x }}
          onClick={(e) => e.stopPropagation()}
        >
          <button onClick={() => { handleStartRename(contextMenu.session); }}>
            ✏️ {t('chat.rename')}
          </button>
          <button onClick={() => { togglePin(contextMenu.session.id); closeContextMenu(); }}>
            {contextMenu.session.pinned ? '📌 ' + t('chat.unpin') : '📌 ' + t('chat.pin')}
          </button>
          <button onClick={() => { exportSession(contextMenu.session.id, 'markdown'); closeContextMenu(); }}>
            📄 {t('chat.export_md')}
          </button>
          <button onClick={() => { exportSession(contextMenu.session.id, 'json'); closeContextMenu(); }}>
            📋 {t('chat.export_json')}
          </button>
          <div className="context-divider" />
          <button
            className="context-danger"
            onClick={() => { deleteSession(contextMenu.session.id); closeContextMenu(); }}
          >
            🗑️ {t('chat.delete')}
          </button>
        </div>
      )}
    </div>
  );
}

function SessionItem({
  session, isActive, isEditing, editTitle,
  onSelect, onContextMenu, onEditChange, onEditFinish,
}) {
  return (
    <div
      className={`session-item ${isActive ? 'session-active' : ''}`}
      onClick={onSelect}
      onContextMenu={onContextMenu}
      role="button"
      tabIndex={0}
    >
      {isEditing ? (
        <input
          className="session-rename-input"
          value={editTitle}
          onChange={(e) => onEditChange(e.target.value)}
          onBlur={onEditFinish}
          onKeyDown={(e) => {
            if (e.key === 'Enter') onEditFinish();
            if (e.key === 'Escape') onEditFinish();
          }}
          autoFocus
          onClick={(e) => e.stopPropagation()}
        />
      ) : (
        <>
          <span className="session-title truncate">{session.title}</span>
          <span className="session-date text-small">
            {new Date(session.updated_at).toLocaleDateString()}
          </span>
        </>
      )}
    </div>
  );
}
