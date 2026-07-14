/**
 * AI Runner — ModelShelf
 * Left panel: installed/downloadable model list with search.
 * Implements FR-101–FR-106.
 */

import { useState, useEffect, useCallback } from 'react';
import useModelStore from '../store/useModelStore';
import { useTranslation } from '../i18n/useTranslation';
import ModelCard from './ModelCard';
import './ModelShelf.css';

export default function ModelShelf({ collapsed = false }) {
  const t = useTranslation();
  const {
    localModels, searchResults, searchQuery,
    isSearching, fetchLocalModels, searchModels, clearSearch,
  } = useModelStore();

  const [query, setQuery] = useState('');
  const [activeTab, setActiveTab] = useState('local'); // 'local' | 'search'

  // Fetch local models on mount
  useEffect(() => {
    fetchLocalModels();
  }, [fetchLocalModels]);

  // Debounced search
  const handleSearch = useCallback((value) => {
    setQuery(value);
    if (value.trim().length >= 2) {
      setActiveTab('search');
      const timer = setTimeout(() => searchModels(value.trim()), 400);
      return () => clearTimeout(timer);
    } else if (value.trim().length === 0) {
      setActiveTab('local');
      clearSearch();
    }
  }, [searchModels, clearSearch]);

  if (collapsed) return null;

  return (
    <aside className="model-shelf" id="model-shelf">
      {/* Header */}
      <div className="shelf-header">
        <h3 className="text-subheading">{t('models.shelf_title')}</h3>
      </div>

      {/* Search */}
      <div className="shelf-search">
        <div className="search-input-wrapper">
          <svg className="search-icon" width="16" height="16" viewBox="0 0 16 16" fill="none">
            <path d="M7.333 12.667A5.333 5.333 0 1 0 7.333 2a5.333 5.333 0 0 0 0 10.667ZM14 14l-2.9-2.9" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
          <input
            type="text"
            className="search-input"
            placeholder={t('models.search_placeholder')}
            value={query}
            onChange={(e) => handleSearch(e.target.value)}
            id="model-search-input"
          />
          {query && (
            <button
              className="search-clear"
              onClick={() => { setQuery(''); clearSearch(); setActiveTab('local'); }}
              aria-label="Clear search"
            >
              ✕
            </button>
          )}
        </div>
      </div>

      {/* Tabs */}
      <div className="shelf-tabs">
        <button
          className={`shelf-tab ${activeTab === 'local' ? 'active' : ''}`}
          onClick={() => setActiveTab('local')}
        >
          {t('models.local_models')}
          <span className="tab-count">{localModels.length}</span>
        </button>
        <button
          className={`shelf-tab ${activeTab === 'search' ? 'active' : ''}`}
          onClick={() => setActiveTab('search')}
        >
          {t('models.hub_results')}
          {searchResults.length > 0 && <span className="tab-count">{searchResults.length}</span>}
        </button>
      </div>

      {/* Model List */}
      <div className="shelf-list">
        {activeTab === 'local' ? (
          localModels.length > 0 ? (
            localModels.map((model) => (
              <ModelCard key={model.id} model={model} isLocal={true} />
            ))
          ) : (
            <div className="shelf-empty">
              <span className="shelf-empty-icon">📦</span>
              <p className="text-small">{t('models.no_local')}</p>
            </div>
          )
        ) : (
          <>
            {isSearching && (
              <div className="shelf-loading">
                <div className="loading-pulse" />
                <span className="text-small">Aranıyor...</span>
              </div>
            )}
            {!isSearching && searchResults.length > 0 && (
              searchResults.map((model) => (
                <ModelCard key={model.id} model={model} isLocal={false} />
              ))
            )}
            {!isSearching && searchResults.length === 0 && query.length >= 2 && (
              <div className="shelf-empty">
                <span className="shelf-empty-icon">🔍</span>
                <p className="text-small">{t('models.no_results')}</p>
              </div>
            )}
          </>
        )}
      </div>
    </aside>
  );
}
