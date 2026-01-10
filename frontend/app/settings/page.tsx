'use client';

import React, { useState, useEffect } from 'react';
import {
  Settings,
  Cpu,
  Key,
  Database,
  Zap,
  Save,
  RefreshCw,
  Check,
  X,
  AlertCircle,
  ArrowLeft,
  Eye,
  EyeOff,
  ChevronDown,
  ChevronUp,
  Power,
  Loader2,
} from 'lucide-react';
import Link from 'next/link';

// API base URL
const API_BASE = process.env.NEXT_PUBLIC_API_URL || '';

// Types
interface LLMProvider {
  enabled: boolean;
  default_model: string;
  models: Record<string, string>;
  has_api_key: boolean;
}

interface LLMConfig {
  default_provider: string;
  default_temperature: number;
  default_max_tokens: number;
  providers: Record<string, LLMProvider>;
}

interface Credential {
  key: string;
  is_set: boolean;
  masked: string | null;
  length: number;
}

interface CredentialsConfig {
  credentials: Record<string, Credential>;
  categories: Record<string, string[]>;
}

interface DataSource {
  enabled?: boolean;
  [key: string]: any;
}

interface RuntimeConfig {
  runtime: {
    hot_reload?: boolean;
    watch_interval_seconds?: number;
    cache?: {
      enabled?: boolean;
      ttl_seconds?: number;
    };
  };
  hot_reload_active: boolean;
  config_path: string;
}

interface APIKey {
  display_name: string;
  category: string;
  is_set: boolean;
  masked: string | null;
  source: string | null;
  updated_at?: string;
}

interface APIKeysConfig {
  api_keys: Record<string, APIKey>;
  database_connected: boolean;
}

type TabType = 'llm' | 'api-keys' | 'credentials' | 'data-sources' | 'runtime';

export default function SettingsPage() {
  const [activeTab, setActiveTab] = useState<TabType>('llm');
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  // LLM Config State
  const [llmConfig, setLlmConfig] = useState<LLMConfig | null>(null);

  // Credentials State
  const [credentials, setCredentials] = useState<CredentialsConfig | null>(null);
  const [credentialInputs, setCredentialInputs] = useState<Record<string, string>>({});
  const [showCredential, setShowCredential] = useState<Record<string, boolean>>({});
  const [savingCredential, setSavingCredential] = useState<string | null>(null);

  // Data Sources State
  const [dataSources, setDataSources] = useState<Record<string, DataSource> | null>(null);

  // Runtime State
  const [runtimeConfig, setRuntimeConfig] = useState<RuntimeConfig | null>(null);

  // API Keys State
  const [apiKeysConfig, setApiKeysConfig] = useState<APIKeysConfig | null>(null);
  const [apiKeyInputs, setApiKeyInputs] = useState<Record<string, string>>({});
  const [showApiKey, setShowApiKey] = useState<Record<string, boolean>>({});
  const [savingApiKey, setSavingApiKey] = useState<string | null>(null);

  // Fetch data on mount
  useEffect(() => {
    fetchAllConfig();
  }, []);

  // Clear messages after timeout
  useEffect(() => {
    if (success) {
      const timer = setTimeout(() => setSuccess(null), 3000);
      return () => clearTimeout(timer);
    }
  }, [success]);

  useEffect(() => {
    if (error) {
      const timer = setTimeout(() => setError(null), 5000);
      return () => clearTimeout(timer);
    }
  }, [error]);

  const fetchAllConfig = async () => {
    setLoading(true);
    try {
      await Promise.all([
        fetchLLMConfig(),
        fetchCredentials(),
        fetchDataSources(),
        fetchRuntimeConfig(),
        fetchApiKeys(),
      ]);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch configuration');
    } finally {
      setLoading(false);
    }
  };

  const fetchLLMConfig = async () => {
    const response = await fetch(`${API_BASE}/config/llm`);
    if (!response.ok) throw new Error('Failed to fetch LLM config');
    const data = await response.json();
    setLlmConfig(data);
  };

  const fetchCredentials = async () => {
    const response = await fetch(`${API_BASE}/config/credentials`);
    if (!response.ok) throw new Error('Failed to fetch credentials');
    const data = await response.json();
    setCredentials(data);
  };

  const fetchDataSources = async () => {
    const response = await fetch(`${API_BASE}/config/data-sources`);
    if (!response.ok) throw new Error('Failed to fetch data sources');
    const data = await response.json();
    setDataSources(data.data_sources);
  };

  const fetchRuntimeConfig = async () => {
    const response = await fetch(`${API_BASE}/config/runtime`);
    if (!response.ok) throw new Error('Failed to fetch runtime config');
    const data = await response.json();
    setRuntimeConfig(data);
  };

  const fetchApiKeys = async () => {
    const response = await fetch(`${API_BASE}/api-keys`);
    if (!response.ok) throw new Error('Failed to fetch API keys');
    const data = await response.json();
    setApiKeysConfig(data);
  };

  // LLM Config Handlers
  const updateLLMConfig = async (updates: any) => {
    setSaving(true);
    try {
      const response = await fetch(`${API_BASE}/config/llm`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(updates),
      });
      if (!response.ok) throw new Error('Failed to update LLM config');
      await fetchLLMConfig();
      setSuccess('LLM configuration updated');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update');
    } finally {
      setSaving(false);
    }
  };

  // Credential Handlers
  const saveCredential = async (credentialId: string) => {
    const value = credentialInputs[credentialId];
    if (!value) return;

    setSavingCredential(credentialId);
    try {
      const response = await fetch(`${API_BASE}/config/credentials/${credentialId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ value }),
      });
      if (!response.ok) throw new Error('Failed to update credential');

      await fetchCredentials();
      setCredentialInputs(prev => ({ ...prev, [credentialId]: '' }));
      setSuccess(`Credential ${credentialId} updated`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update credential');
    } finally {
      setSavingCredential(null);
    }
  };

  // API Key Handlers
  const saveApiKey = async (keyName: string) => {
    const value = apiKeyInputs[keyName];
    if (!value) return;

    setSavingApiKey(keyName);
    try {
      const response = await fetch(`${API_BASE}/api-keys/${keyName}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ key_name: keyName, key_value: value }),
      });
      if (!response.ok) throw new Error('Failed to update API key');

      await fetchApiKeys();
      setApiKeyInputs(prev => ({ ...prev, [keyName]: '' }));
      setSuccess(`API key ${keyName} updated - changes take effect immediately!`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update API key');
    } finally {
      setSavingApiKey(null);
    }
  };

  const deleteApiKey = async (keyName: string) => {
    if (!confirm(`Delete ${keyName} from database? Will fall back to environment variable if set.`)) {
      return;
    }

    setSavingApiKey(keyName);
    try {
      const response = await fetch(`${API_BASE}/api-keys/${keyName}`, {
        method: 'DELETE',
      });
      if (!response.ok) throw new Error('Failed to delete API key');

      await fetchApiKeys();
      setSuccess(`API key ${keyName} deleted from database`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete API key');
    } finally {
      setSavingApiKey(null);
    }
  };

  // Data Source Handlers
  const toggleDataSource = async (sourceId: string, enabled: boolean) => {
    setSaving(true);
    try {
      const response = await fetch(`${API_BASE}/config/data-sources/${sourceId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled }),
      });
      if (!response.ok) throw new Error('Failed to update data source');
      await fetchDataSources();
      setSuccess(`Data source ${sourceId} ${enabled ? 'enabled' : 'disabled'}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update data source');
    } finally {
      setSaving(false);
    }
  };

  // Runtime Handlers
  const updateRuntimeConfig = async (updates: any) => {
    setSaving(true);
    try {
      const response = await fetch(`${API_BASE}/config/runtime`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(updates),
      });
      if (!response.ok) throw new Error('Failed to update runtime config');
      await fetchRuntimeConfig();
      setSuccess('Runtime configuration updated');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update');
    } finally {
      setSaving(false);
    }
  };

  const forceReload = async () => {
    setSaving(true);
    try {
      const response = await fetch(`${API_BASE}/config/reload`, { method: 'POST' });
      if (!response.ok) throw new Error('Failed to reload config');
      await fetchAllConfig();
      setSuccess('Configuration reloaded');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to reload');
    } finally {
      setSaving(false);
    }
  };

  const tabs = [
    { id: 'llm' as TabType, label: 'LLM Providers', icon: Cpu },
    { id: 'api-keys' as TabType, label: 'API Keys', icon: Key },
    { id: 'credentials' as TabType, label: 'Credentials (File)', icon: Key },
    { id: 'data-sources' as TabType, label: 'Data Sources', icon: Database },
    { id: 'runtime' as TabType, label: 'Runtime', icon: Zap },
  ];

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-studio-bg">
        <div className="flex items-center gap-3 text-studio-muted">
          <Loader2 className="w-6 h-6 animate-spin" />
          Loading configuration...
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex flex-col bg-studio-bg text-studio-text">
      {/* Header */}
      <header className="border-b border-studio-border bg-studio-panel">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Link
              href="/"
              className="flex items-center gap-2 text-studio-muted hover:text-studio-text transition-colors"
            >
              <ArrowLeft className="w-4 h-4" />
              Back
            </Link>
            <div className="flex items-center gap-2">
              <Settings className="w-5 h-5 text-studio-accent" />
              <h1 className="text-lg font-semibold">Settings</h1>
            </div>
          </div>
          <button
            onClick={forceReload}
            disabled={saving}
            className="flex items-center gap-2 px-3 py-1.5 text-sm bg-studio-panel hover:bg-studio-border border border-studio-border rounded transition-colors disabled:opacity-50"
          >
            <RefreshCw className={`w-4 h-4 ${saving ? 'animate-spin' : ''}`} />
            Reload Config
          </button>
        </div>
      </header>

      {/* Messages */}
      {error && (
        <div className="max-w-7xl mx-auto px-4 mt-4">
          <div className="p-3 bg-red-900/30 border border-red-800 rounded-lg flex items-center gap-2 text-red-400">
            <AlertCircle className="w-4 h-4" />
            {error}
          </div>
        </div>
      )}
      {success && (
        <div className="max-w-7xl mx-auto px-4 mt-4">
          <div className="p-3 bg-green-900/30 border border-green-800 rounded-lg flex items-center gap-2 text-green-400">
            <Check className="w-4 h-4" />
            {success}
          </div>
        </div>
      )}

      {/* Tab Navigation */}
      <div className="border-b border-studio-border bg-studio-panel">
        <div className="max-w-7xl mx-auto px-4">
          <nav className="flex gap-1">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
                  activeTab === tab.id
                    ? 'border-studio-accent text-studio-accent'
                    : 'border-transparent text-studio-muted hover:text-studio-text'
                }`}
              >
                <tab.icon className="w-4 h-4" />
                {tab.label}
              </button>
            ))}
          </nav>
        </div>
      </div>

      {/* Tab Content */}
      <main className="flex-1 max-w-7xl mx-auto w-full px-4 py-6">
        {activeTab === 'llm' && llmConfig && (
          <LLMTab config={llmConfig} onUpdate={updateLLMConfig} saving={saving} />
        )}
        {activeTab === 'api-keys' && apiKeysConfig && (
          <APIKeysTab
            config={apiKeysConfig}
            inputs={apiKeyInputs}
            setInputs={setApiKeyInputs}
            showKey={showApiKey}
            setShowKey={setShowApiKey}
            onSave={saveApiKey}
            onDelete={deleteApiKey}
            savingKey={savingApiKey}
          />
        )}
        {activeTab === 'credentials' && credentials && (
          <CredentialsTab
            credentials={credentials}
            inputs={credentialInputs}
            setInputs={setCredentialInputs}
            showCredential={showCredential}
            setShowCredential={setShowCredential}
            onSave={saveCredential}
            savingCredential={savingCredential}
          />
        )}
        {activeTab === 'data-sources' && dataSources && (
          <DataSourcesTab sources={dataSources} onToggle={toggleDataSource} saving={saving} />
        )}
        {activeTab === 'runtime' && runtimeConfig && (
          <RuntimeTab config={runtimeConfig} onUpdate={updateRuntimeConfig} saving={saving} />
        )}
      </main>
    </div>
  );
}

// LLM Tab Component
function LLMTab({
  config,
  onUpdate,
  saving,
}: {
  config: LLMConfig;
  onUpdate: (updates: any) => void;
  saving: boolean;
}) {
  const [localConfig, setLocalConfig] = useState(config);

  useEffect(() => {
    setLocalConfig(config);
  }, [config]);

  const handleProviderToggle = (provider: string, enabled: boolean) => {
    onUpdate({ providers: { [provider]: { enabled } } });
  };

  const handleDefaultProviderChange = (provider: string) => {
    onUpdate({ default_provider: provider });
  };

  const handleTemperatureChange = (temp: number) => {
    onUpdate({ default_temperature: temp });
  };

  const handleMaxTokensChange = (tokens: number) => {
    onUpdate({ default_max_tokens: tokens });
  };

  const providerNames: Record<string, string> = {
    groq: 'Groq',
    openai: 'OpenAI',
    anthropic: 'Anthropic',
  };

  return (
    <div className="space-y-6">
      {/* Default Settings */}
      <div className="p-4 rounded-lg border border-studio-border bg-studio-panel">
        <h3 className="text-sm font-medium text-studio-muted mb-4">Default Settings</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Default Provider */}
          <div>
            <label className="block text-sm text-studio-muted mb-1">Default Provider</label>
            <select
              value={config.default_provider}
              onChange={(e) => handleDefaultProviderChange(e.target.value)}
              disabled={saving}
              className="w-full bg-black/50 border border-studio-border rounded px-3 py-2 text-sm focus:outline-none focus:border-studio-accent disabled:opacity-50"
            >
              {Object.entries(config.providers).map(([id, provider]) => (
                <option key={id} value={id} disabled={!provider.enabled}>
                  {providerNames[id] || id} {!provider.enabled && '(disabled)'}
                </option>
              ))}
            </select>
          </div>

          {/* Temperature */}
          <div>
            <label className="block text-sm text-studio-muted mb-1">
              Temperature: {config.default_temperature.toFixed(1)}
            </label>
            <input
              type="range"
              min="0"
              max="2"
              step="0.1"
              value={config.default_temperature}
              onChange={(e) => handleTemperatureChange(parseFloat(e.target.value))}
              disabled={saving}
              className="w-full accent-studio-accent"
            />
          </div>

          {/* Max Tokens */}
          <div>
            <label className="block text-sm text-studio-muted mb-1">Max Tokens</label>
            <input
              type="number"
              value={config.default_max_tokens}
              onChange={(e) => handleMaxTokensChange(parseInt(e.target.value) || 2000)}
              disabled={saving}
              className="w-full bg-black/50 border border-studio-border rounded px-3 py-2 text-sm focus:outline-none focus:border-studio-accent disabled:opacity-50"
            />
          </div>
        </div>
      </div>

      {/* Provider Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {Object.entries(config.providers).map(([id, provider]) => (
          <div
            key={id}
            className={`p-4 rounded-lg border ${
              provider.enabled ? 'border-studio-accent bg-studio-accent/5' : 'border-studio-border bg-studio-panel'
            }`}
          >
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <Cpu className={`w-5 h-5 ${provider.enabled ? 'text-studio-accent' : 'text-studio-muted'}`} />
                <span className="font-medium">{providerNames[id] || id}</span>
              </div>
              <button
                onClick={() => handleProviderToggle(id, !provider.enabled)}
                disabled={saving}
                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                  provider.enabled ? 'bg-studio-accent' : 'bg-studio-border'
                } disabled:opacity-50`}
              >
                <span
                  className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                    provider.enabled ? 'translate-x-6' : 'translate-x-1'
                  }`}
                />
              </button>
            </div>

            <div className="space-y-2 text-sm">
              <div className="flex items-center gap-2">
                <Key className="w-3 h-3 text-studio-muted" />
                <span className={provider.has_api_key ? 'text-green-400' : 'text-red-400'}>
                  API Key: {provider.has_api_key ? 'Set' : 'Not Set'}
                </span>
              </div>
              <div className="text-studio-muted">
                Models: {Object.keys(provider.models).length}
              </div>
              {provider.enabled && (
                <div className="mt-2 pt-2 border-t border-studio-border">
                  <label className="block text-xs text-studio-muted mb-1">Default Model</label>
                  <select
                    value={provider.default_model}
                    onChange={(e) =>
                      onUpdate({ providers: { [id]: { default_model: e.target.value } } })
                    }
                    disabled={saving}
                    className="w-full bg-black/50 border border-studio-border rounded px-2 py-1 text-xs focus:outline-none focus:border-studio-accent disabled:opacity-50"
                  >
                    {Object.entries(provider.models).map(([alias, model]) => (
                      <option key={alias} value={alias}>
                        {alias}: {model}
                      </option>
                    ))}
                  </select>
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// Credentials Tab Component
function CredentialsTab({
  credentials,
  inputs,
  setInputs,
  showCredential,
  setShowCredential,
  onSave,
  savingCredential,
}: {
  credentials: CredentialsConfig;
  inputs: Record<string, string>;
  setInputs: React.Dispatch<React.SetStateAction<Record<string, string>>>;
  showCredential: Record<string, boolean>;
  setShowCredential: React.Dispatch<React.SetStateAction<Record<string, boolean>>>;
  onSave: (credentialId: string) => void;
  savingCredential: string | null;
}) {
  const categoryIcons: Record<string, any> = {
    'LLM Providers': Cpu,
    'Data Sources': Database,
    'Database': Database,
    'Observability': Zap,
  };

  return (
    <div className="space-y-6">
      <div className="p-4 rounded-lg border border-yellow-800/50 bg-yellow-900/20">
        <div className="flex items-start gap-2">
          <AlertCircle className="w-5 h-5 text-yellow-400 mt-0.5" />
          <div className="text-sm">
            <p className="text-yellow-400 font-medium">Security Note</p>
            <p className="text-yellow-400/80 mt-1">
              Credentials are stored in the .env file. Actual values are never displayed - only masked previews.
              Changes take effect immediately.
            </p>
          </div>
        </div>
      </div>

      {Object.entries(credentials.categories).map(([category, credIds]) => {
        const Icon = categoryIcons[category] || Key;
        return (
          <div key={category} className="p-4 rounded-lg border border-studio-border bg-studio-panel">
            <div className="flex items-center gap-2 mb-4">
              <Icon className="w-5 h-5 text-studio-accent" />
              <h3 className="font-medium">{category}</h3>
            </div>
            <div className="space-y-3">
              {credIds.map((credId) => {
                const cred = credentials.credentials[credId];
                if (!cred) return null;

                return (
                  <div key={credId} className="flex items-center gap-3 py-2 border-b border-studio-border last:border-0">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="text-sm font-medium">{cred.key}</span>
                        {cred.is_set ? (
                          <span className="flex items-center gap-1 text-xs text-green-400">
                            <Check className="w-3 h-3" /> Set
                          </span>
                        ) : (
                          <span className="flex items-center gap-1 text-xs text-red-400">
                            <X className="w-3 h-3" /> Not Set
                          </span>
                        )}
                      </div>
                      <div className="flex gap-2">
                        <div className="relative flex-1">
                          <input
                            type={showCredential[credId] ? 'text' : 'password'}
                            value={inputs[credId] || ''}
                            onChange={(e) =>
                              setInputs((prev) => ({ ...prev, [credId]: e.target.value }))
                            }
                            placeholder={cred.masked || 'Enter new value...'}
                            className="w-full bg-black/50 border border-studio-border rounded px-3 py-1.5 text-sm focus:outline-none focus:border-studio-accent pr-10"
                          />
                          <button
                            onClick={() =>
                              setShowCredential((prev) => ({
                                ...prev,
                                [credId]: !prev[credId],
                              }))
                            }
                            className="absolute right-2 top-1/2 -translate-y-1/2 text-studio-muted hover:text-studio-text"
                          >
                            {showCredential[credId] ? (
                              <EyeOff className="w-4 h-4" />
                            ) : (
                              <Eye className="w-4 h-4" />
                            )}
                          </button>
                        </div>
                        <button
                          onClick={() => onSave(credId)}
                          disabled={!inputs[credId] || savingCredential === credId}
                          className="flex items-center gap-1 px-3 py-1.5 text-sm bg-studio-accent hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed rounded transition-colors"
                        >
                          {savingCredential === credId ? (
                            <Loader2 className="w-4 h-4 animate-spin" />
                          ) : (
                            <Save className="w-4 h-4" />
                          )}
                          Save
                        </button>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        );
      })}
    </div>
  );
}

// Data Sources Tab Component
function DataSourcesTab({
  sources,
  onToggle,
  saving,
}: {
  sources: Record<string, DataSource>;
  onToggle: (sourceId: string, enabled: boolean) => void;
  saving: boolean;
}) {
  const sourceInfo: Record<string, { name: string; description: string }> = {
    sec_edgar: { name: 'SEC EDGAR', description: 'US public company filings' },
    finnhub: { name: 'Finnhub', description: 'Stock and market data' },
    tavily: { name: 'Tavily Search', description: 'AI-optimized web search' },
    web_search: { name: 'Web Search', description: 'General web search' },
    courtlistener: { name: 'CourtListener', description: 'Legal records and cases' },
    opencorporates: { name: 'OpenCorporates', description: 'Company registry data' },
    opensanctions: { name: 'OpenSanctions', description: 'Sanctions and PEP data' },
  };

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      {Object.entries(sources).map(([id, source]) => {
        const info = sourceInfo[id] || { name: id, description: 'Data source' };
        const isEnabled = source.enabled !== false;

        return (
          <div
            key={id}
            className={`p-4 rounded-lg border ${
              isEnabled ? 'border-studio-accent bg-studio-accent/5' : 'border-studio-border bg-studio-panel'
            }`}
          >
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <Database className={`w-5 h-5 ${isEnabled ? 'text-studio-accent' : 'text-studio-muted'}`} />
                <span className="font-medium">{info.name}</span>
              </div>
              <button
                onClick={() => onToggle(id, !isEnabled)}
                disabled={saving}
                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                  isEnabled ? 'bg-studio-accent' : 'bg-studio-border'
                } disabled:opacity-50`}
              >
                <span
                  className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                    isEnabled ? 'translate-x-6' : 'translate-x-1'
                  }`}
                />
              </button>
            </div>
            <p className="text-sm text-studio-muted">{info.description}</p>
            {source.rate_limit && (
              <p className="text-xs text-studio-muted mt-2">
                Rate limit: {source.rate_limit}/sec
              </p>
            )}
          </div>
        );
      })}
    </div>
  );
}

// API Keys Tab Component
function APIKeysTab({
  config,
  inputs,
  setInputs,
  showKey,
  setShowKey,
  onSave,
  onDelete,
  savingKey,
}: {
  config: APIKeysConfig;
  inputs: Record<string, string>;
  setInputs: React.Dispatch<React.SetStateAction<Record<string, string>>>;
  showKey: Record<string, boolean>;
  setShowKey: React.Dispatch<React.SetStateAction<Record<string, boolean>>>;
  onSave: (keyName: string) => void;
  onDelete: (keyName: string) => void;
  savingKey: string | null;
}) {
  const categoryIcons: Record<string, any> = {
    llm: Cpu,
    search: Database,
    data: Database,
  };

  // Group keys by category
  const keysByCategory: Record<string, [string, APIKey][]> = {};
  Object.entries(config.api_keys).forEach(([keyName, keyInfo]) => {
    const cat = keyInfo.category || 'other';
    if (!keysByCategory[cat]) keysByCategory[cat] = [];
    keysByCategory[cat].push([keyName, keyInfo]);
  });

  const categoryNames: Record<string, string> = {
    llm: 'LLM Providers',
    search: 'Search APIs',
    data: 'Data Sources',
    other: 'Other',
  };

  return (
    <div className="space-y-6">
      {/* Info Banner */}
      <div className="p-4 rounded-lg border border-green-800/50 bg-green-900/20">
        <div className="flex items-start gap-2">
          <Zap className="w-5 h-5 text-green-400 mt-0.5" />
          <div className="text-sm">
            <p className="text-green-400 font-medium">Runtime API Keys</p>
            <p className="text-green-400/80 mt-1">
              API keys updated here are stored in the database and take effect <strong>immediately</strong> without restarting.
              They override environment variables. Perfect for rotating keys or fixing rate limit issues.
            </p>
          </div>
        </div>
      </div>

      {/* Database Status */}
      <div className="flex items-center gap-2 text-sm">
        <Database className={`w-4 h-4 ${config.database_connected ? 'text-green-400' : 'text-red-400'}`} />
        <span className={config.database_connected ? 'text-green-400' : 'text-red-400'}>
          Database: {config.database_connected ? 'Connected' : 'Not Connected'}
        </span>
      </div>

      {/* Keys by Category */}
      {Object.entries(keysByCategory).map(([category, keys]) => {
        const Icon = categoryIcons[category] || Key;
        return (
          <div key={category} className="p-4 rounded-lg border border-studio-border bg-studio-panel">
            <div className="flex items-center gap-2 mb-4">
              <Icon className="w-5 h-5 text-studio-accent" />
              <h3 className="font-medium">{categoryNames[category] || category}</h3>
            </div>
            <div className="space-y-4">
              {keys.map(([keyName, keyInfo]) => (
                <div key={keyName} className="py-3 border-b border-studio-border last:border-0">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="text-sm font-medium">{keyInfo.display_name}</span>
                    {keyInfo.is_set ? (
                      <span className="flex items-center gap-1 text-xs text-green-400">
                        <Check className="w-3 h-3" /> Active
                      </span>
                    ) : (
                      <span className="flex items-center gap-1 text-xs text-red-400">
                        <X className="w-3 h-3" /> Not Set
                      </span>
                    )}
                    {keyInfo.source && (
                      <span className={`text-xs px-2 py-0.5 rounded ${
                        keyInfo.source === 'database'
                          ? 'bg-blue-900/50 text-blue-400'
                          : 'bg-gray-700 text-gray-400'
                      }`}>
                        {keyInfo.source}
                      </span>
                    )}
                  </div>

                  {keyInfo.masked && (
                    <div className="text-xs text-studio-muted mb-2 font-mono">
                      Current: {keyInfo.masked}
                    </div>
                  )}

                  <div className="flex gap-2">
                    <div className="relative flex-1">
                      <input
                        type={showKey[keyName] ? 'text' : 'password'}
                        value={inputs[keyName] || ''}
                        onChange={(e) =>
                          setInputs((prev) => ({ ...prev, [keyName]: e.target.value }))
                        }
                        placeholder="Enter new API key..."
                        className="w-full bg-black/50 border border-studio-border rounded px-3 py-2 text-sm focus:outline-none focus:border-studio-accent pr-10"
                      />
                      <button
                        onClick={() =>
                          setShowKey((prev) => ({
                            ...prev,
                            [keyName]: !prev[keyName],
                          }))
                        }
                        className="absolute right-2 top-1/2 -translate-y-1/2 text-studio-muted hover:text-studio-text"
                      >
                        {showKey[keyName] ? (
                          <EyeOff className="w-4 h-4" />
                        ) : (
                          <Eye className="w-4 h-4" />
                        )}
                      </button>
                    </div>
                    <button
                      onClick={() => onSave(keyName)}
                      disabled={!inputs[keyName] || savingKey === keyName}
                      className="flex items-center gap-1 px-4 py-2 text-sm bg-studio-accent hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed rounded transition-colors"
                    >
                      {savingKey === keyName ? (
                        <Loader2 className="w-4 h-4 animate-spin" />
                      ) : (
                        <Save className="w-4 h-4" />
                      )}
                      Save
                    </button>
                    {keyInfo.source === 'database' && (
                      <button
                        onClick={() => onDelete(keyName)}
                        disabled={savingKey === keyName}
                        className="flex items-center gap-1 px-3 py-2 text-sm bg-red-900/50 hover:bg-red-800 text-red-400 disabled:opacity-50 disabled:cursor-not-allowed rounded transition-colors"
                        title="Remove from database (will fall back to environment variable)"
                      >
                        <X className="w-4 h-4" />
                      </button>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}

// Runtime Tab Component
function RuntimeTab({
  config,
  onUpdate,
  saving,
}: {
  config: RuntimeConfig;
  onUpdate: (updates: any) => void;
  saving: boolean;
}) {
  const runtime = config.runtime || {};
  const cache = runtime.cache || {};

  return (
    <div className="space-y-6">
      {/* Hot Reload Settings */}
      <div className="p-4 rounded-lg border border-studio-border bg-studio-panel">
        <div className="flex items-center gap-2 mb-4">
          <RefreshCw className="w-5 h-5 text-studio-accent" />
          <h3 className="font-medium">Hot Reload</h3>
        </div>
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium">Enable Hot Reload</p>
              <p className="text-xs text-studio-muted">Automatically reload config when file changes</p>
            </div>
            <button
              onClick={() => onUpdate({ hot_reload: !runtime.hot_reload })}
              disabled={saving}
              className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                runtime.hot_reload ? 'bg-studio-accent' : 'bg-studio-border'
              } disabled:opacity-50`}
            >
              <span
                className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                  runtime.hot_reload ? 'translate-x-6' : 'translate-x-1'
                }`}
              />
            </button>
          </div>
          <div>
            <label className="block text-sm text-studio-muted mb-1">
              Watch Interval: {runtime.watch_interval_seconds || 5} seconds
            </label>
            <input
              type="range"
              min="1"
              max="60"
              value={runtime.watch_interval_seconds || 5}
              onChange={(e) => onUpdate({ watch_interval_seconds: parseInt(e.target.value) })}
              disabled={saving || !runtime.hot_reload}
              className="w-full accent-studio-accent disabled:opacity-50"
            />
          </div>
        </div>
      </div>

      {/* Cache Settings */}
      <div className="p-4 rounded-lg border border-studio-border bg-studio-panel">
        <div className="flex items-center gap-2 mb-4">
          <Zap className="w-5 h-5 text-studio-accent" />
          <h3 className="font-medium">Cache</h3>
        </div>
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium">Enable Cache</p>
              <p className="text-xs text-studio-muted">Cache API responses to reduce calls</p>
            </div>
            <button
              onClick={() => onUpdate({ cache_enabled: !cache.enabled })}
              disabled={saving}
              className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                cache.enabled ? 'bg-studio-accent' : 'bg-studio-border'
              } disabled:opacity-50`}
            >
              <span
                className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                  cache.enabled ? 'translate-x-6' : 'translate-x-1'
                }`}
              />
            </button>
          </div>
          <div>
            <label className="block text-sm text-studio-muted mb-1">Cache TTL (seconds)</label>
            <input
              type="number"
              value={cache.ttl_seconds || 300}
              onChange={(e) => onUpdate({ cache_ttl_seconds: parseInt(e.target.value) || 300 })}
              disabled={saving || !cache.enabled}
              className="w-full bg-black/50 border border-studio-border rounded px-3 py-2 text-sm focus:outline-none focus:border-studio-accent disabled:opacity-50"
            />
          </div>
        </div>
      </div>

      {/* Config Status */}
      <div className="p-4 rounded-lg border border-studio-border bg-studio-panel">
        <div className="flex items-center gap-2 mb-4">
          <Settings className="w-5 h-5 text-studio-accent" />
          <h3 className="font-medium">Configuration Status</h3>
        </div>
        <div className="space-y-2 text-sm">
          <div className="flex justify-between">
            <span className="text-studio-muted">Config File</span>
            <span className="font-mono text-xs">{config.config_path}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-studio-muted">Hot Reload Active</span>
            <span className={config.hot_reload_active ? 'text-green-400' : 'text-red-400'}>
              {config.hot_reload_active ? 'Yes' : 'No'}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
