'use client';

import React, { useState, useEffect } from 'react';
import {
  FileText,
  Save,
  RotateCcw,
  ChevronDown,
  ChevronUp,
  Play,
  Check,
  AlertCircle,
  ArrowLeft,
  Edit3,
  Eye
} from 'lucide-react';
import Link from 'next/link';

// Types
interface Prompt {
  id: string;
  name: string;
  description: string;
  category: string;
  variables: string[];
  system_prompt: string;
  user_template: string;
  is_custom: boolean;
  updated_at?: string;
}

interface TestResult {
  system_prompt: string;
  user_prompt: string;
  variables_used: Record<string, string>;
}

// API base URL - use environment variable or empty for same origin
const API_BASE = process.env.NEXT_PUBLIC_API_URL || '';

export default function PromptsPage() {
  const [prompts, setPrompts] = useState<Prompt[]>([]);
  const [selectedPrompt, setSelectedPrompt] = useState<Prompt | null>(null);
  const [editedPrompt, setEditedPrompt] = useState<Prompt | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [testVariables, setTestVariables] = useState<Record<string, string>>({});
  const [testResult, setTestResult] = useState<TestResult | null>(null);
  const [showPreview, setShowPreview] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [expandedCategories, setExpandedCategories] = useState<Set<string>>(new Set(['input', 'planning', 'synthesis', 'analysis', 'validation']));

  // Fetch prompts on mount
  useEffect(() => {
    fetchPrompts();
  }, []);

  const fetchPrompts = async () => {
    try {
      setLoading(true);
      const response = await fetch(`${API_BASE}/api/prompts`);
      if (!response.ok) throw new Error('Failed to fetch prompts');
      const data = await response.json();
      setPrompts(data.prompts);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch prompts');
    } finally {
      setLoading(false);
    }
  };

  const selectPrompt = (prompt: Prompt) => {
    setSelectedPrompt(prompt);
    setEditedPrompt({ ...prompt });
    setTestResult(null);
    setShowPreview(false);
    // Initialize test variables with empty strings
    const vars: Record<string, string> = {};
    prompt.variables.forEach(v => {
      vars[v] = getSampleValue(v);
    });
    setTestVariables(vars);
  };

  const getSampleValue = (variable: string): string => {
    const samples: Record<string, string> = {
      company_name: 'Apple Inc',
      context: '{"is_public": true, "industry": "Technology"}',
      tool_specs: 'fetch_sec_data, fetch_market_data, web_search',
      tool_reasoning: 'Public company with significant market presence',
      tool_results: '{"sec_data": {...}, "market_data": {...}}',
      company_data: '{"financials": {...}, "filings": [...]}',
      assessment: '{"risk_level": "LOW", "credit_score": 750}',
    };
    return samples[variable] || `Sample ${variable}`;
  };

  const handleSave = async () => {
    if (!editedPrompt) return;

    try {
      setSaving(true);
      setError(null);

      const response = await fetch(`${API_BASE}/api/prompts/${editedPrompt.id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          system_prompt: editedPrompt.system_prompt,
          user_template: editedPrompt.user_template,
        }),
      });

      if (!response.ok) throw new Error('Failed to save prompt');

      const data = await response.json();
      setSelectedPrompt(data.prompt);
      setEditedPrompt(data.prompt);
      setSuccess('Prompt saved successfully!');

      // Refresh prompts list
      fetchPrompts();

      setTimeout(() => setSuccess(null), 3000);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save prompt');
    } finally {
      setSaving(false);
    }
  };

  const handleReset = async () => {
    if (!selectedPrompt) return;

    try {
      setSaving(true);
      setError(null);

      const response = await fetch(`${API_BASE}/api/prompts/${selectedPrompt.id}/reset`, {
        method: 'POST',
      });

      if (!response.ok) throw new Error('Failed to reset prompt');

      const data = await response.json();
      setSelectedPrompt(data.prompt);
      setEditedPrompt(data.prompt);
      setSuccess('Prompt reset to default!');

      // Refresh prompts list
      fetchPrompts();

      setTimeout(() => setSuccess(null), 3000);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to reset prompt');
    } finally {
      setSaving(false);
    }
  };

  const handleTest = async () => {
    if (!editedPrompt) return;

    try {
      setError(null);

      const response = await fetch(`${API_BASE}/api/prompts/test`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt_id: editedPrompt.id,
          variables: testVariables,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to test prompt');
      }

      const data = await response.json();
      setTestResult(data);
      setShowPreview(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to test prompt');
    }
  };

  const toggleCategory = (category: string) => {
    const newExpanded = new Set(expandedCategories);
    if (newExpanded.has(category)) {
      newExpanded.delete(category);
    } else {
      newExpanded.add(category);
    }
    setExpandedCategories(newExpanded);
  };

  const hasChanges = () => {
    if (!selectedPrompt || !editedPrompt) return false;
    return (
      selectedPrompt.system_prompt !== editedPrompt.system_prompt ||
      selectedPrompt.user_template !== editedPrompt.user_template
    );
  };

  // Group prompts by category
  const promptsByCategory = prompts.reduce((acc, prompt) => {
    const category = prompt.category || 'other';
    if (!acc[category]) acc[category] = [];
    acc[category].push(prompt);
    return acc;
  }, {} as Record<string, Prompt[]>);

  const categoryLabels: Record<string, string> = {
    input: 'Input Processing',
    planning: 'Planning & Tool Selection',
    synthesis: 'Synthesis & Assessment',
    analysis: 'Analysis',
    validation: 'Validation',
    other: 'Other',
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-950 text-white flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-950 text-white">
      {/* Header */}
      <header className="bg-gray-900 border-b border-gray-800 px-6 py-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Link href="/" className="text-gray-400 hover:text-white transition-colors">
              <ArrowLeft className="w-5 h-5" />
            </Link>
            <div className="flex items-center gap-3">
              <FileText className="w-6 h-6 text-blue-500" />
              <h1 className="text-xl font-semibold">Prompt Management</h1>
            </div>
          </div>
          <span className="text-sm text-gray-500">{prompts.length} prompts available</span>
        </div>
      </header>

      {/* Alerts */}
      {error && (
        <div className="max-w-7xl mx-auto px-6 pt-4">
          <div className="bg-red-900/50 border border-red-700 text-red-200 px-4 py-3 rounded-lg flex items-center gap-2">
            <AlertCircle className="w-5 h-5" />
            {error}
          </div>
        </div>
      )}
      {success && (
        <div className="max-w-7xl mx-auto px-6 pt-4">
          <div className="bg-green-900/50 border border-green-700 text-green-200 px-4 py-3 rounded-lg flex items-center gap-2">
            <Check className="w-5 h-5" />
            {success}
          </div>
        </div>
      )}

      <div className="max-w-7xl mx-auto px-6 py-6">
        <div className="grid grid-cols-12 gap-6">
          {/* Sidebar - Prompt List */}
          <div className="col-span-4 bg-gray-900 rounded-lg border border-gray-800 overflow-hidden">
            <div className="p-4 border-b border-gray-800">
              <h2 className="font-medium">Available Prompts</h2>
            </div>
            <div className="max-h-[calc(100vh-16rem)] overflow-y-auto">
              {Object.entries(promptsByCategory).map(([category, categoryPrompts]) => (
                <div key={category} className="border-b border-gray-800 last:border-b-0">
                  <button
                    onClick={() => toggleCategory(category)}
                    className="w-full px-4 py-3 flex items-center justify-between hover:bg-gray-800/50 transition-colors"
                  >
                    <span className="text-sm font-medium text-gray-400">
                      {categoryLabels[category] || category}
                    </span>
                    {expandedCategories.has(category) ? (
                      <ChevronUp className="w-4 h-4 text-gray-500" />
                    ) : (
                      <ChevronDown className="w-4 h-4 text-gray-500" />
                    )}
                  </button>
                  {expandedCategories.has(category) && (
                    <div className="pb-2">
                      {categoryPrompts.map(prompt => (
                        <button
                          key={prompt.id}
                          onClick={() => selectPrompt(prompt)}
                          className={`w-full px-4 py-2 text-left transition-colors ${
                            selectedPrompt?.id === prompt.id
                              ? 'bg-blue-600/20 border-l-2 border-blue-500'
                              : 'hover:bg-gray-800/50 border-l-2 border-transparent'
                          }`}
                        >
                          <div className="flex items-center justify-between">
                            <span className="text-sm">{prompt.name}</span>
                            {prompt.is_custom && (
                              <span className="text-xs bg-yellow-600/30 text-yellow-400 px-2 py-0.5 rounded">
                                Modified
                              </span>
                            )}
                          </div>
                          <p className="text-xs text-gray-500 mt-1 truncate">
                            {prompt.description}
                          </p>
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* Main Content - Prompt Editor */}
          <div className="col-span-8">
            {selectedPrompt && editedPrompt ? (
              <div className="bg-gray-900 rounded-lg border border-gray-800">
                {/* Prompt Header */}
                <div className="p-4 border-b border-gray-800 flex items-center justify-between">
                  <div>
                    <h2 className="font-medium text-lg">{editedPrompt.name}</h2>
                    <p className="text-sm text-gray-500">{editedPrompt.description}</p>
                  </div>
                  <div className="flex items-center gap-2">
                    {editedPrompt.is_custom && (
                      <button
                        onClick={handleReset}
                        disabled={saving}
                        className="flex items-center gap-2 px-3 py-1.5 text-sm bg-gray-700 hover:bg-gray-600 rounded transition-colors disabled:opacity-50"
                      >
                        <RotateCcw className="w-4 h-4" />
                        Reset
                      </button>
                    )}
                    <button
                      onClick={handleSave}
                      disabled={saving || !hasChanges()}
                      className="flex items-center gap-2 px-3 py-1.5 text-sm bg-blue-600 hover:bg-blue-500 rounded transition-colors disabled:opacity-50"
                    >
                      <Save className="w-4 h-4" />
                      {saving ? 'Saving...' : 'Save'}
                    </button>
                  </div>
                </div>

                {/* Variables */}
                <div className="p-4 border-b border-gray-800 bg-gray-800/30">
                  <h3 className="text-sm font-medium mb-2 text-gray-400">Variables</h3>
                  <div className="flex flex-wrap gap-2">
                    {editedPrompt.variables.map(variable => (
                      <span
                        key={variable}
                        className="px-2 py-1 bg-gray-700 rounded text-sm font-mono"
                      >
                        {'{' + variable + '}'}
                      </span>
                    ))}
                  </div>
                </div>

                {/* System Prompt */}
                <div className="p-4 border-b border-gray-800">
                  <div className="flex items-center justify-between mb-2">
                    <h3 className="text-sm font-medium text-gray-400">System Prompt</h3>
                    <Edit3 className="w-4 h-4 text-gray-500" />
                  </div>
                  <textarea
                    value={editedPrompt.system_prompt}
                    onChange={(e) => setEditedPrompt({ ...editedPrompt, system_prompt: e.target.value })}
                    className="w-full h-48 bg-gray-800 border border-gray-700 rounded-lg p-3 text-sm font-mono focus:outline-none focus:border-blue-500 resize-y"
                    placeholder="Enter system prompt..."
                  />
                </div>

                {/* Advanced Section Toggle */}
                <div className="p-4 border-b border-gray-800">
                  <button
                    onClick={() => setShowAdvanced(!showAdvanced)}
                    className="flex items-center gap-2 text-sm text-gray-500 hover:text-gray-300 transition-colors"
                  >
                    {showAdvanced ? (
                      <ChevronUp className="w-4 h-4" />
                    ) : (
                      <ChevronDown className="w-4 h-4" />
                    )}
                    Advanced Options
                  </button>
                </div>

                {/* Advanced Section - Hidden by default */}
                {showAdvanced && (
                  <>
                    {/* User Template */}
                    <div className="p-4 border-b border-gray-800 bg-yellow-900/10">
                      <div className="flex items-center justify-between mb-2">
                        <h3 className="text-sm font-medium text-yellow-400">User Template</h3>
                        <AlertCircle className="w-4 h-4 text-yellow-500" />
                      </div>
                      <p className="text-xs text-yellow-600 mb-2">
                        ⚠️ Contains {'{variable}'} placeholders. Modifying incorrectly may break the workflow.
                      </p>
                      <textarea
                        value={editedPrompt.user_template}
                        onChange={(e) => setEditedPrompt({ ...editedPrompt, user_template: e.target.value })}
                        className="w-full h-48 bg-gray-800 border border-yellow-700 rounded-lg p-3 text-sm font-mono focus:outline-none focus:border-yellow-500 resize-y"
                        placeholder="Enter user template..."
                      />
                    </div>

                    {/* Test Section */}
                    <div className="p-4 border-b border-gray-800">
                      <div className="flex items-center justify-between mb-4">
                        <h3 className="text-sm font-medium text-gray-400">Test Prompt</h3>
                        <div className="flex items-center gap-2">
                          <button
                            onClick={() => setShowPreview(!showPreview)}
                            className="flex items-center gap-2 px-3 py-1.5 text-sm bg-gray-700 hover:bg-gray-600 rounded transition-colors"
                          >
                            <Eye className="w-4 h-4" />
                            {showPreview ? 'Hide Preview' : 'Show Preview'}
                          </button>
                          <button
                            onClick={handleTest}
                            className="flex items-center gap-2 px-3 py-1.5 text-sm bg-green-600 hover:bg-green-500 rounded transition-colors"
                          >
                            <Play className="w-4 h-4" />
                            Test
                          </button>
                        </div>
                      </div>

                      {/* Test Variables */}
                      <div className="grid grid-cols-2 gap-3 mb-4">
                        {editedPrompt.variables.map(variable => (
                          <div key={variable}>
                            <label className="text-xs text-gray-500 mb-1 block">{variable}</label>
                            <input
                              type="text"
                              value={testVariables[variable] || ''}
                              onChange={(e) => setTestVariables({ ...testVariables, [variable]: e.target.value })}
                              className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm focus:outline-none focus:border-blue-500"
                              placeholder={`Enter ${variable}...`}
                            />
                          </div>
                        ))}
                      </div>

                      {/* Preview */}
                      {showPreview && testResult && (
                        <div className="space-y-3">
                          <div className="bg-gray-800 rounded-lg p-3">
                            <h4 className="text-xs font-medium text-blue-400 mb-2">System Prompt Preview</h4>
                            <pre className="text-xs text-gray-300 whitespace-pre-wrap font-mono">
                              {testResult.system_prompt}
                            </pre>
                          </div>
                          <div className="bg-gray-800 rounded-lg p-3">
                            <h4 className="text-xs font-medium text-green-400 mb-2">User Prompt Preview</h4>
                            <pre className="text-xs text-gray-300 whitespace-pre-wrap font-mono">
                              {testResult.user_prompt}
                            </pre>
                          </div>
                        </div>
                      )}
                    </div>
                  </>
                )}
              </div>
            ) : (
              <div className="bg-gray-900 rounded-lg border border-gray-800 p-12 text-center">
                <FileText className="w-12 h-12 text-gray-600 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-400 mb-2">No Prompt Selected</h3>
                <p className="text-sm text-gray-500">
                  Select a prompt from the sidebar to view and edit
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
