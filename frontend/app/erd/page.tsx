'use client';

import React, { useState, useEffect } from 'react';
import {
  Database,
  ArrowLeft,
  Table,
  GitBranch,
  Layers,
  ChevronDown,
  ChevronRight,
  Copy,
  Check,
} from 'lucide-react';
import Link from 'next/link';

// Table definitions with their columns
const tables = {
  workflow: [
    {
      name: 'wf_runs',
      description: 'Run summaries with performance metrics',
      columns: [
        { name: 'id', type: 'bigint', key: 'PK' },
        { name: 'run_id', type: 'varchar', key: 'UK' },
        { name: 'company_name', type: 'varchar', key: 'J' },
        { name: 'node', type: 'varchar', key: 'J' },
        { name: 'agent_name', type: 'varchar', key: 'J' },
        { name: 'master_agent', type: 'varchar', key: 'J' },
        { name: 'status', type: 'varchar' },
        { name: 'risk_level', type: 'varchar' },
        { name: 'credit_score', type: 'int' },
        { name: 'confidence', type: 'decimal' },
        { name: 'evaluation_score', type: 'decimal' },
        { name: 'total_time_ms', type: 'decimal' },
        { name: 'total_steps', type: 'int' },
        { name: 'total_llm_calls', type: 'int' },
        { name: 'tool_overall_score', type: 'decimal' },
        { name: 'agent_overall_score', type: 'decimal' },
        { name: 'workflow_overall_score', type: 'decimal' },
        { name: 'timestamp', type: 'timestamptz' },
      ],
    },
    {
      name: 'wf_llm_calls',
      description: 'LLM API call logs',
      columns: [
        { name: 'id', type: 'bigint', key: 'PK' },
        { name: 'run_id', type: 'varchar', key: 'FK' },
        { name: 'company_name', type: 'varchar', key: 'J' },
        { name: 'node', type: 'varchar', key: 'J' },
        { name: 'agent_name', type: 'varchar', key: 'J' },
        { name: 'step_number', type: 'int', key: 'J' },
        { name: 'call_type', type: 'varchar' },
        { name: 'model', type: 'varchar' },
        { name: 'prompt_tokens', type: 'int' },
        { name: 'completion_tokens', type: 'int' },
        { name: 'total_tokens', type: 'int' },
        { name: 'total_cost', type: 'decimal' },
        { name: 'execution_time_ms', type: 'decimal' },
        { name: 'status', type: 'varchar' },
        { name: 'timestamp', type: 'timestamptz' },
      ],
    },
    {
      name: 'wf_tool_calls',
      description: 'Tool execution logs',
      columns: [
        { name: 'id', type: 'bigint', key: 'PK' },
        { name: 'run_id', type: 'varchar', key: 'FK' },
        { name: 'company_name', type: 'varchar', key: 'J' },
        { name: 'node', type: 'varchar', key: 'J' },
        { name: 'agent_name', type: 'varchar', key: 'J' },
        { name: 'step_number', type: 'int', key: 'J' },
        { name: 'tool_name', type: 'varchar' },
        { name: 'tool_input', type: 'jsonb' },
        { name: 'tool_output', type: 'jsonb' },
        { name: 'execution_time_ms', type: 'decimal' },
        { name: 'status', type: 'varchar' },
        { name: 'call_depth', type: 'int' },
        { name: 'timestamp', type: 'timestamptz' },
      ],
    },
    {
      name: 'wf_assessments',
      description: 'Credit assessment outputs',
      columns: [
        { name: 'id', type: 'bigint', key: 'PK' },
        { name: 'run_id', type: 'varchar', key: 'FK' },
        { name: 'company_name', type: 'varchar', key: 'J' },
        { name: 'risk_level', type: 'varchar' },
        { name: 'credit_score', type: 'int' },
        { name: 'confidence', type: 'decimal' },
        { name: 'reasoning', type: 'text' },
        { name: 'recommendations', type: 'text' },
        { name: 'duration_ms', type: 'decimal' },
        { name: 'timestamp', type: 'timestamptz' },
      ],
    },
    {
      name: 'wf_plans',
      description: 'Task plans created',
      columns: [
        { name: 'id', type: 'bigint', key: 'PK' },
        { name: 'run_id', type: 'varchar', key: 'FK' },
        { name: 'company_name', type: 'varchar', key: 'J' },
        { name: 'num_tasks', type: 'int' },
        { name: 'plan_summary', type: 'text' },
        { name: 'full_plan', type: 'jsonb' },
        { name: 'timestamp', type: 'timestamptz' },
      ],
    },
    {
      name: 'wf_data_sources',
      description: 'API data fetch results',
      columns: [
        { name: 'id', type: 'bigint', key: 'PK' },
        { name: 'run_id', type: 'varchar', key: 'FK' },
        { name: 'company_name', type: 'varchar', key: 'J' },
        { name: 'source_name', type: 'varchar' },
        { name: 'records_found', type: 'int' },
        { name: 'execution_time_ms', type: 'decimal' },
        { name: 'status', type: 'varchar' },
        { name: 'timestamp', type: 'timestamptz' },
      ],
    },
    {
      name: 'wf_state_dumps',
      description: 'Workflow state snapshots',
      columns: [
        { name: 'id', type: 'bigint', key: 'PK' },
        { name: 'run_id', type: 'varchar', key: 'FK' },
        { name: 'company_name', type: 'varchar', key: 'J' },
        { name: 'company_info_json', type: 'jsonb' },
        { name: 'plan_json', type: 'jsonb' },
        { name: 'assessment_json', type: 'jsonb' },
        { name: 'evaluation_json', type: 'jsonb' },
        { name: 'total_state_size_bytes', type: 'int' },
        { name: 'timestamp', type: 'timestamptz' },
      ],
    },
  ],
  langgraph: [
    {
      name: 'lg_events',
      description: 'LangGraph execution events',
      columns: [
        { name: 'id', type: 'bigint', key: 'PK' },
        { name: 'run_id', type: 'varchar', key: 'FK' },
        { name: 'company_name', type: 'varchar', key: 'J' },
        { name: 'node', type: 'varchar', key: 'J' },
        { name: 'event_type', type: 'varchar' },
        { name: 'event_name', type: 'varchar' },
        { name: 'tokens', type: 'int' },
        { name: 'duration_ms', type: 'decimal' },
        { name: 'status', type: 'varchar' },
        { name: 'timestamp', type: 'timestamptz' },
      ],
    },
  ],
  evaluation: [
    {
      name: 'eval_results',
      description: 'Main evaluation scores',
      columns: [
        { name: 'id', type: 'bigint', key: 'PK' },
        { name: 'run_id', type: 'varchar', key: 'FK' },
        { name: 'company_name', type: 'varchar', key: 'J' },
        { name: 'tool_selection_score', type: 'decimal' },
        { name: 'data_quality_score', type: 'decimal' },
        { name: 'synthesis_score', type: 'decimal' },
        { name: 'overall_score', type: 'decimal' },
        { name: 'eval_status', type: 'varchar' },
        { name: 'timestamp', type: 'timestamptz' },
      ],
    },
    {
      name: 'eval_tool_selection',
      description: 'Tool selection analysis',
      columns: [
        { name: 'id', type: 'bigint', key: 'PK' },
        { name: 'run_id', type: 'varchar', key: 'FK' },
        { name: 'selected_tools', type: 'jsonb' },
        { name: 'expected_tools', type: 'jsonb' },
        { name: 'correct_tools', type: 'jsonb' },
        { name: 'missing_tools', type: 'jsonb' },
        { name: 'precision', type: 'decimal' },
        { name: 'recall', type: 'decimal' },
        { name: 'f1_score', type: 'decimal' },
        { name: 'timestamp', type: 'timestamptz' },
      ],
    },
    {
      name: 'eval_agent_metrics',
      description: 'Agent efficiency metrics',
      columns: [
        { name: 'id', type: 'bigint', key: 'PK' },
        { name: 'run_id', type: 'varchar', key: 'FK' },
        { name: 'intent_correctness', type: 'decimal' },
        { name: 'plan_quality', type: 'decimal' },
        { name: 'tool_choice_correctness', type: 'decimal' },
        { name: 'tool_completeness', type: 'decimal' },
        { name: 'trajectory_match', type: 'decimal' },
        { name: 'final_answer_quality', type: 'decimal' },
        { name: 'overall_score', type: 'decimal' },
        { name: 'timestamp', type: 'timestamptz' },
      ],
    },
    {
      name: 'eval_llm_judge',
      description: 'LLM-as-Judge results',
      columns: [
        { name: 'id', type: 'bigint', key: 'PK' },
        { name: 'run_id', type: 'varchar', key: 'FK' },
        { name: 'accuracy_score', type: 'decimal' },
        { name: 'completeness_score', type: 'decimal' },
        { name: 'consistency_score', type: 'decimal' },
        { name: 'actionability_score', type: 'decimal' },
        { name: 'data_utilization_score', type: 'decimal' },
        { name: 'overall_score', type: 'decimal' },
        { name: 'timestamp', type: 'timestamptz' },
      ],
    },
    {
      name: 'eval_consistency',
      description: 'Model consistency scores',
      columns: [
        { name: 'id', type: 'bigint', key: 'PK' },
        { name: 'run_id', type: 'varchar', key: 'FK' },
        { name: 'model_name', type: 'varchar' },
        { name: 'num_runs', type: 'int' },
        { name: 'risk_level_consistency', type: 'decimal' },
        { name: 'score_consistency', type: 'decimal' },
        { name: 'score_std', type: 'decimal' },
        { name: 'overall_consistency', type: 'decimal' },
        { name: 'timestamp', type: 'timestamptz' },
      ],
    },
    {
      name: 'eval_cross_model',
      description: 'Cross-model comparison',
      columns: [
        { name: 'id', type: 'bigint', key: 'PK' },
        { name: 'run_id', type: 'varchar', key: 'FK' },
        { name: 'models_compared', type: 'jsonb' },
        { name: 'num_models', type: 'int' },
        { name: 'risk_level_agreement', type: 'decimal' },
        { name: 'credit_score_mean', type: 'decimal' },
        { name: 'credit_score_std', type: 'decimal' },
        { name: 'cross_model_agreement', type: 'decimal' },
        { name: 'timestamp', type: 'timestamptz' },
      ],
    },
    {
      name: 'eval_coalition',
      description: 'Coalition voting results',
      columns: [
        { name: 'id', type: 'bigint', key: 'PK' },
        { name: 'run_id', type: 'varchar', key: 'FK' },
        { name: 'is_correct', type: 'boolean' },
        { name: 'correctness_score', type: 'decimal' },
        { name: 'efficiency_score', type: 'decimal' },
        { name: 'quality_score', type: 'decimal' },
        { name: 'agreement_score', type: 'decimal' },
        { name: 'num_evaluators', type: 'int' },
        { name: 'votes_json', type: 'jsonb' },
        { name: 'timestamp', type: 'timestamptz' },
      ],
    },
    {
      name: 'eval_node_scoring',
      description: 'LLM judge quality scores for each node',
      columns: [
        { name: 'id', type: 'bigint', key: 'PK' },
        { name: 'run_id', type: 'varchar', key: 'FK' },
        { name: 'company_name', type: 'varchar', key: 'J' },
        { name: 'node', type: 'varchar', key: 'J' },
        { name: 'node_type', type: 'varchar' },
        { name: 'agent_name', type: 'varchar', key: 'J' },
        { name: 'master_agent', type: 'varchar' },
        { name: 'step_number', type: 'int' },
        { name: 'task_description', type: 'text' },
        { name: 'task_completed', type: 'boolean' },
        { name: 'quality_score', type: 'decimal' },
        { name: 'quality_reasoning', type: 'text' },
        { name: 'input_summary', type: 'text' },
        { name: 'output_summary', type: 'text' },
        { name: 'judge_model', type: 'varchar' },
        { name: 'timestamp', type: 'timestamptz' },
      ],
    },
    {
      name: 'eval_log_tests',
      description: 'Logging verification',
      columns: [
        { name: 'id', type: 'bigint', key: 'PK' },
        { name: 'run_id', type: 'varchar', key: 'FK' },
        { name: 'runs', type: 'int' },
        { name: 'langgraph_events', type: 'int' },
        { name: 'llm_calls', type: 'int' },
        { name: 'tool_calls', type: 'int' },
        { name: 'total_sheets_logged', type: 'int' },
        { name: 'verification_status', type: 'varchar' },
        { name: 'timestamp', type: 'timestamptz' },
      ],
    },
  ],
  metadata: [
    {
      name: 'meta_prompts',
      description: 'Prompts used during runs',
      columns: [
        { name: 'id', type: 'bigint', key: 'PK' },
        { name: 'run_id', type: 'varchar', key: 'FK' },
        { name: 'company_name', type: 'varchar', key: 'J' },
        { name: 'prompt_id', type: 'varchar' },
        { name: 'prompt_name', type: 'varchar' },
        { name: 'category', type: 'varchar' },
        { name: 'system_prompt', type: 'text' },
        { name: 'user_prompt', type: 'text' },
        { name: 'variables_json', type: 'jsonb' },
        { name: 'timestamp', type: 'timestamptz' },
      ],
    },
  ],
};

// Workflow nodes for the flow diagram
const workflowNodes = [
  { id: 'parse_input', agent: 'llm_parser', description: 'Parse company input' },
  { id: 'validate_company', agent: 'supervisor', description: 'Validate company' },
  { id: 'create_plan', agent: 'tool_supervisor', description: 'LLM tool selection' },
  { id: 'fetch_api_data', agent: 'api_agent', description: 'Fetch API data' },
  { id: 'data_quality_check', agent: 'router', description: 'Route based on data quality', isConditional: true },
  { id: 'search_web', agent: 'search_agent', description: 'Normal web search (>=2 sources)' },
  { id: 'search_web_enhanced', agent: 'search_agent', description: 'Enhanced search (<2 sources)' },
  { id: 'synthesize', agent: 'llm_analyst', description: 'Credit synthesis' },
  { id: 'save_to_database', agent: 'db_writer', description: 'Save to MongoDB' },
  { id: 'evaluate_assessment', agent: 'workflow_evaluator', description: 'All evaluations' },
];

export default function ERDPage() {
  const [activeTab, setActiveTab] = useState<'tables' | 'workflow'>('tables');
  const [expandedTables, setExpandedTables] = useState<Set<string>>(new Set(['wf_runs']));
  const [copied, setCopied] = useState(false);

  const toggleTable = (tableName: string) => {
    setExpandedTables((prev) => {
      const next = new Set(prev);
      if (next.has(tableName)) {
        next.delete(tableName);
      } else {
        next.add(tableName);
      }
      return next;
    });
  };

  const copyMermaid = () => {
    const mermaidCode = `erDiagram
    wf_runs ||--o{ wf_llm_calls : "run_id"
    wf_runs ||--o{ wf_tool_calls : "run_id"
    wf_runs ||--o{ wf_assessments : "run_id"
    wf_runs ||--o{ wf_plans : "run_id"
    wf_runs ||--o{ wf_data_sources : "run_id"
    wf_runs ||--o{ wf_state_dumps : "run_id"
    wf_runs ||--o{ lg_events : "run_id"
    wf_runs ||--o{ eval_results : "run_id"
    wf_runs ||--o{ eval_tool_selection : "run_id"
    wf_runs ||--o{ eval_consistency : "run_id"
    wf_runs ||--o{ eval_coalition : "run_id"
    wf_runs ||--o{ eval_agent_metrics : "run_id"
    wf_runs ||--o{ eval_llm_judge : "run_id"
    wf_runs ||--o{ eval_cross_model : "run_id"
    wf_runs ||--o{ eval_log_tests : "run_id"
    wf_runs ||--o{ meta_prompts : "run_id"`;
    navigator.clipboard.writeText(mermaidCode);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const categoryColors: Record<string, { border: string; bg: string; text: string }> = {
    workflow: { border: 'border-blue-500', bg: 'bg-blue-500/10', text: 'text-blue-400' },
    langgraph: { border: 'border-purple-500', bg: 'bg-purple-500/10', text: 'text-purple-400' },
    evaluation: { border: 'border-green-500', bg: 'bg-green-500/10', text: 'text-green-400' },
    metadata: { border: 'border-yellow-500', bg: 'bg-yellow-500/10', text: 'text-yellow-400' },
  };

  const getKeyBadge = (key?: string) => {
    if (!key) return null;
    const colors: Record<string, string> = {
      PK: 'bg-yellow-600 text-yellow-100',
      FK: 'bg-blue-600 text-blue-100',
      UK: 'bg-purple-600 text-purple-100',
      J: 'bg-gray-600 text-gray-100',
    };
    return (
      <span className={`text-[10px] px-1.5 py-0.5 rounded font-medium ${colors[key] || 'bg-gray-600'}`}>
        {key}
      </span>
    );
  };

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
              <Database className="w-5 h-5 text-studio-accent" />
              <h1 className="text-lg font-semibold">Entity Relationship Diagram</h1>
            </div>
          </div>
          <button
            onClick={copyMermaid}
            className="flex items-center gap-2 px-3 py-1.5 text-sm bg-studio-panel hover:bg-studio-border border border-studio-border rounded transition-colors"
          >
            {copied ? <Check className="w-4 h-4 text-green-400" /> : <Copy className="w-4 h-4" />}
            {copied ? 'Copied!' : 'Copy Mermaid'}
          </button>
        </div>
      </header>

      {/* Tab Navigation */}
      <div className="border-b border-studio-border bg-studio-panel">
        <div className="max-w-7xl mx-auto px-4">
          <nav className="flex gap-1">
            <button
              onClick={() => setActiveTab('tables')}
              className={`flex items-center gap-2 px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
                activeTab === 'tables'
                  ? 'border-studio-accent text-studio-accent'
                  : 'border-transparent text-studio-muted hover:text-studio-text'
              }`}
            >
              <Table className="w-4 h-4" />
              Database Tables (17)
            </button>
            <button
              onClick={() => setActiveTab('workflow')}
              className={`flex items-center gap-2 px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
                activeTab === 'workflow'
                  ? 'border-studio-accent text-studio-accent'
                  : 'border-transparent text-studio-muted hover:text-studio-text'
              }`}
            >
              <GitBranch className="w-4 h-4" />
              Workflow Graph
            </button>
          </nav>
        </div>
      </div>

      {/* Main Content */}
      <main className="flex-1 max-w-7xl mx-auto w-full px-4 py-6">
        {activeTab === 'tables' && (
          <div className="space-y-6">
            {/* Legend */}
            <div className="p-4 rounded-lg border border-studio-border bg-studio-panel">
              <h3 className="text-sm font-medium text-studio-muted mb-3">Legend</h3>
              <div className="flex flex-wrap gap-4 text-sm">
                <div className="flex items-center gap-2">
                  <span className="bg-yellow-600 text-yellow-100 text-[10px] px-1.5 py-0.5 rounded font-medium">PK</span>
                  <span className="text-studio-muted">Primary Key</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="bg-blue-600 text-blue-100 text-[10px] px-1.5 py-0.5 rounded font-medium">FK</span>
                  <span className="text-studio-muted">Foreign Key (links to wf_runs.run_id)</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="bg-purple-600 text-purple-100 text-[10px] px-1.5 py-0.5 rounded font-medium">UK</span>
                  <span className="text-studio-muted">Unique Key</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="bg-gray-600 text-gray-100 text-[10px] px-1.5 py-0.5 rounded font-medium">J</span>
                  <span className="text-studio-muted">Join Column</span>
                </div>
              </div>
            </div>

            {/* Category Sections */}
            {Object.entries(tables).map(([category, categoryTables]) => {
              const colors = categoryColors[category];
              const categoryNames: Record<string, string> = {
                workflow: 'Workflow Tables (wf_*)',
                langgraph: 'LangGraph Tables (lg_*)',
                evaluation: 'Evaluation Tables (eval_*)',
                metadata: 'Metadata Tables (meta_*)',
              };

              return (
                <div key={category} className="space-y-3">
                  <div className={`flex items-center gap-2 ${colors.text}`}>
                    <Layers className="w-5 h-5" />
                    <h2 className="font-semibold">{categoryNames[category]}</h2>
                    <span className="text-xs bg-studio-border px-2 py-0.5 rounded">
                      {categoryTables.length} tables
                    </span>
                  </div>

                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
                    {categoryTables.map((table) => (
                      <div
                        key={table.name}
                        className={`rounded-lg border ${colors.border} ${colors.bg} overflow-hidden`}
                      >
                        <button
                          onClick={() => toggleTable(table.name)}
                          className="w-full p-3 flex items-center justify-between hover:bg-black/20 transition-colors"
                        >
                          <div className="flex items-center gap-2">
                            <Table className={`w-4 h-4 ${colors.text}`} />
                            <span className="font-mono font-medium">{table.name}</span>
                          </div>
                          <div className="flex items-center gap-2">
                            <span className="text-xs text-studio-muted">{table.columns.length} cols</span>
                            {expandedTables.has(table.name) ? (
                              <ChevronDown className="w-4 h-4 text-studio-muted" />
                            ) : (
                              <ChevronRight className="w-4 h-4 text-studio-muted" />
                            )}
                          </div>
                        </button>

                        {expandedTables.has(table.name) && (
                          <div className="border-t border-studio-border bg-black/30">
                            <div className="px-3 py-2 text-xs text-studio-muted border-b border-studio-border">
                              {table.description}
                            </div>
                            <div className="divide-y divide-studio-border/50">
                              {table.columns.map((col) => (
                                <div key={col.name} className="px-3 py-1.5 flex items-center justify-between text-sm">
                                  <div className="flex items-center gap-2">
                                    {getKeyBadge(col.key)}
                                    <span className="font-mono">{col.name}</span>
                                  </div>
                                  <span className="text-studio-muted text-xs">{col.type}</span>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              );
            })}

            {/* Storage Summary */}
            <div className="p-4 rounded-lg border border-studio-border bg-studio-panel">
              <h3 className="text-sm font-medium text-studio-muted mb-3">Storage Summary</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                <div>
                  <span className="text-studio-muted">Google Sheets:</span>
                  <span className="ml-2">17 sheets</span>
                </div>
                <div>
                  <span className="text-studio-muted">PostgreSQL:</span>
                  <span className="ml-2">17 tables + partitions</span>
                </div>
                <div>
                  <span className="text-studio-muted">MongoDB:</span>
                  <span className="ml-2">17 collections</span>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'workflow' && (
          <div className="space-y-6">
            {/* Workflow Flow */}
            <div className="p-6 rounded-lg border border-studio-border bg-studio-panel">
              <h3 className="text-sm font-medium text-studio-muted mb-6">LangGraph Workflow with Conditional Routing</h3>

              <div className="flex flex-col items-center gap-2">
                {workflowNodes.map((node, idx) => (
                  <React.Fragment key={node.id}>
                    {/* Node */}
                    <div
                      className={`w-full max-w-md p-4 rounded-lg border ${
                        node.isConditional
                          ? 'border-yellow-500 bg-yellow-500/10'
                          : 'border-studio-accent bg-studio-accent/10'
                      }`}
                    >
                      <div className="flex items-center justify-between mb-1">
                        <span className="font-mono font-medium">{node.id}</span>
                        <span className="text-xs px-2 py-0.5 rounded bg-studio-border">
                          {node.agent}
                        </span>
                      </div>
                      <p className="text-sm text-studio-muted">{node.description}</p>
                    </div>

                    {/* Connector */}
                    {idx < workflowNodes.length - 1 && (
                      <div className="flex flex-col items-center">
                        {node.id === 'data_quality_check' ? (
                          <div className="flex gap-8 my-2">
                            <div className="flex flex-col items-center">
                              <span className="text-xs text-green-400 mb-1">&gt;=2 sources</span>
                              <div className="w-0.5 h-4 bg-green-500" />
                            </div>
                            <div className="flex flex-col items-center">
                              <span className="text-xs text-orange-400 mb-1">&lt;2 sources</span>
                              <div className="w-0.5 h-4 bg-orange-500" />
                            </div>
                          </div>
                        ) : node.id === 'search_web' || node.id === 'search_web_enhanced' ? (
                          <div className="w-0.5 h-6 bg-studio-border" />
                        ) : (
                          <div className="w-0.5 h-6 bg-studio-border" />
                        )}
                      </div>
                    )}
                  </React.Fragment>
                ))}
              </div>
            </div>

            {/* Agent Names Table */}
            <div className="p-4 rounded-lg border border-studio-border bg-studio-panel">
              <h3 className="text-sm font-medium text-studio-muted mb-3">Canonical Agent Names</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-studio-border">
                      <th className="text-left py-2 px-3 text-studio-muted font-medium">Node</th>
                      <th className="text-left py-2 px-3 text-studio-muted font-medium">agent_name</th>
                      <th className="text-left py-2 px-3 text-studio-muted font-medium">Description</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-studio-border/50">
                    {workflowNodes.filter(n => !n.isConditional).map((node) => (
                      <tr key={node.id}>
                        <td className="py-2 px-3 font-mono">{node.id}</td>
                        <td className="py-2 px-3">
                          <span className="px-2 py-0.5 rounded bg-studio-accent/20 text-studio-accent font-mono">
                            {node.agent}
                          </span>
                        </td>
                        <td className="py-2 px-3 text-studio-muted">{node.description}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Conditional Edge Info */}
            <div className="p-4 rounded-lg border border-yellow-800/50 bg-yellow-900/20">
              <h3 className="text-sm font-medium text-yellow-400 mb-2">Conditional Edge: Data Quality Routing</h3>
              <p className="text-sm text-yellow-400/80 mb-3">
                After <code className="bg-black/30 px-1 rounded">fetch_api_data</code>, the workflow checks API data quality:
              </p>
              <ul className="text-sm text-yellow-400/80 space-y-1 ml-4 list-disc">
                <li>Counts successful API sources (SEC Edgar, Finnhub, CourtListener)</li>
                <li><strong>&gt;=2 sources with data</strong> → Normal <code className="bg-black/30 px-1 rounded">search_web</code></li>
                <li><strong>&lt;2 sources with data</strong> → Enhanced <code className="bg-black/30 px-1 rounded">search_web_enhanced</code> with additional queries</li>
              </ul>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
