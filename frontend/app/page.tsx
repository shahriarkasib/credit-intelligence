'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
import {
  Play,
  Square,
  RefreshCw,
  CheckCircle,
  XCircle,
  Clock,
  Activity,
  ChevronRight,
  ChevronDown,
  Loader2,
  Building2,
  AlertTriangle,
  TrendingUp,
  FileText,
  X,
  Settings,
  List,
  BarChart3,
  Eye,
  Layers,
  Plus,
  Trash2,
  Download,
  ExternalLink
} from 'lucide-react'
import Link from 'next/link'

// Types
interface WorkflowStep {
  step_id: string
  name: string
  status: 'pending' | 'running' | 'completed' | 'failed'
  started_at?: string
  completed_at?: string
  duration_ms?: number
  output_summary?: string
  error?: string
  input_data?: any
  output_data?: any
}

// Trace event from backend
interface TraceEvent {
  _id: string
  run_id: string
  company_name: string
  event_type: string
  event_name: string
  status: string
  duration_ms?: number
  model?: string
  tokens?: number
  input_preview?: string
  output_preview?: string
  error?: string
  timestamp: string
}

// Backend evaluation result
interface BackendEvaluation {
  _id: string
  run_id: string
  company_name: string
  tool_selection_score?: number
  data_quality_score?: number
  synthesis_score?: number
  overall_score?: number
  tool_selection?: any
  data_quality?: any
  synthesis?: any
  evaluated_at?: string
}

interface WorkflowStatus {
  run_id: string
  company_name: string
  status: 'pending' | 'running' | 'completed' | 'failed'
  current_step?: string
  steps: WorkflowStep[]
  result?: {
    risk_level: string
    credit_score: number
    confidence: number
    reasoning: string
    recommendations: string[]
    evaluation_score: number
  }
  started_at: string
  completed_at?: string
}

interface LogEntry {
  timestamp: string
  type: string
  message: string
  data?: any
}

interface RunHistory {
  run_id: string
  company_name: string
  status: string
  risk_level?: string
  credit_score?: number
  evaluation_score?: number
  started_at: string
  completed_at?: string
  duration_ms?: number
}

interface EvalSummary {
  tool_selection_score: number
  data_quality_score: number
  synthesis_score: number
  overall_score: number
  tool_selection?: any
  data_quality?: any
  synthesis?: any
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || ''

// Helper to construct WebSocket URL (handles both relative and absolute API URLs)
const getWebSocketUrl = (path: string): string => {
  if (typeof window === 'undefined') return ''

  if (API_URL) {
    // Absolute URL provided - convert http(s) to ws(s)
    return API_URL.replace(/^http/, 'ws') + path
  } else {
    // Same origin - use current host
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    return `${protocol}//${window.location.host}${path}`
  }
}

// Tab types
type TabType = 'single' | 'batch' | 'history'
type DetailTab = 'result' | 'evaluation' | 'traces' | 'pipeline'

export default function CreditIntelligenceStudio() {
  // Main state
  const [activeTab, setActiveTab] = useState<TabType>('single')
  const [detailTab, setDetailTab] = useState<DetailTab>('result')

  // Single analysis state
  const [companyName, setCompanyName] = useState('')
  const [isRunning, setIsRunning] = useState(false)
  const [workflow, setWorkflow] = useState<WorkflowStatus | null>(null)
  const [logs, setLogs] = useState<LogEntry[]>([])
  const [error, setError] = useState<string | null>(null)
  const [expandedSteps, setExpandedSteps] = useState<Set<string>>(new Set())

  // Batch analysis state
  const [batchCompanies, setBatchCompanies] = useState<string[]>([''])
  const [batchResults, setBatchResults] = useState<WorkflowStatus[]>([])
  const [batchRunning, setBatchRunning] = useState(false)
  const [batchProgress, setBatchProgress] = useState({ current: 0, total: 0 })

  // History state
  const [runHistory, setRunHistory] = useState<RunHistory[]>([])
  const [selectedHistoryRun, setSelectedHistoryRun] = useState<WorkflowStatus | null>(null)

  // Evaluation summary
  const [evalSummary, setEvalSummary] = useState<EvalSummary | null>(null)

  // Backend traces and evaluations
  const [backendTraces, setBackendTraces] = useState<TraceEvent[]>([])
  const [backendEvaluation, setBackendEvaluation] = useState<BackendEvaluation | null>(null)
  const [tracesLoading, setTracesLoading] = useState(false)
  const [historicalRuns, setHistoricalRuns] = useState<RunHistory[]>([])

  const wsRef = useRef<WebSocket | null>(null)
  const logsEndRef = useRef<HTMLDivElement>(null)

  // Auto-scroll logs
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [logs])

  // Fetch historical runs on mount
  useEffect(() => {
    fetchHistoricalRuns()
  }, [])

  // Fetch traces when workflow completes
  useEffect(() => {
    if (workflow?.status === 'completed' && workflow.run_id) {
      fetchTracesAndEval(workflow.run_id)
    }
  }, [workflow?.status, workflow?.run_id])

  // Fetch historical runs from backend
  const fetchHistoricalRuns = async () => {
    try {
      const response = await fetch(`${API_URL}/logs/runs/history?limit=50`)
      if (response.ok) {
        const data = await response.json()
        if (data.runs) {
          setHistoricalRuns(data.runs.map((r: any) => ({
            run_id: r.run_id,
            company_name: r.company_name,
            status: 'completed',
            risk_level: r.risk_level,
            credit_score: r.credit_score,
            evaluation_score: r.overall_score,
            started_at: r.timestamp || new Date().toISOString(),
          })))
        }
      }
    } catch (e) {
      console.error('Failed to fetch historical runs:', e)
    }
  }

  // Fetch traces and evaluation for a run
  const fetchTracesAndEval = async (runId: string) => {
    setTracesLoading(true)
    try {
      // Fetch traces
      const tracesRes = await fetch(`${API_URL}/logs/traces/${runId}`)
      if (tracesRes.ok) {
        const tracesData = await tracesRes.json()
        setBackendTraces(tracesData.traces || [])
      }

      // Fetch evaluation
      const evalRes = await fetch(`${API_URL}/logs/evaluations/${runId}`)
      if (evalRes.ok) {
        const evalData = await evalRes.json()
        setBackendEvaluation(evalData.evaluation)
      }
    } catch (e) {
      console.error('Failed to fetch traces/eval:', e)
    } finally {
      setTracesLoading(false)
    }
  }

  // Add log entry
  const addLog = useCallback((type: string, message: string, data?: any) => {
    setLogs(prev => [...prev, {
      timestamp: new Date().toISOString(),
      type,
      message,
      data
    }])
  }, [])

  // Toggle step expansion
  const toggleStep = (stepId: string) => {
    setExpandedSteps(prev => {
      const next = new Set(prev)
      if (next.has(stepId)) {
        next.delete(stepId)
      } else {
        next.add(stepId)
      }
      return next
    })
  }

  // Extract evaluation from workflow
  useEffect(() => {
    if (workflow?.steps) {
      const evalStep = workflow.steps.find(s => s.step_id === 'evaluate')
      if (evalStep?.output_data) {
        setEvalSummary({
          tool_selection_score: evalStep.output_data.tool_selection_score || 0,
          data_quality_score: evalStep.output_data.data_quality_score || 0,
          synthesis_score: evalStep.output_data.synthesis_score || 0,
          overall_score: evalStep.output_data.overall_score || 0,
          tool_selection: evalStep.output_data.tool_selection,
          data_quality: evalStep.output_data.data_quality,
          synthesis: evalStep.output_data.synthesis,
        })
      }
    }
  }, [workflow])

  // Add to history when run completes
  useEffect(() => {
    if (workflow?.status === 'completed' || workflow?.status === 'failed') {
      const historyEntry: RunHistory = {
        run_id: workflow.run_id,
        company_name: workflow.company_name,
        status: workflow.status,
        risk_level: workflow.result?.risk_level,
        credit_score: workflow.result?.credit_score,
        evaluation_score: workflow.result?.evaluation_score,
        started_at: workflow.started_at,
        completed_at: workflow.completed_at,
        duration_ms: workflow.completed_at && workflow.started_at
          ? new Date(workflow.completed_at).getTime() - new Date(workflow.started_at).getTime()
          : undefined
      }
      setRunHistory(prev => [historyEntry, ...prev.filter(r => r.run_id !== workflow.run_id)])
    }
  }, [workflow?.status])

  // Connect to WebSocket
  const connectWebSocket = useCallback((runId: string) => {
    const wsUrl = getWebSocketUrl(`/ws/${runId}`)
    addLog('system', `Connecting to WebSocket: ${wsUrl}`)

    const ws = new WebSocket(wsUrl)

    ws.onopen = () => {
      addLog('system', 'WebSocket connected')
    }

    ws.onmessage = (event) => {
      const message = JSON.parse(event.data)

      switch (message.type) {
        case 'current_state':
        case 'workflow_started':
          setWorkflow(message.data)
          addLog('workflow', `Workflow started for ${message.data.company_name}`)
          break

        case 'step_update':
          setWorkflow(prev => {
            if (!prev) return prev
            const newSteps = [...prev.steps]
            const updatedStep = {
              ...message.data.step,
              output_data: message.data.output_data || message.data.step.output_data
            }
            newSteps[message.data.step_index] = updatedStep
            return { ...prev, steps: newSteps, current_step: message.data.step.name }
          })
          const step = message.data.step
          if (step.status === 'running') {
            addLog('step', `Starting: ${step.name}`, step)
            setExpandedSteps(prev => new Set([...prev, step.step_id]))
          } else if (step.status === 'completed') {
            addLog('step', `Completed: ${step.name} (${step.duration_ms?.toFixed(0)}ms)`, step)
          } else if (step.status === 'failed') {
            addLog('error', `Failed: ${step.name} - ${step.error}`, step)
          }
          break

        case 'step_output':
          setWorkflow(prev => {
            if (!prev) return prev
            const newSteps = [...prev.steps]
            const stepIdx = newSteps.findIndex(s => s.step_id === message.data.step_id)
            if (stepIdx >= 0) {
              newSteps[stepIdx] = {
                ...newSteps[stepIdx],
                input_data: message.data.input,
                output_data: message.data.output
              }
            }
            return { ...prev, steps: newSteps }
          })
          break

        case 'workflow_completed':
          setWorkflow(message.data)
          setIsRunning(false)
          addLog('success', `Workflow completed! Risk: ${message.data.result?.risk_level}, Score: ${message.data.result?.credit_score}`)
          break

        case 'workflow_failed':
          setWorkflow(message.data.status)
          setIsRunning(false)
          setError(message.data.error)
          addLog('error', `Workflow failed: ${message.data.error}`)
          break

        case 'ping':
          ws.send('pong')
          break
      }
    }

    ws.onerror = () => {
      addLog('error', 'WebSocket error')
    }

    ws.onclose = () => {
      addLog('system', 'WebSocket disconnected')
    }

    wsRef.current = ws
  }, [addLog])

  // Start single analysis
  const startAnalysis = async () => {
    if (!companyName.trim()) {
      setError('Please enter a company name')
      return
    }

    setError(null)
    setIsRunning(true)
    setWorkflow(null)
    setLogs([])
    setExpandedSteps(new Set())
    setEvalSummary(null)
    setDetailTab('result')

    addLog('system', `Starting analysis for: ${companyName}`)

    try {
      const response = await fetch(`${API_URL}/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ company_name: companyName })
      })

      if (!response.ok) {
        throw new Error(`API error: ${response.statusText}`)
      }

      const data = await response.json()
      addLog('system', `Run started: ${data.run_id}`)
      connectWebSocket(data.run_id)

    } catch (e: any) {
      setError(e.message)
      setIsRunning(false)
      addLog('error', `Failed to start: ${e.message}`)
    }
  }

  // Start batch analysis
  const startBatchAnalysis = async () => {
    const companies = batchCompanies.filter(c => c.trim())
    if (companies.length === 0) {
      setError('Please enter at least one company name')
      return
    }

    setError(null)
    setBatchRunning(true)
    setBatchResults([])
    setBatchProgress({ current: 0, total: companies.length })

    for (let i = 0; i < companies.length; i++) {
      setBatchProgress({ current: i + 1, total: companies.length })

      try {
        // Start analysis
        const response = await fetch(`${API_URL}/analyze`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ company_name: companies[i] })
        })

        if (!response.ok) throw new Error(`API error: ${response.statusText}`)

        const data = await response.json()

        // Poll for completion
        let status: WorkflowStatus | null = null
        for (let j = 0; j < 120; j++) { // 2 minute timeout
          await new Promise(r => setTimeout(r, 1000))
          const statusRes = await fetch(`${API_URL}/status/${data.run_id}`)
          if (statusRes.ok) {
            status = await statusRes.json()
            if (status?.status === 'completed' || status?.status === 'failed') {
              break
            }
          }
        }

        if (status) {
          setBatchResults(prev => [...prev, status!])
        }

      } catch (e: any) {
        console.error(`Failed to analyze ${companies[i]}:`, e)
      }
    }

    setBatchRunning(false)
  }

  // Stop/cancel
  const stopAnalysis = () => {
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
    setIsRunning(false)
    addLog('system', 'Analysis stopped by user')
  }

  // Reset
  const resetAnalysis = () => {
    stopAnalysis()
    setWorkflow(null)
    setLogs([])
    setError(null)
    setCompanyName('')
    setExpandedSteps(new Set())
    setEvalSummary(null)
  }

  // Add company to batch
  const addBatchCompany = () => {
    setBatchCompanies(prev => [...prev, ''])
  }

  // Remove company from batch
  const removeBatchCompany = (index: number) => {
    setBatchCompanies(prev => prev.filter((_, i) => i !== index))
  }

  // Update batch company
  const updateBatchCompany = (index: number, value: string) => {
    setBatchCompanies(prev => prev.map((c, i) => i === index ? value : c))
  }

  // Get step icon
  const getStepIcon = (status: string) => {
    switch (status) {
      case 'pending':
        return <Clock className="w-4 h-4 text-studio-muted" />
      case 'running':
        return <Loader2 className="w-4 h-4 text-studio-accent animate-spin" />
      case 'completed':
        return <CheckCircle className="w-4 h-4 text-green-500" />
      case 'failed':
        return <XCircle className="w-4 h-4 text-red-500" />
      default:
        return <Clock className="w-4 h-4 text-studio-muted" />
    }
  }

  // Get risk color
  const getRiskColor = (risk: string) => {
    switch (risk?.toLowerCase()) {
      case 'low': return 'text-green-400 bg-green-900/20 border-green-800'
      case 'medium': return 'text-yellow-400 bg-yellow-900/20 border-yellow-800'
      case 'high': return 'text-orange-400 bg-orange-900/20 border-orange-800'
      case 'critical': return 'text-red-400 bg-red-900/20 border-red-800'
      default: return 'text-gray-400 bg-gray-900/20 border-gray-800'
    }
  }

  // Format score as percentage
  const formatScore = (score: number | undefined) => {
    if (score === undefined) return 'N/A'
    return `${(score * 100).toFixed(0)}%`
  }

  return (
    <div className="min-h-screen flex flex-col bg-studio-bg">
      {/* Header */}
      <header className="border-b border-studio-border px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Activity className="w-6 h-6 text-studio-accent" />
            <h1 className="text-xl font-semibold">Credit Intelligence Playground</h1>
          </div>
          <div className="flex items-center gap-4">
            <Link
              href="/prompts"
              className="flex items-center gap-2 px-3 py-1.5 text-sm bg-studio-panel hover:bg-studio-border rounded transition-colors"
            >
              <Settings className="w-4 h-4" />
              Prompts
            </Link>
            <div className="flex items-center gap-2 text-sm text-studio-muted">
              <span className={`w-2 h-2 rounded-full ${isRunning || batchRunning ? 'bg-green-500 animate-pulse' : 'bg-gray-500'}`} />
              {isRunning || batchRunning ? 'Running' : 'Idle'}
            </div>
          </div>
        </div>
      </header>

      {/* Tab Navigation */}
      <div className="border-b border-studio-border px-6">
        <div className="flex gap-1">
          <button
            onClick={() => setActiveTab('single')}
            className={`px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
              activeTab === 'single'
                ? 'border-studio-accent text-studio-accent'
                : 'border-transparent text-studio-muted hover:text-white'
            }`}
          >
            <div className="flex items-center gap-2">
              <Building2 className="w-4 h-4" />
              Single Analysis
            </div>
          </button>
          <button
            onClick={() => setActiveTab('batch')}
            className={`px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
              activeTab === 'batch'
                ? 'border-studio-accent text-studio-accent'
                : 'border-transparent text-studio-muted hover:text-white'
            }`}
          >
            <div className="flex items-center gap-2">
              <List className="w-4 h-4" />
              Batch Analysis
            </div>
          </button>
          <button
            onClick={() => setActiveTab('history')}
            className={`px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
              activeTab === 'history'
                ? 'border-studio-accent text-studio-accent'
                : 'border-transparent text-studio-muted hover:text-white'
            }`}
          >
            <div className="flex items-center gap-2">
              <Clock className="w-4 h-4" />
              Run History ({runHistory.length})
            </div>
          </button>
        </div>
      </div>

      <div className="flex-1 flex overflow-hidden">
        {/* ==================== SINGLE ANALYSIS TAB ==================== */}
        {activeTab === 'single' && (
          <>
            {/* Left Panel - Input & Pipeline */}
            <div className="w-96 border-r border-studio-border flex flex-col">
              {/* Input Section */}
              <div className="p-4 border-b border-studio-border">
                <label className="block text-sm text-studio-muted mb-2">Company Name</label>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={companyName}
                    onChange={(e) => setCompanyName(e.target.value)}
                    placeholder="e.g., Netflix, Apple, Tesla"
                    className="flex-1 bg-studio-panel border border-studio-border rounded px-3 py-2 text-sm focus:outline-none focus:border-studio-accent"
                    disabled={isRunning}
                    onKeyDown={(e) => e.key === 'Enter' && !isRunning && startAnalysis()}
                  />
                </div>

                <div className="flex gap-2 mt-3">
                  {!isRunning ? (
                    <button
                      onClick={startAnalysis}
                      disabled={!companyName.trim()}
                      className="flex-1 flex items-center justify-center gap-2 bg-studio-accent hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed rounded px-4 py-2 text-sm font-medium transition-colors"
                    >
                      <Play className="w-4 h-4" />
                      Run Analysis
                    </button>
                  ) : (
                    <button
                      onClick={stopAnalysis}
                      className="flex-1 flex items-center justify-center gap-2 bg-red-600 hover:bg-red-700 rounded px-4 py-2 text-sm font-medium transition-colors"
                    >
                      <Square className="w-4 h-4" />
                      Stop
                    </button>
                  )}
                  <button
                    onClick={resetAnalysis}
                    className="p-2 text-studio-muted hover:text-white transition-colors"
                    title="Reset"
                  >
                    <RefreshCw className="w-4 h-4" />
                  </button>
                </div>

                {error && (
                  <div className="mt-3 p-2 bg-red-900/30 border border-red-800 rounded text-sm text-red-400">
                    {error}
                  </div>
                )}
              </div>

              {/* Workflow Pipeline */}
              <div className="flex-1 overflow-auto p-4">
                <h3 className="text-sm font-medium text-studio-muted mb-3 flex items-center gap-2">
                  <Layers className="w-4 h-4" />
                  Workflow Pipeline
                </h3>

                {workflow?.steps && workflow.steps.length > 0 ? (
                  <div className="space-y-2">
                    {workflow.steps.map((step, idx) => (
                      <div key={step.step_id}>
                        {/* Step Header */}
                        <div
                          onClick={() => toggleStep(step.step_id)}
                          className={`p-3 rounded border cursor-pointer transition-all ${
                            step.status === 'running'
                              ? 'border-studio-accent bg-studio-accent/10 pulse-glow'
                              : step.status === 'completed'
                              ? 'border-green-800 bg-green-900/20 hover:bg-green-900/30'
                              : step.status === 'failed'
                              ? 'border-red-800 bg-red-900/20 hover:bg-red-900/30'
                              : 'border-studio-border bg-studio-panel hover:bg-studio-border/50'
                          } ${expandedSteps.has(step.step_id) ? 'rounded-b-none' : ''}`}
                        >
                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2">
                              {getStepIcon(step.status)}
                              <span className="text-sm font-medium">{step.name}</span>
                            </div>
                            <div className="flex items-center gap-2">
                              {step.duration_ms && (
                                <span className="text-xs text-studio-muted">
                                  {step.duration_ms.toFixed(0)}ms
                                </span>
                              )}
                              {step.status !== 'pending' && (
                                expandedSteps.has(step.step_id)
                                  ? <ChevronDown className="w-4 h-4 text-studio-muted" />
                                  : <ChevronRight className="w-4 h-4 text-studio-muted" />
                              )}
                            </div>
                          </div>
                          {step.error && (
                            <div className="text-xs text-red-400 mt-2 ml-6">{step.error}</div>
                          )}
                        </div>

                        {/* Step Details */}
                        {expandedSteps.has(step.step_id) && step.status !== 'pending' && (
                          <div className="border border-t-0 border-studio-border rounded-b bg-black/50 p-3 max-h-64 overflow-auto">
                            {step.output_summary && (
                              <div className="mb-2 p-2 bg-studio-accent/10 rounded border border-studio-accent/30">
                                <div className="text-xs text-studio-accent font-medium">Summary</div>
                                <div className="text-sm text-white">{step.output_summary}</div>
                              </div>
                            )}
                            {step.output_data && Object.keys(step.output_data).length > 0 && (
                              <div className="space-y-1">
                                {Object.entries(step.output_data).slice(0, 5).map(([key, value]) => (
                                  <div key={key} className="text-xs">
                                    <span className="text-yellow-400 font-mono">{key}:</span>{' '}
                                    <span className="text-green-300">
                                      {typeof value === 'object' ? JSON.stringify(value).slice(0, 100) + '...' : String(value).slice(0, 100)}
                                    </span>
                                  </div>
                                ))}
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-8 text-studio-muted text-sm">
                    Enter a company name and click "Run Analysis" to start
                  </div>
                )}
              </div>
            </div>

            {/* Main Content */}
            <div className="flex-1 flex flex-col overflow-hidden">
              {/* Detail Tabs */}
              {workflow && (
                <div className="border-b border-studio-border px-4">
                  <div className="flex gap-1">
                    {['result', 'evaluation', 'traces', 'pipeline'].map((tab) => (
                      <button
                        key={tab}
                        onClick={() => setDetailTab(tab as DetailTab)}
                        className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors capitalize ${
                          detailTab === tab
                            ? 'border-studio-accent text-studio-accent'
                            : 'border-transparent text-studio-muted hover:text-white'
                        }`}
                      >
                        {tab === 'result' && <span className="flex items-center gap-2"><FileText className="w-4 h-4" />Result</span>}
                        {tab === 'evaluation' && <span className="flex items-center gap-2"><BarChart3 className="w-4 h-4" />Evaluation</span>}
                        {tab === 'traces' && <span className="flex items-center gap-2"><Eye className="w-4 h-4" />Traces</span>}
                        {tab === 'pipeline' && <span className="flex items-center gap-2"><Layers className="w-4 h-4" />Pipeline</span>}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {/* Tab Content */}
              <div className="flex-1 overflow-auto p-4">
                {/* Result Tab */}
                {detailTab === 'result' && workflow?.result && (
                  <div className="space-y-4">
                    <h2 className="text-lg font-semibold flex items-center gap-2">
                      <Building2 className="w-5 h-5" />
                      Assessment: {workflow.company_name}
                    </h2>

                    <div className="grid grid-cols-4 gap-4">
                      <div className={`p-4 rounded-lg border ${getRiskColor(workflow.result.risk_level)}`}>
                        <div className="text-sm text-studio-muted mb-1">Risk Level</div>
                        <div className="text-2xl font-bold capitalize">{workflow.result.risk_level}</div>
                      </div>
                      <div className="p-4 rounded-lg border border-studio-border bg-studio-panel">
                        <div className="text-sm text-studio-muted mb-1">Credit Score</div>
                        <div className="text-2xl font-bold text-studio-accent">{workflow.result.credit_score}</div>
                      </div>
                      <div className="p-4 rounded-lg border border-studio-border bg-studio-panel">
                        <div className="text-sm text-studio-muted mb-1">Confidence</div>
                        <div className="text-2xl font-bold">{(workflow.result.confidence * 100).toFixed(0)}%</div>
                      </div>
                      <div className="p-4 rounded-lg border border-studio-border bg-studio-panel">
                        <div className="text-sm text-studio-muted mb-1">Eval Score</div>
                        <div className="text-2xl font-bold">{(workflow.result.evaluation_score * 100).toFixed(0)}%</div>
                      </div>
                    </div>

                    {workflow.result.reasoning && (
                      <div className="p-4 rounded-lg border border-studio-border bg-studio-panel">
                        <div className="text-sm text-studio-muted mb-2 flex items-center gap-2">
                          <FileText className="w-4 h-4" />
                          Analysis Summary
                        </div>
                        <p className="text-sm">{workflow.result.reasoning}</p>
                      </div>
                    )}

                    {workflow.result.recommendations?.length > 0 && (
                      <div className="p-4 rounded-lg border border-studio-border bg-studio-panel">
                        <div className="text-sm text-studio-muted mb-2 flex items-center gap-2">
                          <TrendingUp className="w-4 h-4" />
                          Recommendations
                        </div>
                        <ul className="text-sm space-y-1">
                          {workflow.result.recommendations.map((rec, idx) => (
                            <li key={idx} className="flex items-start gap-2">
                              <ChevronRight className="w-4 h-4 mt-0.5 text-studio-muted" />
                              {rec}
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                )}

                {/* Evaluation Tab */}
                {detailTab === 'evaluation' && (
                  <div className="space-y-4">
                    <h2 className="text-lg font-semibold flex items-center gap-2">
                      <BarChart3 className="w-5 h-5" />
                      Evaluation Summary
                    </h2>

                    {(evalSummary || backendEvaluation) ? (
                      <>
                        {/* Score Cards - use evalSummary or backendEvaluation */}
                        <div className="grid grid-cols-4 gap-4">
                          <div className="p-4 rounded-lg border border-studio-border bg-studio-panel">
                            <div className="text-sm text-studio-muted mb-1">Tool Selection</div>
                            <div className="text-2xl font-bold text-blue-400">
                              {formatScore(evalSummary?.tool_selection_score ?? backendEvaluation?.tool_selection_score)}
                            </div>
                          </div>
                          <div className="p-4 rounded-lg border border-studio-border bg-studio-panel">
                            <div className="text-sm text-studio-muted mb-1">Data Quality</div>
                            <div className="text-2xl font-bold text-purple-400">
                              {formatScore(evalSummary?.data_quality_score ?? backendEvaluation?.data_quality_score)}
                            </div>
                          </div>
                          <div className="p-4 rounded-lg border border-studio-border bg-studio-panel">
                            <div className="text-sm text-studio-muted mb-1">Synthesis</div>
                            <div className="text-2xl font-bold text-green-400">
                              {formatScore(evalSummary?.synthesis_score ?? backendEvaluation?.synthesis_score)}
                            </div>
                          </div>
                          <div className="p-4 rounded-lg border border-studio-border bg-studio-panel">
                            <div className="text-sm text-studio-muted mb-1">Overall</div>
                            <div className="text-2xl font-bold text-studio-accent">
                              {formatScore(evalSummary?.overall_score ?? backendEvaluation?.overall_score)}
                            </div>
                          </div>
                        </div>

                        {/* Detailed Breakdowns */}
                        <div className="grid grid-cols-3 gap-4">
                          {(evalSummary?.tool_selection || backendEvaluation?.tool_selection) && (
                            <div className="p-4 rounded-lg border border-studio-border bg-studio-panel">
                              <div className="text-sm font-medium mb-2 text-blue-400">Tool Selection Details</div>
                              <div className="text-xs space-y-1 text-studio-muted">
                                {(() => {
                                  const ts = evalSummary?.tool_selection || backendEvaluation?.tool_selection || {}
                                  return (
                                    <>
                                      <p><span className="text-green-400">Precision:</span> {((ts.precision || 0) * 100).toFixed(0)}%</p>
                                      <p><span className="text-green-400">Recall:</span> {((ts.recall || 0) * 100).toFixed(0)}%</p>
                                      <p><span className="text-green-400">F1:</span> {((ts.f1_score || 0) * 100).toFixed(0)}%</p>
                                      {ts.reasoning && (
                                        <p className="mt-2 text-white">{ts.reasoning}</p>
                                      )}
                                    </>
                                  )
                                })()}
                              </div>
                            </div>
                          )}
                          {(evalSummary?.data_quality || backendEvaluation?.data_quality) && (
                            <div className="p-4 rounded-lg border border-studio-border bg-studio-panel">
                              <div className="text-sm font-medium mb-2 text-purple-400">Data Quality Details</div>
                              <div className="text-xs space-y-1 text-studio-muted">
                                {(() => {
                                  const dq = evalSummary?.data_quality || backendEvaluation?.data_quality || {}
                                  return (
                                    <>
                                      <p><span className="text-green-400">Sources:</span> {dq.sources_used?.join(', ') || 'N/A'}</p>
                                      <p><span className="text-green-400">Completeness:</span> {((dq.completeness || 0) * 100).toFixed(0)}%</p>
                                      {dq.reasoning && (
                                        <p className="mt-2 text-white">{dq.reasoning}</p>
                                      )}
                                    </>
                                  )
                                })()}
                              </div>
                            </div>
                          )}
                          {(evalSummary?.synthesis || backendEvaluation?.synthesis) && (
                            <div className="p-4 rounded-lg border border-studio-border bg-studio-panel">
                              <div className="text-sm font-medium mb-2 text-green-400">Synthesis Details</div>
                              <div className="text-xs space-y-1 text-studio-muted">
                                {(() => {
                                  const syn = evalSummary?.synthesis || backendEvaluation?.synthesis || {}
                                  return (
                                    <>
                                      <p><span className="text-green-400">Score:</span> {((syn.score || 0) * 100).toFixed(0)}%</p>
                                      <p><span className="text-green-400">Fields:</span> {syn.present_fields?.join(', ') || 'N/A'}</p>
                                      {syn.reasoning && (
                                        <p className="mt-2 text-white">{syn.reasoning}</p>
                                      )}
                                    </>
                                  )
                                })()}
                              </div>
                            </div>
                          )}
                        </div>

                        {/* Backend Evaluation Source */}
                        {backendEvaluation && (
                          <div className="text-xs text-studio-muted flex items-center gap-2">
                            <span>Source: MongoDB</span>
                            <span>|</span>
                            <span>Evaluated: {backendEvaluation.evaluated_at ? new Date(backendEvaluation.evaluated_at).toLocaleString() : 'N/A'}</span>
                          </div>
                        )}
                      </>
                    ) : (
                      <div className="text-center py-12 text-studio-muted">
                        No evaluation data available. Run an analysis first.
                      </div>
                    )}
                  </div>
                )}

                {/* Traces Tab */}
                {detailTab === 'traces' && (
                  <div className="space-y-4">
                    <h2 className="text-lg font-semibold flex items-center gap-2">
                      <Eye className="w-5 h-5" />
                      Execution Traces
                      {tracesLoading && <Loader2 className="w-4 h-4 animate-spin ml-2" />}
                    </h2>

                    {/* Backend Traces from MongoDB */}
                    {backendTraces.length > 0 && (
                      <div className="p-4 rounded-lg border border-studio-border bg-studio-panel">
                        <h3 className="text-sm font-medium mb-3 text-purple-400">LangGraph Events</h3>
                        <div className="overflow-x-auto">
                          <table className="w-full text-xs">
                            <thead className="bg-black/30">
                              <tr>
                                <th className="text-left p-2 text-studio-muted">Timestamp</th>
                                <th className="text-left p-2 text-studio-muted">Event</th>
                                <th className="text-left p-2 text-studio-muted">Type</th>
                                <th className="text-left p-2 text-studio-muted">Status</th>
                                <th className="text-left p-2 text-studio-muted">Duration</th>
                                <th className="text-left p-2 text-studio-muted">Model</th>
                                <th className="text-left p-2 text-studio-muted">Tokens</th>
                              </tr>
                            </thead>
                            <tbody>
                              {backendTraces.slice(0, 50).map((trace, idx) => (
                                <tr key={trace._id || idx} className="border-t border-studio-border/50 hover:bg-black/20">
                                  <td className="p-2 text-studio-muted font-mono">
                                    {new Date(trace.timestamp).toLocaleTimeString()}
                                  </td>
                                  <td className="p-2 text-white">{trace.event_name}</td>
                                  <td className="p-2">
                                    <span className="px-1.5 py-0.5 rounded bg-blue-900/30 text-blue-400">
                                      {trace.event_type}
                                    </span>
                                  </td>
                                  <td className="p-2">
                                    <span className={`px-1.5 py-0.5 rounded ${
                                      trace.status === 'success' || trace.status === 'completed'
                                        ? 'bg-green-900/30 text-green-400'
                                        : trace.status === 'error' || trace.status === 'failed'
                                        ? 'bg-red-900/30 text-red-400'
                                        : 'bg-gray-900/30 text-gray-400'
                                    }`}>
                                      {trace.status}
                                    </span>
                                  </td>
                                  <td className="p-2 text-studio-muted">
                                    {trace.duration_ms ? `${trace.duration_ms.toFixed(0)}ms` : '-'}
                                  </td>
                                  <td className="p-2 text-studio-muted">{trace.model || '-'}</td>
                                  <td className="p-2 text-studio-muted">{trace.tokens || '-'}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                        {backendTraces.length > 50 && (
                          <div className="text-xs text-studio-muted mt-2">
                            Showing 50 of {backendTraces.length} events
                          </div>
                        )}
                      </div>
                    )}

                    {/* Local Logs */}
                    <div className="bg-black/50 rounded-lg border border-studio-border p-4 max-h-[400px] overflow-auto">
                      <h3 className="text-sm font-medium mb-3 text-green-400">WebSocket Logs</h3>
                      {logs.length > 0 ? (
                        <div className="space-y-1 font-mono text-xs">
                          {logs.map((log, idx) => (
                            <div key={idx} className="flex gap-3">
                              <span className="text-studio-muted w-24 flex-shrink-0">
                                {new Date(log.timestamp).toLocaleTimeString()}
                              </span>
                              <span className={`w-16 flex-shrink-0 ${
                                log.type === 'error' ? 'text-red-400' :
                                log.type === 'success' ? 'text-green-400' :
                                log.type === 'step' ? 'text-studio-accent' :
                                log.type === 'workflow' ? 'text-purple-400' :
                                log.type === 'output' ? 'text-yellow-400' :
                                'text-studio-muted'
                              }`}>
                                [{log.type}]
                              </span>
                              <span className="text-white">{log.message}</span>
                            </div>
                          ))}
                          <div ref={logsEndRef} />
                        </div>
                      ) : (
                        <div className="text-center py-4 text-studio-muted">
                          No local logs yet.
                        </div>
                      )}
                    </div>

                    {workflow?.run_id && (
                      <div className="flex items-center gap-2 text-sm text-studio-muted">
                        <span>Run ID: {workflow.run_id}</span>
                        <button
                          onClick={() => navigator.clipboard.writeText(workflow.run_id)}
                          className="text-studio-accent hover:underline"
                        >
                          Copy
                        </button>
                        <button
                          onClick={() => fetchTracesAndEval(workflow.run_id)}
                          className="ml-4 text-studio-accent hover:underline flex items-center gap-1"
                        >
                          <RefreshCw className="w-3 h-3" />
                          Refresh Traces
                        </button>
                      </div>
                    )}
                  </div>
                )}

                {/* Pipeline Tab */}
                {detailTab === 'pipeline' && workflow && (
                  <div className="space-y-4">
                    <h2 className="text-lg font-semibold flex items-center gap-2">
                      <Layers className="w-5 h-5" />
                      Pipeline Overview
                    </h2>

                    {/* Pipeline Diagram */}
                    <div className="p-4 rounded-lg border border-studio-border bg-studio-panel">
                      <div className="flex items-center justify-between gap-2 overflow-x-auto">
                        {workflow.steps.map((step, idx) => (
                          <div key={step.step_id} className="flex items-center">
                            <div className={`flex flex-col items-center p-3 rounded-lg min-w-[120px] ${
                              step.status === 'completed' ? 'bg-green-900/30 border border-green-800' :
                              step.status === 'running' ? 'bg-blue-900/30 border border-blue-800' :
                              step.status === 'failed' ? 'bg-red-900/30 border border-red-800' :
                              'bg-gray-900/30 border border-gray-800'
                            }`}>
                              {getStepIcon(step.status)}
                              <span className="text-xs mt-2 text-center">{step.name}</span>
                              {step.duration_ms && (
                                <span className="text-xs text-studio-muted">{step.duration_ms.toFixed(0)}ms</span>
                              )}
                            </div>
                            {idx < workflow.steps.length - 1 && (
                              <ChevronRight className="w-4 h-4 text-studio-muted mx-2" />
                            )}
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Step Details Table */}
                    <div className="rounded-lg border border-studio-border overflow-hidden">
                      <table className="w-full text-sm">
                        <thead className="bg-studio-panel">
                          <tr>
                            <th className="text-left p-3 text-studio-muted">Step</th>
                            <th className="text-left p-3 text-studio-muted">Status</th>
                            <th className="text-left p-3 text-studio-muted">Duration</th>
                            <th className="text-left p-3 text-studio-muted">Output</th>
                          </tr>
                        </thead>
                        <tbody>
                          {workflow.steps.map((step) => (
                            <tr key={step.step_id} className="border-t border-studio-border">
                              <td className="p-3 font-medium">{step.name}</td>
                              <td className="p-3">
                                <span className={`px-2 py-1 rounded text-xs ${
                                  step.status === 'completed' ? 'bg-green-900/50 text-green-400' :
                                  step.status === 'running' ? 'bg-blue-900/50 text-blue-400' :
                                  step.status === 'failed' ? 'bg-red-900/50 text-red-400' :
                                  'bg-gray-900/50 text-gray-400'
                                }`}>
                                  {step.status}
                                </span>
                              </td>
                              <td className="p-3 text-studio-muted">
                                {step.duration_ms ? `${step.duration_ms.toFixed(0)}ms` : '-'}
                              </td>
                              <td className="p-3 text-studio-muted text-xs truncate max-w-xs">
                                {step.output_summary || (step.error ? `Error: ${step.error}` : '-')}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}

                {/* No workflow state */}
                {!workflow && (
                  <div className="flex items-center justify-center h-full text-studio-muted">
                    <div className="text-center">
                      <Activity className="w-12 h-12 mx-auto mb-4 opacity-50" />
                      <p>Enter a company name and run an analysis to see results</p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </>
        )}

        {/* ==================== BATCH ANALYSIS TAB ==================== */}
        {activeTab === 'batch' && (
          <div className="flex-1 p-6 overflow-auto">
            <div className="max-w-4xl mx-auto">
              <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <List className="w-5 h-5" />
                Batch Company Analysis
              </h2>

              {/* Company List Input */}
              <div className="p-4 rounded-lg border border-studio-border bg-studio-panel mb-4">
                <div className="flex items-center justify-between mb-3">
                  <label className="text-sm text-studio-muted">Companies to Analyze</label>
                  <button
                    onClick={addBatchCompany}
                    className="text-sm text-studio-accent hover:underline flex items-center gap-1"
                    disabled={batchRunning}
                  >
                    <Plus className="w-4 h-4" />
                    Add Company
                  </button>
                </div>

                <div className="space-y-2">
                  {batchCompanies.map((company, idx) => (
                    <div key={idx} className="flex gap-2">
                      <input
                        type="text"
                        value={company}
                        onChange={(e) => updateBatchCompany(idx, e.target.value)}
                        placeholder={`Company ${idx + 1}`}
                        className="flex-1 bg-black/50 border border-studio-border rounded px-3 py-2 text-sm focus:outline-none focus:border-studio-accent"
                        disabled={batchRunning}
                      />
                      {batchCompanies.length > 1 && (
                        <button
                          onClick={() => removeBatchCompany(idx)}
                          className="p-2 text-red-400 hover:text-red-300"
                          disabled={batchRunning}
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      )}
                    </div>
                  ))}
                </div>

                <div className="flex gap-2 mt-4">
                  <button
                    onClick={startBatchAnalysis}
                    disabled={batchRunning || batchCompanies.every(c => !c.trim())}
                    className="flex-1 flex items-center justify-center gap-2 bg-studio-accent hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed rounded px-4 py-2 text-sm font-medium"
                  >
                    {batchRunning ? (
                      <>
                        <Loader2 className="w-4 h-4 animate-spin" />
                        Running {batchProgress.current}/{batchProgress.total}...
                      </>
                    ) : (
                      <>
                        <Play className="w-4 h-4" />
                        Run Batch Analysis
                      </>
                    )}
                  </button>
                </div>
              </div>

              {/* Batch Results */}
              {batchResults.length > 0 && (
                <div className="rounded-lg border border-studio-border overflow-hidden">
                  <div className="p-3 bg-studio-panel border-b border-studio-border flex items-center justify-between">
                    <h3 className="font-medium">Batch Results ({batchResults.length})</h3>
                  </div>
                  <table className="w-full text-sm">
                    <thead className="bg-studio-panel/50">
                      <tr>
                        <th className="text-left p-3 text-studio-muted">Company</th>
                        <th className="text-left p-3 text-studio-muted">Status</th>
                        <th className="text-left p-3 text-studio-muted">Risk</th>
                        <th className="text-left p-3 text-studio-muted">Score</th>
                        <th className="text-left p-3 text-studio-muted">Confidence</th>
                        <th className="text-left p-3 text-studio-muted">Eval</th>
                      </tr>
                    </thead>
                    <tbody>
                      {batchResults.map((result) => (
                        <tr key={result.run_id} className="border-t border-studio-border hover:bg-studio-panel/30">
                          <td className="p-3 font-medium">{result.company_name}</td>
                          <td className="p-3">
                            <span className={`px-2 py-1 rounded text-xs ${
                              result.status === 'completed' ? 'bg-green-900/50 text-green-400' :
                              'bg-red-900/50 text-red-400'
                            }`}>
                              {result.status}
                            </span>
                          </td>
                          <td className="p-3">
                            <span className={`px-2 py-1 rounded text-xs ${getRiskColor(result.result?.risk_level || '')}`}>
                              {result.result?.risk_level || 'N/A'}
                            </span>
                          </td>
                          <td className="p-3">{result.result?.credit_score || 'N/A'}</td>
                          <td className="p-3">{result.result?.confidence ? `${(result.result.confidence * 100).toFixed(0)}%` : 'N/A'}</td>
                          <td className="p-3">{result.result?.evaluation_score ? `${(result.result.evaluation_score * 100).toFixed(0)}%` : 'N/A'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </div>
        )}

        {/* ==================== HISTORY TAB ==================== */}
        {activeTab === 'history' && (
          <div className="flex-1 p-6 overflow-auto">
            <div className="max-w-5xl mx-auto space-y-6">
              {/* Session Runs */}
              <div>
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-lg font-semibold flex items-center gap-2">
                    <Clock className="w-5 h-5" />
                    Session Runs ({runHistory.length})
                  </h2>
                </div>

                {runHistory.length > 0 ? (
                  <div className="rounded-lg border border-studio-border overflow-hidden">
                    <table className="w-full text-sm">
                      <thead className="bg-studio-panel">
                        <tr>
                          <th className="text-left p-3 text-studio-muted">Company</th>
                          <th className="text-left p-3 text-studio-muted">Status</th>
                          <th className="text-left p-3 text-studio-muted">Risk</th>
                          <th className="text-left p-3 text-studio-muted">Score</th>
                          <th className="text-left p-3 text-studio-muted">Duration</th>
                          <th className="text-left p-3 text-studio-muted">Time</th>
                        </tr>
                      </thead>
                      <tbody>
                        {runHistory.map((run) => (
                          <tr key={run.run_id} className="border-t border-studio-border hover:bg-studio-panel/30">
                            <td className="p-3 font-medium">{run.company_name}</td>
                            <td className="p-3">
                              <span className={`px-2 py-1 rounded text-xs ${
                                run.status === 'completed' ? 'bg-green-900/50 text-green-400' :
                                'bg-red-900/50 text-red-400'
                              }`}>
                                {run.status}
                              </span>
                            </td>
                            <td className="p-3">
                              <span className={`px-2 py-1 rounded text-xs ${getRiskColor(run.risk_level || '')}`}>
                                {run.risk_level || 'N/A'}
                              </span>
                            </td>
                            <td className="p-3">{run.credit_score || 'N/A'}</td>
                            <td className="p-3 text-studio-muted">
                              {run.duration_ms ? `${(run.duration_ms / 1000).toFixed(1)}s` : '-'}
                            </td>
                            <td className="p-3 text-studio-muted">
                              {new Date(run.started_at).toLocaleString()}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <div className="text-center py-8 text-studio-muted border border-studio-border rounded-lg bg-studio-panel">
                    <p>No session runs yet. Run an analysis to see history.</p>
                  </div>
                )}
              </div>

              {/* Historical Runs from Backend */}
              <div>
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-lg font-semibold flex items-center gap-2">
                    <Activity className="w-5 h-5" />
                    Historical Runs ({historicalRuns.length})
                  </h2>
                  <button
                    onClick={fetchHistoricalRuns}
                    className="text-sm text-studio-accent hover:underline flex items-center gap-1"
                  >
                    <RefreshCw className="w-3 h-3" />
                    Refresh
                  </button>
                </div>

                {historicalRuns.length > 0 ? (
                  <div className="rounded-lg border border-studio-border overflow-hidden">
                    <table className="w-full text-sm">
                      <thead className="bg-studio-panel">
                        <tr>
                          <th className="text-left p-3 text-studio-muted">Company</th>
                          <th className="text-left p-3 text-studio-muted">Risk</th>
                          <th className="text-left p-3 text-studio-muted">Score</th>
                          <th className="text-left p-3 text-studio-muted">Confidence</th>
                          <th className="text-left p-3 text-studio-muted">Time</th>
                          <th className="text-left p-3 text-studio-muted">Actions</th>
                        </tr>
                      </thead>
                      <tbody>
                        {historicalRuns.map((run) => (
                          <tr key={run.run_id} className="border-t border-studio-border hover:bg-studio-panel/30">
                            <td className="p-3 font-medium">{run.company_name}</td>
                            <td className="p-3">
                              <span className={`px-2 py-1 rounded text-xs ${getRiskColor(run.risk_level || '')}`}>
                                {run.risk_level || 'N/A'}
                              </span>
                            </td>
                            <td className="p-3">{run.credit_score || 'N/A'}</td>
                            <td className="p-3 text-studio-muted">
                              {run.evaluation_score ? `${(run.evaluation_score * 100).toFixed(0)}%` : '-'}
                            </td>
                            <td className="p-3 text-studio-muted">
                              {new Date(run.started_at).toLocaleString()}
                            </td>
                            <td className="p-3">
                              <button
                                onClick={() => {
                                  setCompanyName(run.company_name)
                                  setActiveTab('single')
                                }}
                                className="text-studio-accent hover:underline text-xs"
                              >
                                Re-analyze
                              </button>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <div className="text-center py-8 text-studio-muted border border-studio-border rounded-lg bg-studio-panel">
                    <Clock className="w-8 h-8 mx-auto mb-2 opacity-50" />
                    <p>No historical data. Connect MongoDB to see past runs.</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
