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
  X
} from 'lucide-react'

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
  // NEW: Store full output data
  input_data?: any
  output_data?: any
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

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export default function CreditIntelligenceStudio() {
  const [companyName, setCompanyName] = useState('')
  const [isRunning, setIsRunning] = useState(false)
  const [workflow, setWorkflow] = useState<WorkflowStatus | null>(null)
  const [logs, setLogs] = useState<LogEntry[]>([])
  const [error, setError] = useState<string | null>(null)
  const [selectedStep, setSelectedStep] = useState<WorkflowStep | null>(null)
  const [expandedSteps, setExpandedSteps] = useState<Set<string>>(new Set())

  const wsRef = useRef<WebSocket | null>(null)
  const logsEndRef = useRef<HTMLDivElement>(null)

  // Auto-scroll logs
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [logs])

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

  // Connect to WebSocket
  const connectWebSocket = useCallback((runId: string) => {
    const wsUrl = API_URL.replace('http', 'ws') + `/ws/${runId}`
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
            // Merge step data with output_data
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
            // Auto-expand running step
            setExpandedSteps(prev => new Set([...prev, step.step_id]))
          } else if (step.status === 'completed') {
            const outputInfo = message.data.output_data ? ` | ${Object.keys(message.data.output_data).length} fields` : ''
            addLog('step', `Completed: ${step.name} (${step.duration_ms?.toFixed(0)}ms)${outputInfo}`, step)
          } else if (step.status === 'failed') {
            addLog('error', `Failed: ${step.name} - ${step.error}`, step)
          }
          break

        case 'step_output':
          // NEW: Handle detailed step output
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
          addLog('output', `Output for ${message.data.step_id}`, message.data)
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

    ws.onerror = (e) => {
      addLog('error', 'WebSocket error')
      console.error('WebSocket error:', e)
    }

    ws.onclose = () => {
      addLog('system', 'WebSocket disconnected')
    }

    wsRef.current = ws
  }, [addLog])

  // Start analysis
  const startAnalysis = async () => {
    if (!companyName.trim()) {
      setError('Please enter a company name')
      return
    }

    setError(null)
    setIsRunning(true)
    setWorkflow(null)
    setLogs([])
    setSelectedStep(null)
    setExpandedSteps(new Set())

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

      // Connect WebSocket for real-time updates
      connectWebSocket(data.run_id)

    } catch (e: any) {
      setError(e.message)
      setIsRunning(false)
      addLog('error', `Failed to start: ${e.message}`)
    }
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
    setSelectedStep(null)
    setExpandedSteps(new Set())
  }

  // Format JSON for display
  const formatJSON = (data: any) => {
    if (!data) return 'No data'
    try {
      return JSON.stringify(data, null, 2)
    } catch {
      return String(data)
    }
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

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="border-b border-studio-border px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Activity className="w-6 h-6 text-studio-accent" />
            <h1 className="text-xl font-semibold">Credit Intelligence Studio</h1>
          </div>
          <div className="flex items-center gap-2 text-sm text-studio-muted">
            <span className={`w-2 h-2 rounded-full ${isRunning ? 'bg-green-500 animate-pulse' : 'bg-gray-500'}`} />
            {isRunning ? 'Running' : 'Idle'}
          </div>
        </div>
      </header>

      <div className="flex-1 flex">
        {/* Left Panel - Input & Steps */}
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

          {/* Workflow Steps */}
          <div className="flex-1 overflow-auto p-4">
            <h3 className="text-sm font-medium text-studio-muted mb-3">Workflow Steps</h3>
            <p className="text-xs text-studio-muted mb-3">Click on a step to see its output</p>

            {workflow?.steps && workflow.steps.length > 0 ? (
              <div className="space-y-2">
                {workflow.steps.map((step, idx) => (
                  <div key={step.step_id}>
                    {/* Step Header - Clickable */}
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
                        <div className="text-xs text-red-400 mt-2 ml-6">
                          {step.error}
                        </div>
                      )}
                    </div>

                    {/* Step Details - Expandable */}
                    {expandedSteps.has(step.step_id) && step.status !== 'pending' && (
                      <div className="border border-t-0 border-studio-border rounded-b bg-black/50 p-3 max-h-96 overflow-auto">
                        {/* Output Summary */}
                        {step.output_summary && (
                          <div className="mb-3 p-2 bg-studio-accent/10 rounded border border-studio-accent/30">
                            <div className="text-xs text-studio-accent font-medium mb-1">Summary</div>
                            <div className="text-sm text-white">{step.output_summary}</div>
                          </div>
                        )}

                        {/* Output Data */}
                        {step.output_data && Object.keys(step.output_data).length > 0 ? (
                          <div className="space-y-2">
                            <div className="text-xs text-studio-muted font-medium">Details</div>
                            {Object.entries(step.output_data).map(([key, value]) => (
                              <div key={key} className="bg-black/30 rounded p-2">
                                <div className="text-xs text-yellow-400 font-mono mb-1">{key}:</div>
                                <div className="text-xs font-mono text-studio-text pl-2">
                                  {typeof value === 'object' ? (
                                    <pre className="whitespace-pre-wrap break-words text-green-300">
                                      {JSON.stringify(value, null, 2)}
                                    </pre>
                                  ) : (
                                    <span className="text-green-300">{String(value)}</span>
                                  )}
                                </div>
                              </div>
                            ))}
                          </div>
                        ) : !step.output_summary ? (
                          <div className="text-xs text-studio-muted italic">
                            {step.status === 'running' ? (
                              <span className="flex items-center gap-2">
                                <Loader2 className="w-3 h-3 animate-spin" />
                                Processing...
                              </span>
                            ) : 'No output data available'}
                          </div>
                        ) : null}
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
        <div className="flex-1 flex flex-col">
          {/* Results Panel */}
          {workflow?.result && (
            <div className="border-b border-studio-border p-6">
              <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <Building2 className="w-5 h-5" />
                Assessment Result: {workflow.company_name}
              </h2>

              <div className="grid grid-cols-4 gap-4">
                {/* Risk Level */}
                <div className={`p-4 rounded-lg border ${
                  workflow.result.risk_level === 'low' ? 'border-green-800 bg-green-900/20' :
                  workflow.result.risk_level === 'medium' ? 'border-yellow-800 bg-yellow-900/20' :
                  workflow.result.risk_level === 'high' ? 'border-orange-800 bg-orange-900/20' :
                  'border-red-800 bg-red-900/20'
                }`}>
                  <div className="text-sm text-studio-muted mb-1">Risk Level</div>
                  <div className={`text-2xl font-bold capitalize ${
                    workflow.result.risk_level === 'low' ? 'text-green-400' :
                    workflow.result.risk_level === 'medium' ? 'text-yellow-400' :
                    workflow.result.risk_level === 'high' ? 'text-orange-400' :
                    'text-red-400'
                  }`}>
                    {workflow.result.risk_level}
                  </div>
                </div>

                {/* Credit Score */}
                <div className="p-4 rounded-lg border border-studio-border bg-studio-panel">
                  <div className="text-sm text-studio-muted mb-1">Credit Score</div>
                  <div className="text-2xl font-bold text-studio-accent">
                    {workflow.result.credit_score}
                  </div>
                </div>

                {/* Confidence */}
                <div className="p-4 rounded-lg border border-studio-border bg-studio-panel">
                  <div className="text-sm text-studio-muted mb-1">Confidence</div>
                  <div className="text-2xl font-bold">
                    {(workflow.result.confidence * 100).toFixed(0)}%
                  </div>
                </div>

                {/* Evaluation */}
                <div className="p-4 rounded-lg border border-studio-border bg-studio-panel">
                  <div className="text-sm text-studio-muted mb-1">Evaluation Score</div>
                  <div className="text-2xl font-bold">
                    {(workflow.result.evaluation_score * 100).toFixed(0)}%
                  </div>
                </div>
              </div>

              {/* Reasoning */}
              {workflow.result.reasoning && (
                <div className="mt-4 p-4 rounded-lg border border-studio-border bg-studio-panel">
                  <div className="text-sm text-studio-muted mb-2 flex items-center gap-2">
                    <FileText className="w-4 h-4" />
                    Analysis Summary
                  </div>
                  <p className="text-sm">{workflow.result.reasoning}</p>
                </div>
              )}

              {/* Recommendations */}
              {workflow.result.recommendations && workflow.result.recommendations.length > 0 && (
                <div className="mt-4 p-4 rounded-lg border border-studio-border bg-studio-panel">
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

          {/* Logs Panel */}
          <div className="flex-1 flex flex-col">
            <div className="px-4 py-2 border-b border-studio-border bg-studio-panel flex items-center justify-between">
              <h3 className="text-sm font-medium">Execution Logs</h3>
              <span className="text-xs text-studio-muted">{logs.length} entries</span>
            </div>

            <div className="flex-1 overflow-auto p-4 log-container bg-black/30">
              {logs.length > 0 ? (
                <div className="space-y-1">
                  {logs.map((log, idx) => (
                    <div key={idx} className="flex gap-3">
                      <span className="text-studio-muted text-xs w-20 flex-shrink-0">
                        {new Date(log.timestamp).toLocaleTimeString()}
                      </span>
                      <span className={`text-xs w-16 flex-shrink-0 ${
                        log.type === 'error' ? 'text-red-400' :
                        log.type === 'success' ? 'text-green-400' :
                        log.type === 'step' ? 'text-studio-accent' :
                        log.type === 'workflow' ? 'text-purple-400' :
                        log.type === 'output' ? 'text-yellow-400' :
                        'text-studio-muted'
                      }`}>
                        [{log.type}]
                      </span>
                      <span className="text-sm">{log.message}</span>
                    </div>
                  ))}
                  <div ref={logsEndRef} />
                </div>
              ) : (
                <div className="text-center py-8 text-studio-muted text-sm">
                  Logs will appear here when you run an analysis
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
