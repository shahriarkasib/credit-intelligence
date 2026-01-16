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
  SlidersHorizontal,
  List,
  BarChart3,
  Eye,
  Layers,
  Plus,
  Trash2,
  Download,
  ExternalLink,
  HelpCircle,
  Info,
  Cpu,
  GitBranch,
  Shield,
  AlertCircle,
  Database,
} from 'lucide-react'
import Link from 'next/link'

// Types
interface WorkflowStep {
  step_id: string  // Node name (e.g., parse_input, synthesize)
  name: string  // Display name (e.g., "Parsing Input")
  agent_name: string  // Canonical agent name (e.g., llm_parser, llm_analyst)
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

// ==================== INFO TOOLTIPS & HELP CONTENT ====================

// Simple tooltip component that shows on hover
const InfoTooltip = ({ text, children }: { text: string; children?: React.ReactNode }) => {
  const [show, setShow] = useState(false)

  return (
    <div className="relative inline-block">
      <button
        onMouseEnter={() => setShow(true)}
        onMouseLeave={() => setShow(false)}
        onClick={() => setShow(!show)}
        className="ml-1 text-studio-muted hover:text-studio-accent transition-colors"
        type="button"
      >
        {children || <HelpCircle className="w-4 h-4" />}
      </button>
      {show && (
        <div className="absolute z-50 bottom-full left-1/2 transform -translate-x-1/2 mb-2 w-64 p-3 bg-black border border-studio-border rounded-lg shadow-lg text-xs text-white">
          <div className="whitespace-pre-wrap">{text}</div>
          <div className="absolute bottom-0 left-1/2 transform -translate-x-1/2 translate-y-1/2 rotate-45 w-2 h-2 bg-black border-r border-b border-studio-border" />
        </div>
      )}
    </div>
  )
}

// Help modal with full documentation
const HelpModal = ({ isOpen, onClose }: { isOpen: boolean; onClose: () => void }) => {
  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80" onClick={onClose}>
      <div
        className="bg-studio-bg border border-studio-border rounded-lg w-full max-w-3xl max-h-[80vh] overflow-auto m-4"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="sticky top-0 bg-studio-bg border-b border-studio-border p-4 flex items-center justify-between">
          <h2 className="text-lg font-semibold flex items-center gap-2">
            <HelpCircle className="w-5 h-5 text-studio-accent" />
            Understanding Your Credit Analysis
          </h2>
          <button onClick={onClose} className="text-studio-muted hover:text-white p-1">
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="p-6 space-y-6 text-sm">
          {/* Result Metrics */}
          <section>
            <h3 className="text-studio-accent font-semibold mb-3 text-base">📊 Result Metrics</h3>

            <div className="space-y-4">
              <div className="p-3 bg-studio-panel rounded border border-studio-border">
                <h4 className="font-medium text-green-400 mb-1">Risk Level</h4>
                <p className="text-studio-muted">
                  <strong>What it means:</strong> How risky is it to give credit to this company?
                </p>
                <ul className="mt-2 space-y-1 text-studio-muted">
                  <li>• <span className="text-green-400">Low</span> = Very safe, company is financially healthy</li>
                  <li>• <span className="text-yellow-400">Medium</span> = Some concerns, proceed with caution</li>
                  <li>• <span className="text-orange-400">High</span> = Significant risk, may have financial troubles</li>
                  <li>• <span className="text-red-400">Critical</span> = Very dangerous, high chance of default</li>
                </ul>
              </div>

              <div className="p-3 bg-studio-panel rounded border border-studio-border">
                <h4 className="font-medium text-blue-400 mb-1">Credit Score (0-1000)</h4>
                <p className="text-studio-muted">
                  <strong>What it means:</strong> A number rating the company's creditworthiness, like a personal credit score but for businesses.
                </p>
                <ul className="mt-2 space-y-1 text-studio-muted">
                  <li>• <span className="text-green-400">800-1000</span> = Excellent - top tier company</li>
                  <li>• <span className="text-green-400">650-799</span> = Good - reliable company</li>
                  <li>• <span className="text-yellow-400">500-649</span> = Fair - some concerns</li>
                  <li>• <span className="text-orange-400">350-499</span> = Poor - high risk</li>
                  <li>• <span className="text-red-400">0-349</span> = Very Poor - likely to default</li>
                </ul>
              </div>

              <div className="p-3 bg-studio-panel rounded border border-studio-border">
                <h4 className="font-medium text-purple-400 mb-1">Confidence (%)</h4>
                <p className="text-studio-muted">
                  <strong>What it means:</strong> How sure is the AI about its assessment?
                </p>
                <ul className="mt-2 space-y-1 text-studio-muted">
                  <li>• <span className="text-green-400">80-100%</span> = Very confident - lots of data available</li>
                  <li>• <span className="text-yellow-400">60-79%</span> = Moderately confident - some data gaps</li>
                  <li>• <span className="text-orange-400">40-59%</span> = Low confidence - limited data</li>
                  <li>• <span className="text-red-400">0-39%</span> = Very uncertain - take results with caution</li>
                </ul>
              </div>

              <div className="p-3 bg-studio-panel rounded border border-studio-border">
                <h4 className="font-medium text-studio-accent mb-1">Eval Score (%)</h4>
                <p className="text-studio-muted">
                  <strong>What it means:</strong> How well did the AI perform the analysis? This measures the AI's work quality, NOT the company's quality.
                </p>
                <div className="mt-2 p-2 bg-yellow-900/20 border border-yellow-800 rounded">
                  <p className="text-yellow-400 font-medium">⚠️ Why is Eval Score 0%?</p>
                  <p className="text-studio-muted mt-1">
                    This happens when the evaluation step hasn't run yet or there was no evaluation data saved. It does NOT mean the analysis is bad. The Result (Risk Level, Credit Score) is still valid.
                  </p>
                </div>
              </div>
            </div>
          </section>

          {/* Evaluation Metrics */}
          <section>
            <h3 className="text-studio-accent font-semibold mb-3 text-base">🎯 Evaluation Metrics (AI Performance)</h3>
            <p className="text-studio-muted mb-4">
              These scores measure how well the AI performed the analysis. Think of it like grading a student's homework - these grades are about the AI's work, not the company.
            </p>

            <div className="space-y-4">
              <div className="p-3 bg-studio-panel rounded border border-studio-border">
                <h4 className="font-medium text-blue-400 mb-1">Tool Selection Score</h4>
                <p className="text-studio-muted">
                  <strong>What it means:</strong> Did the AI use the right data sources for this company?
                </p>
                <div className="mt-2 bg-black/30 p-2 rounded">
                  <p className="text-studio-muted text-xs mb-2"><strong>How it's calculated:</strong></p>
                  <ul className="space-y-1 text-studio-muted text-xs">
                    <li>• <span className="text-green-400">Precision</span>: Of the tools it used, how many were actually useful?</li>
                    <li>• <span className="text-green-400">Recall</span>: Did it use all the tools it should have?</li>
                    <li>• <span className="text-green-400">F1 Score</span>: The balance between Precision and Recall</li>
                  </ul>
                  <p className="text-studio-muted text-xs mt-2">
                    Example: If analyzing a public company, it should use SEC filings. If it skipped that, recall goes down. If it used irrelevant sources, precision goes down.
                  </p>
                </div>
              </div>

              <div className="p-3 bg-studio-panel rounded border border-studio-border">
                <h4 className="font-medium text-purple-400 mb-1">Data Quality Score</h4>
                <p className="text-studio-muted">
                  <strong>What it means:</strong> How complete and useful was the data the AI gathered?
                </p>
                <div className="mt-2 bg-black/30 p-2 rounded">
                  <p className="text-studio-muted text-xs mb-2"><strong>Factors considered:</strong></p>
                  <ul className="space-y-1 text-studio-muted text-xs">
                    <li>• <span className="text-green-400">Completeness</span>: Did it find financial data, news, legal info?</li>
                    <li>• <span className="text-green-400">Sources Used</span>: How many different sources provided data?</li>
                    <li>• <span className="text-green-400">Data Freshness</span>: Is the data recent and relevant?</li>
                  </ul>
                  <p className="text-studio-muted text-xs mt-2">
                    Low score means: Missing important data, couldn't access some sources, or data is outdated.
                  </p>
                </div>
              </div>

              <div className="p-3 bg-studio-panel rounded border border-studio-border">
                <h4 className="font-medium text-green-400 mb-1">Synthesis Score</h4>
                <p className="text-studio-muted">
                  <strong>What it means:</strong> How well did the AI combine all the data into a final assessment?
                </p>
                <div className="mt-2 bg-black/30 p-2 rounded">
                  <p className="text-studio-muted text-xs mb-2"><strong>Checks if the output has:</strong></p>
                  <ul className="space-y-1 text-studio-muted text-xs">
                    <li>• A clear risk level</li>
                    <li>• A credit score with justification</li>
                    <li>• Specific recommendations</li>
                    <li>• Reasoning that matches the data</li>
                  </ul>
                  <p className="text-studio-muted text-xs mt-2">
                    Low score means: The AI might have skipped some output fields or the reasoning doesn't fully explain the conclusion.
                  </p>
                </div>
              </div>

              <div className="p-3 bg-studio-panel rounded border border-studio-border">
                <h4 className="font-medium text-studio-accent mb-1">Overall Score</h4>
                <p className="text-studio-muted">
                  <strong>What it means:</strong> The average of Tool Selection, Data Quality, and Synthesis scores.
                </p>
                <ul className="mt-2 space-y-1 text-studio-muted">
                  <li>• <span className="text-green-400">80-100%</span> = Excellent analysis - trust the results</li>
                  <li>• <span className="text-yellow-400">60-79%</span> = Good analysis - minor issues</li>
                  <li>• <span className="text-orange-400">40-59%</span> = Fair analysis - review manually</li>
                  <li>• <span className="text-red-400">0-39%</span> = Poor analysis - may need to re-run or check manually</li>
                </ul>
              </div>
            </div>
          </section>

          {/* FAQ */}
          <section>
            <h3 className="text-studio-accent font-semibold mb-3 text-base">❓ Common Questions</h3>

            <div className="space-y-3">
              <div className="p-3 bg-studio-panel rounded border border-studio-border">
                <h4 className="font-medium text-white mb-1">Why is my Eval Score 0%?</h4>
                <p className="text-studio-muted">
                  The evaluation step may not have completed or saved. This is a technical issue and doesn't affect the actual credit analysis. Your Risk Level and Credit Score are still valid.
                </p>
              </div>

              <div className="p-3 bg-studio-panel rounded border border-studio-border">
                <h4 className="font-medium text-white mb-1">Why is Tool Selection low?</h4>
                <p className="text-studio-muted">
                  Either: (1) Some data sources were unavailable, (2) The company type didn't match available tools (e.g., private company with no SEC filings), or (3) The AI chose unnecessary tools.
                </p>
              </div>

              <div className="p-3 bg-studio-panel rounded border border-studio-border">
                <h4 className="font-medium text-white mb-1">Can I trust a low-confidence result?</h4>
                <p className="text-studio-muted">
                  Use it as a starting point, but verify with your own research. Low confidence usually means limited public data is available for that company.
                </p>
              </div>

              <div className="p-3 bg-studio-panel rounded border border-studio-border">
                <h4 className="font-medium text-white mb-1">What data sources are used?</h4>
                <p className="text-studio-muted">
                  SEC filings, Finnhub financial data, Tavily web search, CourtListener legal records, and more. Check the Traces tab to see exactly which sources were used.
                </p>
              </div>
            </div>
          </section>
        </div>
      </div>
    </div>
  )
}

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
  const [historySearch, setHistorySearch] = useState('')

  // Help modal state
  const [showHelpModal, setShowHelpModal] = useState(false)

  // Run details modal state
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null)
  const [runDetails, setRunDetails] = useState<any>(null)
  const [runDetailsLoading, setRunDetailsLoading] = useState(false)

  // Real-time LLM streaming state - per section
  const [streamingTextByStep, setStreamingTextByStep] = useState<Record<string, string>>({})
  const [currentThinkingNode, setCurrentThinkingNode] = useState<string>('')
  const [stepDescription, setStepDescription] = useState<string>('')
  const [stepDescriptionByStep, setStepDescriptionByStep] = useState<Record<string, string>>({})
  const [progressPercent, setProgressPercent] = useState<number>(0)

  const wsRef = useRef<WebSocket | null>(null)
  const logsEndRef = useRef<HTMLDivElement>(null)
  const runningStepRef = useRef<HTMLDivElement>(null)
  const resultSectionRef = useRef<HTMLDivElement>(null)
  const pipelineScrollRef = useRef<HTMLDivElement>(null)

  // Auto-scroll logs
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [logs])

  // Auto-scroll to running step when current_step changes
  useEffect(() => {
    if (workflow?.current_step && workflow?.steps) {
      // Find the running step and scroll to it by ID
      const runningStep = workflow.steps.find(s => s.status === 'running')
      if (runningStep) {
        const timer = setTimeout(() => {
          const stepElement = document.querySelector(`[data-step-id="${runningStep.step_id}"]`) as HTMLElement
          if (stepElement) {
            stepElement.scrollIntoView({ behavior: 'smooth', block: 'center' })
          }
        }, 150)
        return () => clearTimeout(timer)
      }
    }
  }, [workflow?.current_step, workflow?.steps])

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
  const fetchHistoricalRuns = async (search?: string) => {
    try {
      const params = new URLSearchParams({ limit: '50' })
      if (search) {
        params.append('search', search)
      }
      const response = await fetch(`${API_URL}/logs/runs/history?${params}`)
      if (response.ok) {
        const data = await response.json()
        if (data.runs) {
          setHistoricalRuns(data.runs.map((r: any) => ({
            run_id: r.run_id,
            company_name: r.company_name,
            status: r.status || 'completed',
            risk_level: r.risk_level,
            credit_score: r.credit_score,
            evaluation_score: r.overall_score,
            started_at: r.timestamp || new Date().toISOString(),
            completed_at: r.timestamp || new Date().toISOString(),
            duration_ms: r.duration_ms,
          })))
        }
      }
    } catch (e) {
      console.error('Failed to fetch historical runs:', e)
    }
  }

  // Fetch run details when clicking on a row
  const fetchRunDetails = async (runId: string) => {
    setSelectedRunId(runId)
    setRunDetailsLoading(true)
    setRunDetails(null)
    try {
      // Fetch run details and coalition evaluation in parallel
      const [detailsResponse, coalitionResponse] = await Promise.all([
        fetch(`${API_URL}/logs/runs/${runId}/details`),
        fetch(`${API_URL}/evaluate/coalition/${runId}`).catch(() => null)
      ])

      let data: any = {}
      if (detailsResponse.ok) {
        data = await detailsResponse.json()
      }

      // Add coalition evaluation if available
      if (coalitionResponse?.ok) {
        try {
          data.coalition = await coalitionResponse.json()
        } catch (e) {
          console.warn('Failed to parse coalition response')
        }
      }

      setRunDetails(data)
    } catch (e) {
      console.error('Failed to fetch run details:', e)
    } finally {
      setRunDetailsLoading(false)
    }
  }

  // Close run details modal
  const closeRunDetails = () => {
    setSelectedRunId(null)
    setRunDetails(null)
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
          // Update step description and progress
          if (message.data.description) {
            setStepDescription(message.data.description)
            // Store description per step for reliable display
            setStepDescriptionByStep(prev => ({ ...prev, [step.step_id]: message.data.description }))
          }
          if (message.data.progress_percent !== undefined) {
            setProgressPercent(message.data.progress_percent)
          }
          if (step.status === 'running') {
            addLog('step', `Starting: ${step.name}`, step)
            setExpandedSteps(prev => new Set([...prev, step.step_id]))
            // Initialize streaming text for this step
            setStreamingTextByStep(prev => ({ ...prev, [step.step_id]: '' }))
            setCurrentThinkingNode(step.step_id)
            // Auto-scroll to this specific step by ID (not by status, which may not have updated)
            const stepId = step.step_id
            const scrollToStep = () => {
              // Find the specific step element by its ID
              const stepElement = document.querySelector(`[data-step-id="${stepId}"]`) as HTMLElement
              if (stepElement) {
                stepElement.scrollIntoView({ behavior: 'smooth', block: 'center' })
              }
            }
            // Scroll immediately and also after delays to ensure visibility
            scrollToStep()
            setTimeout(scrollToStep, 100)
            setTimeout(scrollToStep, 300)
          } else if (step.status === 'completed') {
            addLog('step', `Completed: ${step.name} (${step.duration_ms?.toFixed(0)}ms)`, step)
            // Keep the streaming text for this step (don't clear)
            setCurrentThinkingNode('')
          } else if (step.status === 'failed') {
            addLog('error', `Failed: ${step.name} - ${step.error}`, step)
            setCurrentThinkingNode('')
          }
          break

        case 'llm_token':
          // Real-time LLM token streaming - store per section
          const node = message.data.node || currentThinkingNode
          if (node) {
            setStreamingTextByStep(prev => ({
              ...prev,
              [node]: (prev[node] || '') + (message.data.content || '')
            }))
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
          setCurrentThinkingNode('')
          setStepDescription('')
          setProgressPercent(100)
          addLog('success', `Workflow completed! Risk: ${message.data.result?.risk_level}, Score: ${message.data.result?.credit_score}`)
          // Switch to result tab and scroll to top after a short delay
          setDetailTab('result')
          setTimeout(() => {
            resultSectionRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
            // Also scroll the main content area to top
            window.scrollTo({ top: 0, behavior: 'smooth' })
          }, 300)
          break

        case 'workflow_failed':
          setWorkflow(message.data.status)
          setIsRunning(false)
          setCurrentThinkingNode('')
          setStepDescription('')
          setError(message.data.error)
          addLog('error', `Workflow failed: ${message.data.error}`)
          break

        case 'api_error':
          // Display API errors (rate limits, quota exceeded, etc.) prominently
          const apiError = message.data
          const errorIcon = apiError.error_type === 'rate_limit' ? 'Rate Limit' :
                           apiError.error_type === 'quota_exceeded' ? 'Quota Exceeded' : 'API Error'
          addLog('warning', `${errorIcon}: ${apiError.message}`)
          // Show as a warning, not a fatal error - workflow may continue with fallbacks
          setError(`${errorIcon}: ${apiError.message}`)
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

  // Check for active runs on mount (for background run resumption)
  useEffect(() => {
    const checkActiveRuns = async () => {
      try {
        const response = await fetch(`${API_URL}/runs/active`)
        if (response.ok) {
          const data = await response.json()
          if (data.runs?.length > 0) {
            // Found an active run - reconnect
            const activeRun = data.runs[0]
            console.log('Found active run, reconnecting:', activeRun.run_id)

            const steps = activeRun.steps || []

            // Restore workflow state
            setWorkflow({
              run_id: activeRun.run_id,
              company_name: activeRun.company_name,
              status: activeRun.status,
              current_step: activeRun.current_step,
              steps: steps,
              started_at: activeRun.started_at,
            })
            setCompanyName(activeRun.company_name)
            setIsRunning(true)
            setActiveTab('single')

            // Calculate progress from completed steps
            const completedSteps = steps.filter((s: any) => s.status === 'completed').length
            const totalSteps = steps.length || 8  // Default to 8 steps
            const progress = Math.round((completedSteps / totalSteps) * 100)
            setProgressPercent(progress)

            // Expand completed steps
            const completedIds = steps
              .filter((s: any) => s.status === 'completed' || s.status === 'running')
              .map((s: any) => s.step_id)
            setExpandedSteps(new Set(completedIds))

            // Find current running step for description
            const runningStep = steps.find((s: any) => s.status === 'running')
            if (runningStep) {
              setCurrentThinkingNode(runningStep.step_id)
            }

            // Connect WebSocket to resume streaming
            connectWebSocket(activeRun.run_id)

            // Log will be added when WebSocket connects
            setLogs([{
              timestamp: new Date().toISOString(),
              type: 'system',
              message: `Resumed active analysis for ${activeRun.company_name}`
            }])
          }
        }
      } catch (e) {
        console.error('Failed to check for active runs:', e)
      }
    }
    checkActiveRuns()
  }, [connectWebSocket])

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
            <Link
              href="/settings"
              className="flex items-center gap-2 px-3 py-1.5 text-sm bg-studio-panel hover:bg-studio-border rounded transition-colors"
            >
              <SlidersHorizontal className="w-4 h-4" />
              Settings
            </Link>
            <a
              href="/erd"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 px-3 py-1.5 text-sm bg-studio-panel hover:bg-studio-border rounded transition-colors"
            >
              <Database className="w-4 h-4" />
              ERD
            </a>
            <button
              onClick={() => setShowHelpModal(true)}
              className="flex items-center gap-2 px-3 py-1.5 text-sm bg-yellow-600 hover:bg-yellow-700 rounded transition-colors"
            >
              <HelpCircle className="w-4 h-4" />
              Help
            </button>
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
                  <div className={`mt-3 p-2 rounded text-sm ${
                    error.includes('Rate Limit') || error.includes('Quota Exceeded')
                      ? 'bg-orange-900/30 border border-orange-800 text-orange-400'
                      : 'bg-red-900/30 border border-red-800 text-red-400'
                  }`}>
                    {error}
                  </div>
                )}
              </div>

              {/* Workflow Pipeline */}
              <div ref={pipelineScrollRef} className="flex-1 overflow-auto p-4">
                <h3 className="text-sm font-medium text-studio-muted mb-3 flex items-center gap-2">
                  <Layers className="w-4 h-4" />
                  Workflow Pipeline
                </h3>

                {workflow?.steps && workflow.steps.length > 0 ? (
                  <div className="space-y-2">
                    {workflow.steps.map((step, idx) => (
                      <div
                        key={step.step_id}
                        data-step-status={step.status}
                        data-step-id={step.step_id}
                        ref={step.status === 'running' ? runningStepRef : null}
                      >
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
                              <div className="flex flex-col">
                                <span className="text-sm font-medium">{step.name}</span>
                                <span className="text-xs text-studio-muted font-mono">{step.agent_name}</span>
                              </div>
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
                          {/* Show description when step is running */}
                          {step.status === 'running' && stepDescriptionByStep[step.step_id] && (
                            <div className="mt-2 ml-6 flex items-center gap-2">
                              <div className="flex gap-1">
                                <span className="w-1.5 h-1.5 rounded-full bg-studio-accent animate-pulse" />
                                <span className="w-1.5 h-1.5 rounded-full bg-studio-accent animate-pulse" style={{ animationDelay: '0.2s' }} />
                                <span className="w-1.5 h-1.5 rounded-full bg-studio-accent animate-pulse" style={{ animationDelay: '0.4s' }} />
                              </div>
                              <span className="text-xs text-studio-accent italic">{stepDescriptionByStep[step.step_id]}</span>
                            </div>
                          )}
                          {step.error && (
                            <div className="text-xs text-red-400 mt-2 ml-6">{step.error}</div>
                          )}
                        </div>

                        {/* Step Details with Section-wise AI Thinking */}
                        {expandedSteps.has(step.step_id) && step.status !== 'pending' && (
                          <div className="border border-t-0 border-studio-border rounded-b bg-black/50 p-3 max-h-96 overflow-auto">
                            {/* AI Thinking for this section - show when running or has streaming content */}
                            {(step.status === 'running' || streamingTextByStep[step.step_id]) && (
                              <div className="mb-3 bg-studio-panel/50 border border-studio-accent/30 rounded p-3">
                                <div className="flex items-center gap-2 mb-2">
                                  <Cpu className={`w-4 h-4 text-studio-accent ${step.status === 'running' ? 'animate-pulse' : ''}`} />
                                  <span className="text-xs text-studio-accent font-medium">
                                    AI Thinking
                                  </span>
                                </div>
                                <div className="font-mono text-xs text-studio-text whitespace-pre-wrap max-h-32 overflow-auto bg-black/30 rounded p-2">
                                  {streamingTextByStep[step.step_id] || 'Processing...'}
                                  {step.status === 'running' && (
                                    <span className="inline-block w-1.5 h-3 bg-studio-accent animate-pulse ml-0.5" />
                                  )}
                                </div>
                              </div>
                            )}

                            {step.output_summary && (
                              <div className="mb-2 p-2 bg-studio-accent/10 rounded border border-studio-accent/30">
                                <div className="text-xs text-studio-accent font-medium">Summary</div>
                                <div className="text-sm text-white">{step.output_summary}</div>
                              </div>
                            )}
                            {step.output_data && Object.keys(step.output_data).length > 0 && (
                              <div className="space-y-1">
                                {Object.entries(step.output_data).map(([key, value]) => (
                                  <div key={key} className="text-xs">
                                    <span className="text-yellow-400 font-mono">{key}:</span>{' '}
                                    <pre className="text-green-300 whitespace-pre-wrap break-all inline">
                                      {typeof value === 'object' ? JSON.stringify(value, null, 2) : String(value)}
                                    </pre>
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
            <div className="flex-1 flex flex-col min-h-0">
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
              <div className="flex-1 overflow-auto">
                {/* Progress Bar - Shows during analysis (sticky inside scroll container) */}
                {isRunning && workflow && (
                  <div className="sticky top-0 z-10 px-4 py-3 border-b border-studio-border bg-studio-panel/95 backdrop-blur-sm shadow-lg">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <Activity className="w-4 h-4 text-studio-accent animate-pulse" />
                        <span className="text-sm font-medium text-studio-text">
                          Analyzing {workflow.company_name}
                        </span>
                        {/* Step indicator */}
                        {workflow.steps && workflow.steps.length > 0 && (
                          <span className="text-xs bg-studio-accent/20 text-studio-accent px-2 py-0.5 rounded-full">
                            Step {(workflow.steps.findIndex(s => s.status === 'running') + 1) || workflow.steps.filter(s => s.status === 'completed').length} of {workflow.steps.length}
                          </span>
                        )}
                      </div>
                      <span className="text-sm text-studio-accent font-mono">{progressPercent}%</span>
                    </div>
                    <div className="h-2 bg-studio-border rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-studio-accent to-blue-400 transition-all duration-500 ease-out"
                        style={{ width: `${progressPercent}%` }}
                      />
                    </div>
                    {/* Current step name */}
                    {workflow.current_step && (
                      <div className="text-xs text-studio-accent mt-2 font-medium">
                        {workflow.current_step}
                      </div>
                    )}
                    {stepDescription && (
                      <div className="text-xs text-studio-muted mt-1 italic">
                        {stepDescription}
                      </div>
                    )}
                  </div>
                )}

                {/* Result Tab */}
                <div className="p-4">
                {detailTab === 'result' && workflow?.result && (
                  <div ref={resultSectionRef} className="space-y-4">
                    <div className="flex items-center justify-between">
                      <h2 className="text-lg font-semibold flex items-center gap-2">
                        <Building2 className="w-5 h-5" />
                        Assessment: {workflow.company_name}
                      </h2>
                      {workflow.run_id && (
                        <div className="flex items-center gap-2 text-sm bg-studio-panel px-3 py-1.5 rounded border border-studio-border">
                          <span className="text-studio-muted">Run ID:</span>
                          <code className="text-studio-accent font-mono">{workflow.run_id}</code>
                          <button
                            onClick={() => {
                              navigator.clipboard.writeText(workflow.run_id)
                            }}
                            className="ml-1 px-2 py-0.5 bg-studio-accent/20 hover:bg-studio-accent/30 text-studio-accent rounded text-xs"
                          >
                            Copy
                          </button>
                        </div>
                      )}
                    </div>

                    <div className="grid grid-cols-4 gap-4">
                      <div className={`p-4 rounded-lg border ${getRiskColor(workflow.result.risk_level)}`}>
                        <div className="text-sm text-studio-muted mb-1 flex items-center">
                          Risk Level
                          <InfoTooltip text="How risky is it to give credit to this company?\n\n• Low = Very safe\n• Medium = Some concerns\n• High = Significant risk\n• Critical = Very dangerous" />
                        </div>
                        <div className="text-2xl font-bold capitalize">{workflow.result.risk_level}</div>
                      </div>
                      <div className="p-4 rounded-lg border border-studio-border bg-studio-panel">
                        <div className="text-sm text-studio-muted mb-1 flex items-center">
                          Credit Score
                          <InfoTooltip text="Business credit score (0-1000).\n\n• 800-1000 = Excellent\n• 650-799 = Good\n• 500-649 = Fair\n• 350-499 = Poor\n• 0-349 = Very Poor" />
                        </div>
                        <div className="text-2xl font-bold text-studio-accent">{workflow.result.credit_score}</div>
                      </div>
                      <div className="p-4 rounded-lg border border-studio-border bg-studio-panel">
                        <div className="text-sm text-studio-muted mb-1 flex items-center">
                          Confidence
                          <InfoTooltip text="How confident is the AI about this result?\n\n• 80-100% = Very confident\n• 60-79% = Moderate\n• 40-59% = Low\n• 0-39% = Very uncertain\n\nLow confidence = limited data available." />
                        </div>
                        <div className="text-2xl font-bold">{(workflow.result.confidence * 100).toFixed(0)}%</div>
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
                            <div className="text-sm text-studio-muted mb-1 flex items-center">
                              Tool Selection
                              <InfoTooltip text="Did the AI use the right data sources?\n\nCalculated using:\n• Precision: Were the tools useful?\n• Recall: Did it use all needed tools?\n• F1: Balance of both\n\nLow score = wrong or missing data sources" />
                            </div>
                            <div className="text-2xl font-bold text-blue-400">
                              {formatScore(evalSummary?.tool_selection_score ?? backendEvaluation?.tool_selection_score)}
                            </div>
                          </div>
                          <div className="p-4 rounded-lg border border-studio-border bg-studio-panel">
                            <div className="text-sm text-studio-muted mb-1 flex items-center">
                              Data Quality
                              <InfoTooltip text="Was the gathered data good enough?\n\nChecks:\n• Completeness of data\n• Number of sources used\n• Data freshness\n\nLow score = missing data or outdated info" />
                            </div>
                            <div className="text-2xl font-bold text-purple-400">
                              {formatScore(evalSummary?.data_quality_score ?? backendEvaluation?.data_quality_score)}
                            </div>
                          </div>
                          <div className="p-4 rounded-lg border border-studio-border bg-studio-panel">
                            <div className="text-sm text-studio-muted mb-1 flex items-center">
                              Synthesis
                              <InfoTooltip text="Did the AI create a good final report?\n\nChecks if output has:\n• Clear risk level\n• Justified credit score\n• Recommendations\n• Proper reasoning\n\nLow score = incomplete report" />
                            </div>
                            <div className="text-2xl font-bold text-green-400">
                              {formatScore(evalSummary?.synthesis_score ?? backendEvaluation?.synthesis_score)}
                            </div>
                          </div>
                          <div className="p-4 rounded-lg border border-studio-border bg-studio-panel">
                            <div className="text-sm text-studio-muted mb-1 flex items-center">
                              Overall
                              <InfoTooltip text="Average of all evaluation scores.\n\n• 80-100% = Excellent analysis\n• 60-79% = Good analysis\n• 40-59% = Fair - review manually\n• 0-39% = Poor - may need re-run" />
                            </div>
                            <div className="text-2xl font-bold text-studio-accent">
                              {formatScore(evalSummary?.overall_score ?? backendEvaluation?.overall_score)}
                            </div>
                          </div>
                        </div>

                        {/* Detailed Breakdowns */}
                        <div className="grid grid-cols-3 gap-4">
                          {(evalSummary?.tool_selection || backendEvaluation?.tool_selection) && (
                            <div className="p-4 rounded-lg border border-studio-border bg-studio-panel">
                              <div className="text-sm font-medium mb-2 text-blue-400 flex items-center">
                                Tool Selection Details
                                <InfoTooltip text="Shows how the score was calculated:\n\n• Precision: Of tools used, how many were useful?\n• Recall: Did it use all the tools it should?\n• F1: The balance between precision and recall" />
                              </div>
                              <div className="text-xs space-y-1 text-studio-muted">
                                {(() => {
                                  const ts = evalSummary?.tool_selection || backendEvaluation?.tool_selection || {}
                                  return (
                                    <>
                                      <p><span className="text-green-400">Precision:</span> {((ts.precision || 0) * 100).toFixed(0)}% <span className="text-studio-muted">(useful tools / total used)</span></p>
                                      <p><span className="text-green-400">Recall:</span> {((ts.recall || 0) * 100).toFixed(0)}% <span className="text-studio-muted">(used tools / needed tools)</span></p>
                                      <p><span className="text-green-400">F1:</span> {((ts.f1_score || 0) * 100).toFixed(0)}% <span className="text-studio-muted">(balanced score)</span></p>
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
                              <div className="text-sm font-medium mb-2 text-purple-400 flex items-center">
                                Data Quality Details
                                <InfoTooltip text="Shows data gathering results:\n\n• Sources: Which data sources provided info\n• Completeness: % of expected data fields found\n\nMore sources + higher completeness = better score" />
                              </div>
                              <div className="text-xs space-y-1 text-studio-muted">
                                {(() => {
                                  const dq = evalSummary?.data_quality || backendEvaluation?.data_quality || {}
                                  return (
                                    <>
                                      <p><span className="text-green-400">Sources:</span> {dq.sources_used?.join(', ') || 'N/A'}</p>
                                      <p><span className="text-green-400">Completeness:</span> {((dq.completeness || 0) * 100).toFixed(0)}% <span className="text-studio-muted">(data fields found)</span></p>
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
                              <div className="text-sm font-medium mb-2 text-green-400 flex items-center">
                                Synthesis Details
                                <InfoTooltip text="Shows final report quality:\n\n• Score: Overall synthesis quality\n• Fields: Which required output fields are present\n\nAll fields present = higher score" />
                              </div>
                              <div className="text-xs space-y-1 text-studio-muted">
                                {(() => {
                                  const syn = evalSummary?.synthesis || backendEvaluation?.synthesis || {}
                                  return (
                                    <>
                                      <p><span className="text-green-400">Score:</span> {((syn.score || 0) * 100).toFixed(0)}% <span className="text-studio-muted">(report quality)</span></p>
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
                        <AlertTriangle className="w-8 h-8 mx-auto mb-3 text-yellow-500" />
                        <p className="font-medium text-white mb-2">No evaluation data available</p>
                        <p className="text-sm max-w-md mx-auto">
                          Run an analysis first. If Eval Score shows 0% after running, it means the evaluation step hasn't saved data yet - this is normal and doesn't affect your credit analysis results.
                        </p>
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
                                log.type === 'warning' ? 'text-orange-400' :
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
                              <span className="text-[10px] text-studio-muted font-mono">{step.agent_name}</span>
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
                            <th className="text-left p-3 text-studio-muted">Agent</th>
                            <th className="text-left p-3 text-studio-muted">Status</th>
                            <th className="text-left p-3 text-studio-muted">Duration</th>
                            <th className="text-left p-3 text-studio-muted">Output</th>
                          </tr>
                        </thead>
                        <tbody>
                          {workflow.steps.map((step) => (
                            <tr key={step.step_id} className="border-t border-studio-border">
                              <td className="p-3 font-medium">{step.name}</td>
                              <td className="p-3 font-mono text-xs text-studio-muted">{step.agent_name}</td>
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
                              <td className="p-3 text-studio-muted text-xs whitespace-pre-wrap break-all">
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
                        <th className="text-left p-3 text-studio-muted">Run ID</th>
                        <th className="text-left p-3 text-studio-muted">Company</th>
                        <th className="text-left p-3 text-studio-muted">Status</th>
                        <th className="text-left p-3 text-studio-muted">Risk</th>
                        <th className="text-left p-3 text-studio-muted">Score</th>
                        <th className="text-left p-3 text-studio-muted">Confidence</th>
                      </tr>
                    </thead>
                    <tbody>
                      {batchResults.map((result) => (
                        <tr key={result.run_id} className="border-t border-studio-border hover:bg-studio-panel/30">
                          <td className="p-3">
                            <div className="flex items-center gap-1">
                              <code className="text-xs font-mono text-studio-accent bg-studio-panel px-1.5 py-0.5 rounded break-all" title={result.run_id}>
                                {result.run_id}
                              </code>
                              <button
                                onClick={() => navigator.clipboard.writeText(result.run_id)}
                                className="text-xs text-studio-muted hover:text-studio-accent px-1"
                                title="Copy full Run ID"
                              >
                                Copy
                              </button>
                            </div>
                          </td>
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
                          <th className="text-left p-3 text-studio-muted">Run ID</th>
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
                            <td className="p-3">
                              <div className="flex items-center gap-1">
                                <code className="text-xs font-mono text-studio-accent bg-studio-panel px-1.5 py-0.5 rounded break-all" title={run.run_id}>
                                  {run.run_id}
                                </code>
                                <button
                                  onClick={() => navigator.clipboard.writeText(run.run_id)}
                                  className="text-xs text-studio-muted hover:text-studio-accent px-1"
                                  title="Copy full Run ID"
                                >
                                  Copy
                                </button>
                              </div>
                            </td>
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
                              {new Date(run.completed_at || run.started_at).toLocaleString()}
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
                    onClick={() => fetchHistoricalRuns(historySearch)}
                    className="text-sm text-studio-accent hover:underline flex items-center gap-1"
                  >
                    <RefreshCw className="w-3 h-3" />
                    Refresh
                  </button>
                </div>

                {/* Search input */}
                <div className="flex gap-2 mb-4">
                  <input
                    type="text"
                    value={historySearch}
                    onChange={(e) => setHistorySearch(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && fetchHistoricalRuns(historySearch)}
                    placeholder="Search by run ID or company name..."
                    className="flex-1 bg-studio-panel border border-studio-border rounded px-3 py-2 text-sm focus:outline-none focus:border-studio-accent"
                  />
                  <button
                    onClick={() => fetchHistoricalRuns(historySearch)}
                    className="px-4 py-2 bg-studio-accent hover:bg-blue-600 rounded text-sm font-medium transition-colors"
                  >
                    Search
                  </button>
                  {historySearch && (
                    <button
                      onClick={() => {
                        setHistorySearch('')
                        fetchHistoricalRuns()
                      }}
                      className="px-3 py-2 text-studio-muted hover:text-white transition-colors"
                      title="Clear search"
                    >
                      Clear
                    </button>
                  )}
                </div>

                {historicalRuns.length > 0 ? (
                  <div className="rounded-lg border border-studio-border overflow-hidden">
                    <table className="w-full text-sm">
                      <thead className="bg-studio-panel">
                        <tr>
                          <th className="text-left p-3 text-studio-muted">Run ID</th>
                          <th className="text-left p-3 text-studio-muted">Company</th>
                          <th className="text-left p-3 text-studio-muted">Risk</th>
                          <th className="text-left p-3 text-studio-muted">Score</th>
                          <th className="text-left p-3 text-studio-muted">Time</th>
                          <th className="text-left p-3 text-studio-muted">Actions</th>
                        </tr>
                      </thead>
                      <tbody>
                        {historicalRuns.map((run) => (
                          <tr
                            key={run.run_id}
                            className="border-t border-studio-border hover:bg-studio-panel/50 cursor-pointer transition-colors"
                            onClick={() => fetchRunDetails(run.run_id)}
                          >
                            <td className="p-3">
                              <div className="flex items-center gap-1">
                                <code className="text-xs font-mono text-studio-accent bg-studio-panel px-1.5 py-0.5 rounded break-all" title={run.run_id}>
                                  {run.run_id.slice(0, 8)}...
                                </code>
                                <button
                                  onClick={(e) => {
                                    e.stopPropagation()
                                    navigator.clipboard.writeText(run.run_id)
                                  }}
                                  className="text-xs text-studio-muted hover:text-studio-accent px-1"
                                  title="Copy full Run ID"
                                >
                                  Copy
                                </button>
                              </div>
                            </td>
                            <td className="p-3 font-medium">{run.company_name}</td>
                            <td className="p-3">
                              <span className={`px-2 py-1 rounded text-xs ${getRiskColor(run.risk_level || '')}`}>
                                {run.risk_level || 'N/A'}
                              </span>
                            </td>
                            <td className="p-3">{run.credit_score || 'N/A'}</td>
                            <td className="p-3 text-studio-muted">
                              {new Date(run.completed_at || run.started_at).toLocaleString()}
                            </td>
                            <td className="p-3">
                              <button
                                onClick={(e) => {
                                  e.stopPropagation()
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

      {/* Run Details Modal */}
      {selectedRunId && (
        <RunDetailsModal
          runId={selectedRunId}
          details={runDetails}
          loading={runDetailsLoading}
          onClose={closeRunDetails}
        />
      )}

      {/* Help Modal */}
      <HelpModal isOpen={showHelpModal} onClose={() => setShowHelpModal(false)} />
    </div>
  )
}

// Run Details Modal Component
function RunDetailsModal({
  runId,
  details,
  loading,
  onClose,
}: {
  runId: string
  details: any
  loading: boolean
  onClose: () => void
}) {
  const [activeSection, setActiveSection] = useState<'summary' | 'evaluation' | 'llm' | 'nodes' | 'assessment' | 'coalition' | 'state'>('summary')
  const [stateDump, setStateDump] = useState<any>(null)
  const [stateLoading, setStateLoading] = useState(false)

  const formatDuration = (ms: number) => {
    if (ms < 1000) return `${ms.toFixed(0)}ms`
    return `${(ms / 1000).toFixed(2)}s`
  }

  const formatCost = (cost: number) => {
    if (cost < 0.01) return `$${cost.toFixed(6)}`
    return `$${cost.toFixed(4)}`
  }

  return (
    <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4" onClick={onClose}>
      <div
        className="bg-studio-bg border border-studio-border rounded-lg w-full max-w-5xl max-h-[90vh] overflow-hidden flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="p-4 border-b border-studio-border bg-studio-panel flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold flex items-center gap-2">
              <Activity className="w-5 h-5 text-studio-accent" />
              Run Details
            </h2>
            <div className="flex items-center gap-2 mt-1">
              <code className="text-xs font-mono text-studio-muted bg-black/30 px-2 py-0.5 rounded">
                {runId}
              </code>
              {details?.summary?.company_name && (
                <span className="text-sm text-studio-accent">{details.summary.company_name}</span>
              )}
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-studio-border rounded transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Loading state */}
        {loading && (
          <div className="flex-1 flex items-center justify-center py-20">
            <div className="flex items-center gap-3 text-studio-muted">
              <Loader2 className="w-6 h-6 animate-spin" />
              Loading run details...
            </div>
          </div>
        )}

        {/* Content */}
        {!loading && details && (
          <>
            {/* Navigation tabs */}
            <div className="border-b border-studio-border bg-studio-panel px-4">
              <nav className="flex gap-1">
                {[
                  { id: 'summary', label: 'Summary', icon: FileText },
                  { id: 'evaluation', label: 'Evaluation', icon: BarChart3 },
                  { id: 'coalition', label: 'Coalition', icon: CheckCircle },
                  { id: 'llm', label: 'LLM Calls', icon: Cpu },
                  { id: 'nodes', label: 'Nodes & Agents', icon: GitBranch },
                  { id: 'assessment', label: 'Assessment', icon: Shield },
                  { id: 'state', label: 'State', icon: Database },
                ].map((tab) => (
                  <button
                    key={tab.id}
                    onClick={() => setActiveSection(tab.id as any)}
                    className={`flex items-center gap-2 px-3 py-2 text-sm font-medium border-b-2 transition-colors ${
                      activeSection === tab.id
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

            {/* Tab Content */}
            <div className="flex-1 overflow-y-auto p-4">
              {/* Summary Tab */}
              {activeSection === 'summary' && (
                <div className="space-y-4">
                  {/* Quick Stats */}
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="p-4 rounded-lg border border-studio-border bg-studio-panel">
                      <div className="text-xs text-studio-muted mb-1">Status</div>
                      <div className={`font-semibold ${
                        details.summary?.status === 'completed' ? 'text-green-400' : 'text-yellow-400'
                      }`}>
                        {details.summary?.status || 'Unknown'}
                      </div>
                    </div>
                    <div className="p-4 rounded-lg border border-studio-border bg-studio-panel">
                      <div className="text-xs text-studio-muted mb-1">Risk Level</div>
                      <div className={`font-semibold ${
                        details.summary?.risk_level === 'low' ? 'text-green-400' :
                        details.summary?.risk_level === 'medium' ? 'text-yellow-400' : 'text-red-400'
                      }`}>
                        {details.summary?.risk_level || 'N/A'}
                      </div>
                    </div>
                    <div className="p-4 rounded-lg border border-studio-border bg-studio-panel">
                      <div className="text-xs text-studio-muted mb-1">Credit Score</div>
                      <div className="font-semibold text-studio-accent">
                        {details.summary?.credit_score || 'N/A'}
                      </div>
                    </div>
                    <div className="p-4 rounded-lg border border-studio-border bg-studio-panel">
                      <div className="text-xs text-studio-muted mb-1">Duration</div>
                      <div className="font-semibold">
                        {details.summary?.duration_ms ? formatDuration(details.summary.duration_ms) : 'N/A'}
                      </div>
                    </div>
                  </div>

                  {/* Decision */}
                  {details.summary?.final_decision && (
                    <div className="p-4 rounded-lg border border-studio-border bg-studio-panel">
                      <div className="text-sm text-studio-muted mb-2">Final Decision</div>
                      <div className={`text-lg font-bold ${
                        details.summary.final_decision === 'APPROVED' ? 'text-green-400' :
                        details.summary.final_decision === 'REJECTED' ? 'text-red-400' : 'text-yellow-400'
                      }`}>
                        {details.summary.final_decision}
                      </div>
                      {details.summary.decision_reasoning && (
                        <p className="text-sm text-studio-muted mt-2">{details.summary.decision_reasoning}</p>
                      )}
                    </div>
                  )}

                  {/* Reasoning */}
                  {details.summary?.reasoning && (
                    <div className="p-4 rounded-lg border border-studio-border bg-studio-panel">
                      <div className="text-sm text-studio-muted mb-2">Analysis Reasoning</div>
                      <p className="text-sm whitespace-pre-wrap">{details.summary.reasoning}</p>
                    </div>
                  )}

                  {/* Cost & Tokens */}
                  <div className="grid grid-cols-3 gap-4">
                    <div className="p-4 rounded-lg border border-studio-border bg-studio-panel">
                      <div className="text-xs text-studio-muted mb-1">Total Tokens</div>
                      <div className="font-semibold">{details.summary?.total_tokens?.toLocaleString() || 'N/A'}</div>
                    </div>
                    <div className="p-4 rounded-lg border border-studio-border bg-studio-panel">
                      <div className="text-xs text-studio-muted mb-1">Total Cost</div>
                      <div className="font-semibold">{details.summary?.total_cost ? formatCost(details.summary.total_cost) : 'N/A'}</div>
                    </div>
                    <div className="p-4 rounded-lg border border-studio-border bg-studio-panel">
                      <div className="text-xs text-studio-muted mb-1">LLM Calls</div>
                      <div className="font-semibold">{details.summary?.llm_calls_count || details.llm_calls?.length || 'N/A'}</div>
                    </div>
                  </div>

                </div>
              )}

              {/* Evaluation Tab */}
              {activeSection === 'evaluation' && (
                <div className="space-y-4">
                  {/* Scores Grid */}
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="p-4 rounded-lg border border-studio-border bg-studio-panel">
                      <div className="text-xs text-studio-muted mb-1">Overall Score</div>
                      <div className="text-2xl font-bold text-studio-accent">
                        {details.summary?.overall_score ? (details.summary.overall_score * 100).toFixed(1) + '%' : 'N/A'}
                      </div>
                    </div>
                    <div className="p-4 rounded-lg border border-studio-border bg-studio-panel">
                      <div className="text-xs text-studio-muted mb-1">Tool Selection</div>
                      <div className="text-2xl font-bold">
                        {details.summary?.tool_selection_score ? (details.summary.tool_selection_score * 100).toFixed(1) + '%' : 'N/A'}
                      </div>
                    </div>
                    <div className="p-4 rounded-lg border border-studio-border bg-studio-panel">
                      <div className="text-xs text-studio-muted mb-1">Data Quality</div>
                      <div className="text-2xl font-bold">
                        {details.summary?.data_quality_score ? (details.summary.data_quality_score * 100).toFixed(1) + '%' : 'N/A'}
                      </div>
                    </div>
                    <div className="p-4 rounded-lg border border-studio-border bg-studio-panel">
                      <div className="text-xs text-studio-muted mb-1">Synthesis</div>
                      <div className="text-2xl font-bold">
                        {details.summary?.synthesis_score ? (details.summary.synthesis_score * 100).toFixed(1) + '%' : 'N/A'}
                      </div>
                    </div>
                  </div>

                  {/* Confidence */}
                  <div className="p-4 rounded-lg border border-studio-border bg-studio-panel">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm text-studio-muted">Confidence</span>
                      <span className="font-semibold">{details.summary?.confidence ? (details.summary.confidence * 100).toFixed(1) + '%' : 'N/A'}</span>
                    </div>
                    <div className="w-full bg-studio-border rounded-full h-2">
                      <div
                        className="bg-studio-accent h-2 rounded-full transition-all"
                        style={{ width: `${(details.summary?.confidence || 0) * 100}%` }}
                      />
                    </div>
                  </div>

                  {/* Evaluation Details */}
                  {details.evaluation && (
                    <div className="p-4 rounded-lg border border-studio-border bg-studio-panel">
                      <h3 className="text-sm font-medium mb-3">Evaluation Details</h3>
                      <pre className="text-xs text-studio-muted overflow-x-auto bg-black/30 p-3 rounded">
                        {JSON.stringify(details.evaluation, null, 2)}
                      </pre>
                    </div>
                  )}

                  {/* Tools Used */}
                  {details.summary?.tools_used && details.summary.tools_used.length > 0 && (
                    <div className="p-4 rounded-lg border border-studio-border bg-studio-panel">
                      <h3 className="text-sm font-medium mb-3">Tools Used</h3>
                      <div className="flex flex-wrap gap-2">
                        {details.summary.tools_used.map((tool: string, i: number) => (
                          <span key={i} className="px-2 py-1 bg-studio-accent/20 text-studio-accent rounded text-xs">
                            {tool}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* LLM Calls Tab */}
              {activeSection === 'llm' && (
                <div className="space-y-4">
                  {/* LLM Summary */}
                  {details.llm_summary && (
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div className="p-4 rounded-lg border border-studio-border bg-studio-panel">
                        <div className="text-xs text-studio-muted mb-1">Total Calls</div>
                        <div className="text-xl font-bold">{details.llm_summary.total_calls || 0}</div>
                      </div>
                      <div className="p-4 rounded-lg border border-studio-border bg-studio-panel">
                        <div className="text-xs text-studio-muted mb-1">Input Tokens</div>
                        <div className="text-xl font-bold">{details.llm_summary.total_input_tokens?.toLocaleString() || 0}</div>
                      </div>
                      <div className="p-4 rounded-lg border border-studio-border bg-studio-panel">
                        <div className="text-xs text-studio-muted mb-1">Output Tokens</div>
                        <div className="text-xl font-bold">{details.llm_summary.total_output_tokens?.toLocaleString() || 0}</div>
                      </div>
                      <div className="p-4 rounded-lg border border-studio-border bg-studio-panel">
                        <div className="text-xs text-studio-muted mb-1">Estimated Cost</div>
                        <div className="text-xl font-bold">{formatCost(details.llm_summary.total_cost || 0)}</div>
                      </div>
                    </div>
                  )}

                  {/* Individual Calls */}
                  {details.llm_calls && details.llm_calls.length > 0 ? (
                    <div className="rounded-lg border border-studio-border overflow-hidden">
                      <table className="w-full text-sm">
                        <thead className="bg-studio-panel">
                          <tr>
                            <th className="text-left p-3 text-studio-muted">Model</th>
                            <th className="text-left p-3 text-studio-muted">Call Type</th>
                            <th className="text-left p-3 text-studio-muted">Tokens (In/Out)</th>
                            <th className="text-left p-3 text-studio-muted">Estimated Cost</th>
                            <th className="text-left p-3 text-studio-muted">Duration</th>
                          </tr>
                        </thead>
                        <tbody>
                          {details.llm_calls.map((call: any, i: number) => (
                            <tr key={i} className="border-t border-studio-border">
                              <td className="p-3">
                                <code className="text-xs bg-studio-panel px-1.5 py-0.5 rounded">
                                  {call.model || 'Unknown'}
                                </code>
                              </td>
                              <td className="p-3">
                                <span className="px-2 py-0.5 bg-blue-900/30 text-blue-400 rounded text-xs">
                                  {call.call_type || '-'}
                                </span>
                              </td>
                              <td className="p-3">
                                {(call.prompt_tokens || call.input_tokens || 0).toLocaleString()} / {(call.completion_tokens || call.output_tokens || 0).toLocaleString()}
                              </td>
                              <td className="p-3">{formatCost(call.total_cost || call.cost || 0)}</td>
                              <td className="p-3">{(call.execution_time_ms || call.duration_ms) ? formatDuration(call.execution_time_ms || call.duration_ms) : '-'}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ) : (
                    <div className="text-center py-8 text-studio-muted border border-studio-border rounded-lg">
                      No LLM call data available
                    </div>
                  )}
                </div>
              )}

              {/* Nodes & Agents Tab */}
              {activeSection === 'nodes' && (
                <div className="space-y-4">
                  {/* LangGraph Summary */}
                  {details.langgraph_summary && (
                    <div className="p-4 rounded-lg border border-studio-border bg-studio-panel">
                      <h3 className="text-sm font-medium mb-3">Workflow Summary</h3>
                      <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
                        <div>
                          <span className="text-studio-muted">Total Nodes: </span>
                          <span className="font-medium">{details.langgraph_summary.total_nodes || 0}</span>
                        </div>
                        <div>
                          <span className="text-studio-muted">Total Events: </span>
                          <span className="font-medium">{details.langgraph_summary.total_events || 0}</span>
                        </div>
                        <div>
                          <span className="text-studio-muted">Duration: </span>
                          <span className="font-medium">
                            {details.langgraph_summary.total_duration_ms ? formatDuration(details.langgraph_summary.total_duration_ms) : 'N/A'}
                          </span>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Agents Used */}
                  {details.summary?.agents_used && details.summary.agents_used.length > 0 && (
                    <div className="p-4 rounded-lg border border-studio-border bg-studio-panel">
                      <h3 className="text-sm font-medium mb-3">Agents Used</h3>
                      <div className="flex flex-wrap gap-2">
                        {details.summary.agents_used.map((agent: string, i: number) => (
                          <span key={i} className="px-3 py-1.5 bg-purple-900/30 text-purple-400 border border-purple-800 rounded text-sm">
                            {agent}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* LangGraph Events */}
                  {details.langgraph_events && details.langgraph_events.length > 0 ? (
                    <div className="space-y-3">
                      <div className="flex items-center justify-between">
                        <h3 className="text-sm font-medium">Workflow Steps ({details.langgraph_events.length})</h3>
                      </div>
                      <div className="space-y-2 max-h-[600px] overflow-y-auto pr-2">
                        {details.langgraph_events.map((event: any, i: number) => {
                          const eventType = event.event_type || event.event_name || 'step';
                          const nodeName = event.node || event.node_name || event.event_name || 'Unknown';
                          const agentName = event.agent_name || '';
                          const duration = event.duration_ms || event.execution_time_ms || 0;
                          const status = event.status || (eventType.includes('end') || eventType.includes('exit') ? 'completed' : 'started');
                          const output = event.output_preview || event.output || event.output_data || event.input_preview || event.input_data;

                          return (
                            <div key={i} className="p-4 rounded-lg border border-studio-border bg-studio-panel hover:border-studio-accent/50 transition-colors">
                              {/* Header Row */}
                              <div className="flex items-center gap-2 flex-wrap">
                                <span className="text-lg font-mono text-studio-muted">#{i + 1}</span>
                                <span className={`px-2 py-1 rounded text-xs font-medium ${
                                  status === 'completed' || status === 'success' ? 'bg-green-900/50 text-green-400 border border-green-800' :
                                  status === 'started' || status === 'running' ? 'bg-blue-900/50 text-blue-400 border border-blue-800' :
                                  status === 'error' ? 'bg-red-900/50 text-red-400 border border-red-800' :
                                  'bg-gray-800 text-gray-400 border border-gray-700'
                                }`}>
                                  {status}
                                </span>
                                <span className="font-semibold text-studio-text">{nodeName}</span>
                                {agentName && (
                                  <span className="px-2 py-1 bg-purple-900/30 text-purple-400 border border-purple-800 rounded text-xs">
                                    {agentName}
                                  </span>
                                )}
                                {duration > 0 && (
                                  <span className="ml-auto text-sm text-studio-accent font-medium">{formatDuration(duration)}</span>
                                )}
                              </div>

                              {/* Details Row */}
                              <div className="flex items-center gap-4 mt-2 text-xs text-studio-muted">
                                {event.company_name && (
                                  <span>Company: <span className="text-studio-text">{event.company_name}</span></span>
                                )}
                                {event.step_number !== undefined && (
                                  <span>Step: <span className="text-studio-text">{event.step_number}</span></span>
                                )}
                                {(event.timestamp || event.logged_at) && (
                                  <span>Time: <span className="text-studio-text">{new Date(event.timestamp || event.logged_at).toLocaleTimeString()}</span></span>
                                )}
                              </div>

                              {/* Output Preview */}
                              {output && (
                                <div className="mt-3 p-3 bg-black/40 rounded border border-studio-border">
                                  <div className="text-xs text-studio-muted mb-1 font-medium">Output Preview:</div>
                                  <pre className="text-xs text-studio-text whitespace-pre-wrap overflow-x-auto max-h-40 overflow-y-auto">
                                    {typeof output === 'string'
                                      ? output.substring(0, 500) + (output.length > 500 ? '...' : '')
                                      : JSON.stringify(output, null, 2).substring(0, 500)}
                                  </pre>
                                </div>
                              )}

                              {/* Error */}
                              {event.error && (
                                <div className="mt-3 p-3 bg-red-900/20 rounded border border-red-800">
                                  <div className="text-xs text-red-400 font-medium">Error:</div>
                                  <p className="text-xs text-red-300 mt-1">{event.error}</p>
                                </div>
                              )}
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  ) : (
                    <div className="text-center py-8 text-studio-muted border border-studio-border rounded-lg">
                      No workflow event data available
                    </div>
                  )}
                </div>
              )}

              {/* Assessment Tab */}
              {activeSection === 'assessment' && (
                <div className="space-y-4">
                  {details.assessment ? (
                    <>
                      {/* Assessment Summary */}
                      <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                        <div className="p-4 rounded-lg border border-studio-border bg-studio-panel">
                          <div className="text-xs text-studio-muted mb-1">Risk Level</div>
                          <div className={`text-xl font-bold ${
                            (details.assessment.overall_risk_level || details.assessment.risk_level) === 'low' ? 'text-green-400' :
                            (details.assessment.overall_risk_level || details.assessment.risk_level) === 'medium' ? 'text-yellow-400' : 'text-red-400'
                          }`}>
                            {details.assessment.overall_risk_level || details.assessment.risk_level || 'N/A'}
                          </div>
                        </div>
                        <div className="p-4 rounded-lg border border-studio-border bg-studio-panel">
                          <div className="text-xs text-studio-muted mb-1">Credit Score</div>
                          <div className="text-xl font-bold text-studio-accent">
                            {details.assessment.credit_score_estimate || details.assessment.credit_score || 'N/A'}
                          </div>
                        </div>
                        <div className="p-4 rounded-lg border border-studio-border bg-studio-panel">
                          <div className="text-xs text-studio-muted mb-1">Confidence</div>
                          <div className="text-xl font-bold">
                            {(details.assessment.confidence_score || details.assessment.confidence)
                              ? ((details.assessment.confidence_score || details.assessment.confidence) * 100).toFixed(0) + '%'
                              : 'N/A'}
                          </div>
                        </div>
                      </div>

                      {/* Reasoning */}
                      {(details.assessment.llm_reasoning || details.assessment.reasoning) && (
                        <div className="p-4 rounded-lg border border-studio-border bg-studio-panel">
                          <h3 className="text-sm font-medium mb-2">Analysis Reasoning</h3>
                          <p className="text-sm text-studio-muted whitespace-pre-wrap">
                            {details.assessment.llm_reasoning || details.assessment.reasoning}
                          </p>
                        </div>
                      )}

                      {/* Risk & Positive Factors */}
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        {details.assessment.risk_factors && details.assessment.risk_factors.length > 0 && (
                          <div className="p-4 rounded-lg border border-red-900/50 bg-red-900/10">
                            <h3 className="text-sm font-medium text-red-400 mb-2">Risk Factors</h3>
                            <ul className="text-sm text-studio-muted space-y-1">
                              {details.assessment.risk_factors.map((factor: string, i: number) => (
                                <li key={i} className="flex items-start gap-2">
                                  <span className="text-red-400">•</span>
                                  {factor}
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}
                        {details.assessment.positive_factors && details.assessment.positive_factors.length > 0 && (
                          <div className="p-4 rounded-lg border border-green-900/50 bg-green-900/10">
                            <h3 className="text-sm font-medium text-green-400 mb-2">Positive Factors</h3>
                            <ul className="text-sm text-studio-muted space-y-1">
                              {details.assessment.positive_factors.map((factor: string, i: number) => (
                                <li key={i} className="flex items-start gap-2">
                                  <span className="text-green-400">•</span>
                                  {factor}
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}
                      </div>

                      {/* Recommendations */}
                      {details.assessment.recommendations && details.assessment.recommendations.length > 0 && (
                        <div className="p-4 rounded-lg border border-studio-accent/50 bg-studio-accent/10">
                          <h3 className="text-sm font-medium text-studio-accent mb-2">Recommendations</h3>
                          <ul className="text-sm text-studio-muted space-y-1">
                            {details.assessment.recommendations.map((rec: string, i: number) => (
                              <li key={i} className="flex items-start gap-2">
                                <span className="text-studio-accent">{i + 1}.</span>
                                {rec}
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}

                      {/* Full Assessment JSON */}
                      <div className="p-4 rounded-lg border border-studio-border bg-studio-panel">
                        <h3 className="text-sm font-medium mb-3">Full Assessment Data</h3>
                        <pre className="text-xs text-studio-muted overflow-x-auto bg-black/30 p-3 rounded max-h-96 overflow-y-auto">
                          {JSON.stringify(details.assessment, null, 2)}
                        </pre>
                      </div>
                    </>
                  ) : (
                    <div className="text-center py-8 text-studio-muted border border-studio-border rounded-lg">
                      No assessment data available
                    </div>
                  )}
                </div>
              )}

              {/* Coalition Tab */}
              {activeSection === 'coalition' && (
                <div className="space-y-4">
                  {details.coalition ? (
                    <>
                      {/* Correctness Overview */}
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <div className="p-4 rounded-lg border border-studio-border bg-studio-panel">
                          <div className="text-xs text-studio-muted mb-1">Correctness</div>
                          <div className={`text-2xl font-bold ${
                            details.coalition.correctness_category === 'high' ? 'text-green-400' :
                            details.coalition.correctness_category === 'medium' ? 'text-yellow-400' : 'text-red-400'
                          }`}>
                            {details.coalition.correctness_score ? (details.coalition.correctness_score * 100).toFixed(1) + '%' : 'N/A'}
                          </div>
                          <div className={`text-xs mt-1 ${
                            details.coalition.correctness_category === 'high' ? 'text-green-400' :
                            details.coalition.correctness_category === 'medium' ? 'text-yellow-400' : 'text-red-400'
                          }`}>
                            {details.coalition.correctness_category?.toUpperCase()}
                          </div>
                        </div>
                        <div className="p-4 rounded-lg border border-studio-border bg-studio-panel">
                          <div className="text-xs text-studio-muted mb-1">Confidence</div>
                          <div className="text-2xl font-bold">
                            {details.coalition.confidence ? (details.coalition.confidence * 100).toFixed(1) + '%' : 'N/A'}
                          </div>
                        </div>
                        <div className="p-4 rounded-lg border border-studio-border bg-studio-panel">
                          <div className="text-xs text-studio-muted mb-1">Agreement</div>
                          <div className="text-2xl font-bold">
                            {details.coalition.agreement_score ? (details.coalition.agreement_score * 100).toFixed(1) + '%' : 'N/A'}
                          </div>
                        </div>
                        <div className="p-4 rounded-lg border border-studio-border bg-studio-panel">
                          <div className="text-xs text-studio-muted mb-1">Evaluators</div>
                          <div className="text-2xl font-bold">{details.coalition.num_evaluators || 0}</div>
                        </div>
                      </div>

                      {/* Component Scores */}
                      <div className="p-4 rounded-lg border border-studio-border bg-studio-panel">
                        <h3 className="text-sm font-medium mb-4">Component Scores</h3>
                        <div className="space-y-3">
                          {[
                            { label: 'Agent Efficiency', score: details.coalition.efficiency_score, color: 'blue' },
                            { label: 'LLM Quality', score: details.coalition.quality_score, color: 'purple' },
                            { label: 'Tool Selection', score: details.coalition.tool_score, color: 'green' },
                            { label: 'Consistency', score: details.coalition.consistency_score, color: 'orange' },
                          ].map(({ label, score, color }) => (
                            <div key={label}>
                              <div className="flex items-center justify-between mb-1">
                                <span className="text-sm text-studio-muted">{label}</span>
                                <span className="text-sm font-medium">{score ? (score * 100).toFixed(1) + '%' : 'N/A'}</span>
                              </div>
                              <div className="w-full bg-studio-border rounded-full h-2">
                                <div
                                  className={`h-2 rounded-full transition-all bg-${color}-500`}
                                  style={{
                                    width: `${(score || 0) * 100}%`,
                                    backgroundColor: color === 'blue' ? '#3b82f6' :
                                      color === 'purple' ? '#a855f7' :
                                      color === 'green' ? '#22c55e' : '#f97316'
                                  }}
                                />
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>

                      {/* Individual Votes */}
                      {details.coalition.votes && details.coalition.votes.length > 0 && (
                        <div className="rounded-lg border border-studio-border overflow-hidden">
                          <div className="p-3 bg-studio-panel border-b border-studio-border">
                            <h3 className="text-sm font-medium">Evaluator Votes</h3>
                          </div>
                          <table className="w-full text-sm">
                            <thead className="bg-studio-panel/50">
                              <tr>
                                <th className="text-left p-3 text-studio-muted">Evaluator</th>
                                <th className="text-left p-3 text-studio-muted">Score</th>
                                <th className="text-left p-3 text-studio-muted">Confidence</th>
                                <th className="text-left p-3 text-studio-muted">Weight</th>
                                <th className="text-left p-3 text-studio-muted">Weighted</th>
                              </tr>
                            </thead>
                            <tbody>
                              {details.coalition.votes.map((vote: any, i: number) => (
                                <tr key={i} className="border-t border-studio-border">
                                  <td className="p-3 font-medium">{vote.evaluator?.replace('_', ' ')}</td>
                                  <td className="p-3">
                                    <span className={vote.score >= 0.7 ? 'text-green-400' : vote.score >= 0.4 ? 'text-yellow-400' : 'text-red-400'}>
                                      {(vote.score * 100).toFixed(1)}%
                                    </span>
                                  </td>
                                  <td className="p-3">{(vote.confidence * 100).toFixed(0)}%</td>
                                  <td className="p-3">{(vote.weight * 100).toFixed(0)}%</td>
                                  <td className="p-3">{(vote.weighted_score * 100).toFixed(2)}%</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      )}

                      {/* Vote Details */}
                      <div className="p-4 rounded-lg border border-studio-border bg-studio-panel">
                        <h3 className="text-sm font-medium mb-3">Full Evaluation Details</h3>
                        <pre className="text-xs text-studio-muted overflow-x-auto bg-black/30 p-3 rounded max-h-96 overflow-y-auto">
                          {JSON.stringify(details.coalition, null, 2)}
                        </pre>
                      </div>
                    </>
                  ) : (
                    <div className="text-center py-8 text-studio-muted border border-studio-border rounded-lg">
                      <AlertCircle className="w-8 h-8 mx-auto mb-2 opacity-50" />
                      <p>Coalition evaluation not available for this run</p>
                      <p className="text-xs mt-1">Evaluation may still be processing</p>
                    </div>
                  )}
                </div>
              )}

              {/* State Tab */}
              {activeSection === 'state' && (
                <div className="space-y-4">
                  {/* Load State Button */}
                  {!stateDump && !stateLoading && (
                    <div className="text-center py-8">
                      <button
                        onClick={async () => {
                          setStateLoading(true)
                          try {
                            const res = await fetch(`${API_URL}/pg/state-dump/${runId}`)
                            if (res.ok) {
                              const data = await res.json()
                              setStateDump(data)
                            }
                          } catch (err) {
                            console.error('Failed to load state dump:', err)
                          } finally {
                            setStateLoading(false)
                          }
                        }}
                        className="px-4 py-2 bg-studio-accent text-white rounded hover:opacity-80 transition-opacity"
                      >
                        Load Full State Dump
                      </button>
                      <p className="text-xs text-studio-muted mt-2">
                        Click to fetch the complete workflow state
                      </p>
                    </div>
                  )}

                  {stateLoading && (
                    <div className="text-center py-8">
                      <Loader2 className="w-6 h-6 animate-spin mx-auto text-studio-muted" />
                      <p className="text-studio-muted mt-2">Loading state dump...</p>
                    </div>
                  )}

                  {stateDump && (
                    <>
                      {/* State Metadata */}
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <div className="p-4 rounded-lg border border-studio-border bg-studio-panel">
                          <div className="text-xs text-studio-muted mb-1">Total Size</div>
                          <div className="font-semibold">
                            {stateDump.metadata?.total_state_size_bytes
                              ? `${(stateDump.metadata.total_state_size_bytes / 1024).toFixed(1)} KB`
                              : 'N/A'}
                          </div>
                        </div>
                        <div className="p-4 rounded-lg border border-studio-border bg-studio-panel">
                          <div className="text-xs text-studio-muted mb-1">Duration</div>
                          <div className="font-semibold">
                            {stateDump.metadata?.duration_ms
                              ? `${(stateDump.metadata.duration_ms / 1000).toFixed(2)}s`
                              : 'N/A'}
                          </div>
                        </div>
                        <div className="p-4 rounded-lg border border-studio-border bg-studio-panel">
                          <div className="text-xs text-studio-muted mb-1">Coalition Score</div>
                          <div className="font-semibold text-studio-accent">
                            {stateDump.scores?.coalition_score
                              ? `${(stateDump.scores.coalition_score * 100).toFixed(1)}%`
                              : 'N/A'}
                          </div>
                        </div>
                        <div className="p-4 rounded-lg border border-studio-border bg-studio-panel">
                          <div className="text-xs text-studio-muted mb-1">Agent Score</div>
                          <div className="font-semibold">
                            {stateDump.scores?.agent_metrics_score
                              ? `${(stateDump.scores.agent_metrics_score * 100).toFixed(1)}%`
                              : 'N/A'}
                          </div>
                        </div>
                      </div>

                      {/* Company Info */}
                      {stateDump.state?.company_info && Object.keys(stateDump.state.company_info).length > 0 && (
                        <div className="p-4 rounded-lg border border-studio-border bg-studio-panel">
                          <h3 className="text-sm font-medium mb-3">Company Info</h3>
                          <pre className="text-xs text-studio-muted overflow-x-auto bg-black/30 p-3 rounded max-h-48 overflow-y-auto">
                            {JSON.stringify(stateDump.state.company_info, null, 2)}
                          </pre>
                        </div>
                      )}

                      {/* Plan */}
                      {stateDump.state?.plan && stateDump.state.plan.length > 0 && (
                        <div className="p-4 rounded-lg border border-studio-border bg-studio-panel">
                          <h3 className="text-sm font-medium mb-3">Task Plan ({stateDump.state.plan.length} tasks)</h3>
                          <pre className="text-xs text-studio-muted overflow-x-auto bg-black/30 p-3 rounded max-h-48 overflow-y-auto">
                            {JSON.stringify(stateDump.state.plan, null, 2)}
                          </pre>
                        </div>
                      )}

                      {/* API Data */}
                      {stateDump.state?.api_data && Object.keys(stateDump.state.api_data).length > 0 && (
                        <div className="p-4 rounded-lg border border-studio-border bg-studio-panel">
                          <h3 className="text-sm font-medium mb-3">API Data ({Object.keys(stateDump.state.api_data).length} sources)</h3>
                          <div className="flex flex-wrap gap-2 mb-3">
                            {Object.keys(stateDump.state.api_data).map((source: string) => (
                              <span key={source} className="px-2 py-1 bg-blue-900/30 text-blue-400 rounded text-xs">
                                {source}
                              </span>
                            ))}
                          </div>
                          <pre className="text-xs text-studio-muted overflow-x-auto bg-black/30 p-3 rounded max-h-64 overflow-y-auto">
                            {JSON.stringify(stateDump.state.api_data, null, 2)}
                          </pre>
                        </div>
                      )}

                      {/* Search Data */}
                      {stateDump.state?.search_data && Object.keys(stateDump.state.search_data).length > 0 && (
                        <div className="p-4 rounded-lg border border-studio-border bg-studio-panel">
                          <h3 className="text-sm font-medium mb-3">Search Data</h3>
                          <pre className="text-xs text-studio-muted overflow-x-auto bg-black/30 p-3 rounded max-h-48 overflow-y-auto">
                            {JSON.stringify(stateDump.state.search_data, null, 2)}
                          </pre>
                        </div>
                      )}

                      {/* Assessment */}
                      {stateDump.state?.assessment && Object.keys(stateDump.state.assessment).length > 0 && (
                        <div className="p-4 rounded-lg border border-green-900/50 bg-green-900/10">
                          <h3 className="text-sm font-medium text-green-400 mb-3">Assessment</h3>
                          <pre className="text-xs text-studio-muted overflow-x-auto bg-black/30 p-3 rounded max-h-48 overflow-y-auto">
                            {JSON.stringify(stateDump.state.assessment, null, 2)}
                          </pre>
                        </div>
                      )}

                      {/* Evaluation */}
                      {stateDump.state?.evaluation && Object.keys(stateDump.state.evaluation).length > 0 && (
                        <div className="p-4 rounded-lg border border-purple-900/50 bg-purple-900/10">
                          <h3 className="text-sm font-medium text-purple-400 mb-3">Evaluation</h3>
                          <pre className="text-xs text-studio-muted overflow-x-auto bg-black/30 p-3 rounded max-h-48 overflow-y-auto">
                            {JSON.stringify(stateDump.state.evaluation, null, 2)}
                          </pre>
                        </div>
                      )}

                      {/* Errors */}
                      {stateDump.state?.errors && stateDump.state.errors.length > 0 && (
                        <div className="p-4 rounded-lg border border-red-900/50 bg-red-900/10">
                          <h3 className="text-sm font-medium text-red-400 mb-3">Errors ({stateDump.state.errors.length})</h3>
                          <ul className="text-sm text-studio-muted space-y-1">
                            {stateDump.state.errors.map((error: string, i: number) => (
                              <li key={i} className="flex items-start gap-2">
                                <span className="text-red-400">•</span>
                                {error}
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </>
                  )}

                  {!stateLoading && stateDump === null && (
                    <div className="text-center py-8 text-studio-muted border border-studio-border rounded-lg">
                      <Database className="w-8 h-8 mx-auto mb-2 opacity-50" />
                      <p>State dump available on demand</p>
                      <p className="text-xs mt-1">Full state includes all workflow data</p>
                    </div>
                  )}
                </div>
              )}
            </div>
          </>
        )}

        {/* Not Found */}
        {!loading && (!details || !details.found) && (
          <div className="flex-1 flex items-center justify-center py-20">
            <div className="text-center text-studio-muted">
              <AlertCircle className="w-12 h-12 mx-auto mb-3 opacity-50" />
              <p>No data found for this run</p>
              <p className="text-xs mt-1">Run ID: {runId}</p>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
