import {
  startTransition,
  useDeferredValue,
  useEffect,
  useEffectEvent,
  useState,
  useTransition,
  useRef,
} from 'react'
import TwinScene3D from './components/TwinScene3D'
import './App.css'

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8000'
const LAYER_LABELS = {
  roads: 'road lines',
  drains: 'drain lines',
  rivers: 'river/canal lines',
  waterbodies: 'waterbodies',
  houses: 'houses',
  playgrounds: 'playgrounds',
  parks: 'parks',
  blocked_drain_network: 'blocked segments',
  blocked_drains: 'blocked markers',
  flood_zones: 'flood zones',
  informal_zones: 'informal zones',
}

const DASHBOARD_SUBSECTIONS = [
  {
    id: 'overview',
    label: 'Risk & Indicators',
    description: 'Neighborhood risk profile, active problems, and indicator snapshots.',
  },
  {
    id: 'prioritization',
    label: 'Policy Prioritization',
    description: 'AI-ranked micro-works with evidence-based justifications.',
  },
  {
    id: 'simulation',
    label: 'Budget & Scenario',
    description: 'Scenario simulation for budget allocation and outcome forecasting.',
  },
  {
    id: 'governance',
    label: 'Exports & Governance',
    description: 'SDG cards, policy briefs, and official document exports.',
  },
]

const PLANNING_STRATEGIES = [
  {
    id: 'balanced',
    label: 'Balanced',
    description: 'Steady portfolio across impact, equity, urgency, and readiness.',
  },
  {
    id: 'climate_resilience',
    label: 'Climate',
    description: 'Favors flood, drainage, water, and green resilience actions.',
  },
  {
    id: 'equity_first',
    label: 'Equity',
    description: 'Pushes high-need, high-beneficiary work to the front.',
  },
  {
    id: 'fast_delivery',
    label: 'Fast Delivery',
    description: 'Biases toward permit-light, ready-to-move projects.',
  },
]

function App() {
  const [token, setToken] = useState('')
  const [user, setUser] = useState(null)
  const [mode, setMode] = useState('login')
  const [authForm, setAuthForm] = useState({
    username: '',
    password: '',
    role: 'viewer',
  })
  const [wards, setWards] = useState([])
  const [selectedWard, setSelectedWard] = useState(null)
  const [digitalTwin, setDigitalTwin] = useState(null)
  const [digitalTwinScene, setDigitalTwinScene] = useState(null)
  const [workflow, setWorkflow] = useState(null)
  const [sdgCard, setSdgCard] = useState(null)
  const [interagencyPacket, setInteragencyPacket] = useState(null)
  const [aiComponents, setAiComponents] = useState([])
  const [auditTrail, setAuditTrail] = useState([])
  const [notificationFeed, setNotificationFeed] = useState([])
  const [liveResult, setLiveResult] = useState(null)
  const [autoLive, setAutoLive] = useState(false)
  const [liveConfig, setLiveConfig] = useState({
    citizen_reports: 3,
    odk_forms: 2,
    sensor_pulses: 1,
  })
  const [worklist, setWorklist] = useState([])
  const [scenario, setScenario] = useState(null)
  const [aiBudgetPlan, setAiBudgetPlan] = useState(null)
  const [budgetInput, setBudgetInput] = useState(50)
  const [isPredicting, setIsPredicting] = useState(false)
  const [reports, setReports] = useState([])
  const [budget, setBudget] = useState(9)
  const [planningStrategy, setPlanningStrategy] = useState('balanced')
  const [groundMode, setGroundMode] = useState('satellite-sim')
  const [activeSubsection, setActiveSubsection] = useState('overview')
  const [reportText, setReportText] = useState('')
  const [chatHistory, setChatHistory] = useState([])
  const [isChatting, setIsChatting] = useState(false)
  const chatScrollRef = useRef(null)
  const [odkForm, setOdkForm] = useState({
    text: '',
    issue_type: 'auto',
    severity: 'medium',
    location_hint: '',
    reporter_name: '',
  })
  const [classified, setClassified] = useState(null)
  const [error, setError] = useState('')
  const [info, setInfo] = useState('')
  const [loading, startLoading] = useTransition()
  const deferredReportText = useDeferredValue(reportText)

  const authedHeaders = token
    ? { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' }
    : { 'Content-Type': 'application/json' }

  async function request(path, options = {}) {
    const response = await fetch(`${API_BASE}${path}`, {
      ...options,
      headers: {
        ...authedHeaders,
        ...(options.headers ?? {}),
      },
    })
    if (!response.ok) {
      const body = await response.text()
      throw new Error(body || `Request failed: ${response.status}`)
    }
    const contentType = response.headers.get('content-type') || ''
    if (contentType.includes('application/json')) {
      return response.json()
    }
    return response
  }

  const loadWardsEvent = useEffectEvent(async () => {
    const wardRows = await request('/api/wards')
    startTransition(() => {
      setWards(wardRows)
      if (!selectedWard && wardRows.length > 0) {
        setSelectedWard(wardRows[0].id)
      }
    })
  })

  async function loadWardData(
    wardId,
    currentBudget = budget,
    currentStrategy = planningStrategy
  ) {
    if (!wardId) {
      return
    }
    startLoading(async () => {
      try {
        const [
          twin,
          scene,
          workflowSummary,
          top,
          sim,
          latestReports,
          card,
          packet,
          components,
          audit,
          notifications,
        ] = await Promise.all([
          request(`/api/wards/${wardId}/digital-twin`),
          request(`/api/wards/${wardId}/digital-twin-scene`),
          request(`/api/wards/${wardId}/workflow`),
          request(`/api/wards/${wardId}/top-worklist?top_n=10`),
          request(
            `/api/wards/${wardId}/scenario?budget_lakh=${currentBudget}&strategy=${currentStrategy}`
          ),
          request(`/api/reports?ward_id=${wardId}&limit=12`),
          request(`/api/wards/${wardId}/sdg11-card`),
          request(`/api/wards/${wardId}/interagency-packet`),
          request('/api/ai-components'),
          request(`/api/audit-trail?ward_id=${wardId}&limit=20`),
          request(`/api/wards/${wardId}/notification-feed?limit=25`),
        ])
        setDigitalTwin(twin)
        setDigitalTwinScene(scene)
        setWorkflow(workflowSummary)
        setSdgCard(card)
        setInteragencyPacket(packet)
        setAiComponents(components)
        setAuditTrail(audit)
        setNotificationFeed(notifications)
        setWorklist(top.items || [])
        setScenario(sim)
        setReports(latestReports)
        setError('')
      } catch (err) {
        setError(String(err.message || err))
      }
    })
  }

  const syncWardData = useEffectEvent((wardId, currentBudget, currentStrategy) => {
    loadWardData(wardId, currentBudget, currentStrategy)
  })

  async function handleAuthSubmit(event) {
    event.preventDefault()
    setError('')
    setInfo('')
    try {
      if (mode === 'register') {
        await request('/api/auth/register', {
          method: 'POST',
          body: JSON.stringify(authForm),
        })
        setInfo('Registration complete. Please log in with your credentials.')
        setMode('login')
        return
      }
      const loginRes = await request('/api/auth/login', {
        method: 'POST',
        body: JSON.stringify({
          username: authForm.username,
          password: authForm.password,
        }),
      })
      setToken(loginRes.access_token)
      setUser({ username: loginRes.username, role: loginRes.role })
    } catch (err) {
      setError(String(err.message || err))
    }
  }

  async function classifyCurrentText() {
    if (!deferredReportText || !selectedWard) {
      return
    }
    try {
      const prediction = await request('/api/reports/classify', {
        method: 'POST',
        body: JSON.stringify({
          ward_id: selectedWard,
          text: deferredReportText,
          language: 'mixed',
        }),
      })
      setClassified(prediction)
    } catch (err) {
      setError(String(err.message || err))
    }
  }

  useEffect(() => {
    if (chatScrollRef.current) {
      chatScrollRef.current.scrollTop = chatScrollRef.current.scrollHeight
    }
  }, [chatHistory])

  async function handleChatQuery() {
    if (!reportText.trim()) return
    const query = reportText
    setReportText('')
    setIsChatting(true)
    
    // Optimistic update
    const userMsg = { role: 'user', text: query, timestamp: new Date() }
    setChatHistory(prev => [...prev, userMsg])
    
    try {
      const res = await request('/api/reports/chat', {
        method: 'POST',
        body: JSON.stringify({ 
          query,
          ward_id: selectedWard // Inject live ward context
        }),
      })
      const aiMsg = { 
        role: 'ai', 
        text: res.answer, 
        sources: res.sources,
        timestamp: new Date() 
      }
      setChatHistory(prev => [...prev, aiMsg])
    } catch (err) {
      setError(String(err.message || err))
      setChatHistory(prev => [...prev, { 
        role: 'ai', 
        text: `Error: ${err.message}`, 
        isError: true,
        timestamp: new Date() 
      }])
    } finally {
      setIsChatting(false)
    }
  }

  async function submitReport() {
    if (!reportText || !selectedWard) {
      return
    }
    try {
      await request('/api/reports', {
        method: 'POST',
        body: JSON.stringify({
          ward_id: selectedWard,
          text: reportText,
          language: 'mixed',
        }),
      })
      setReportText('')
      setClassified(null)
      await loadWardData(selectedWard, budget)
    } catch (err) {
      setError(String(err.message || err))
    }
  }

  async function submitOdkForm() {
    if (!selectedWard || !odkForm.text.trim()) {
      return
    }
    try {
      await request('/api/reports/odk-submit', {
        method: 'POST',
        body: JSON.stringify({
          ward_id: selectedWard,
          text: odkForm.text,
          language: 'mixed',
          issue_type: odkForm.issue_type,
          location_hint: odkForm.location_hint,
          reporter_name: odkForm.reporter_name,
          severity: odkForm.severity,
        }),
      })
      setOdkForm({
        text: '',
        issue_type: 'auto',
        severity: 'medium',
        location_hint: '',
        reporter_name: '',
      })
      await loadWardData(selectedWard, budget)
    } catch (err) {
      setError(String(err.message || err))
    }
  }

  async function runScenario() {
    if (!selectedWard) {
      return
    }
    try {
      const sim = await request(
        `/api/wards/${selectedWard}/scenario?budget_lakh=${budget}&strategy=${planningStrategy}`
      )
      setScenario(sim)
    } catch (err) {
      setError(String(err.message || err))
    }
  }

  async function runAiBudgetPlan() {
    if (!selectedWard) return
    setIsPredicting(true)
    try {
      const plan = await request(
        `/api/wards/${selectedWard}/ai-budget-plan?budget_lakh=${budgetInput}`
      )
      setAiBudgetPlan(plan)
      setError('')
    } catch (err) {
      setError(String(err.message || err))
    } finally {
      setIsPredicting(false)
    }
  }

  async function runLiveIngest() {
    if (!selectedWard) {
      return
    }
    try {
      const result = await request(`/api/wards/${selectedWard}/live-ingest`, {
        method: 'POST',
        body: JSON.stringify(liveConfig),
      })
      setLiveResult(result)
      await loadWardData(selectedWard, budget)
    } catch (err) {
      setError(String(err.message || err))
    }
  }

  async function downloadExport(path, filename) {
    try {
      const response = await fetch(`${API_BASE}${path}`, {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      })
      if (!response.ok) {
        throw new Error(`Export failed: ${response.status}`)
      }
      const blob = await response.blob()
      const url = URL.createObjectURL(blob)
      const anchor = document.createElement('a')
      anchor.href = url
      anchor.download = filename
      anchor.click()
      URL.revokeObjectURL(url)
    } catch (err) {
      setError(String(err.message || err))
    }
  }

  useEffect(() => {
    if (!token) {
      return
    }
    loadWardsEvent()
  }, [token])

  useEffect(() => {
    if (!token || !selectedWard) {
      return
    }
    syncWardData(selectedWard, budget, planningStrategy)
  }, [token, selectedWard, budget, planningStrategy])

  const autoLiveTick = useEffectEvent(async () => {
    await runLiveIngest()
  })

  useEffect(() => {
    if (!token || !selectedWard || !autoLive) {
      return undefined
    }
    const intervalId = window.setInterval(() => {
      autoLiveTick()
    }, 30000)
    return () => window.clearInterval(intervalId)
  }, [
    token,
    selectedWard,
    autoLive,
    liveConfig.citizen_reports,
    liveConfig.odk_forms,
    liveConfig.sensor_pulses,
  ])

  if (!token || !user) {
    return (
      <main className="auth-shell">
        <section className="auth-card">
          <h1>UrbanSightAI Pilot</h1>
          <p>Data-driven governance for resilient city planning.</p>
          <div className="mode-toggle">
            <button
              className={mode === 'login' ? 'active' : ''}
              onClick={() => setMode('login')}
            >
              Login
            </button>
            <button
              className={mode === 'register' ? 'active' : ''}
              onClick={() => setMode('register')}
            >
              Register
            </button>
          </div>
          <form onSubmit={handleAuthSubmit} className="auth-form">
            <input
              placeholder="Username"
              value={authForm.username}
              onChange={(event) =>
                setAuthForm({ ...authForm, username: event.target.value })
              }
              required
            />
            <input
              type="password"
              placeholder="Password"
              value={authForm.password}
              onChange={(event) =>
                setAuthForm({ ...authForm, password: event.target.value })
              }
              required
            />
            {mode === 'register' && (
              <select
                value={authForm.role}
                onChange={(event) =>
                  setAuthForm({ ...authForm, role: event.target.value })
                }
              >
                <option value="viewer">viewer</option>
                <option value="enumerator">enumerator</option>
                <option value="planner">planner</option>
              </select>
            )}
            <button type="submit">
              {mode === 'login' ? 'Enter Dashboard' : 'Create Account'}
            </button>
          </form>
          <small className="hint">
            Default planner login: <code>planner / pilot123</code>
          </small>
          {info && <p className="status ok">{info}</p>}
          {error && <p className="status err">{error}</p>}
        </section>
      </main>
    )
  }

  const wardSummary = wards.find((ward) => ward.id === selectedWard)
  const layerSummary = digitalTwinScene?.layers?.summary ?? {}
  const displayedScores = digitalTwinScene?.scores ?? digitalTwin?.indicators
  const activeSubsectionMeta =
    DASHBOARD_SUBSECTIONS.find((section) => section.id === activeSubsection) ??
    DASHBOARD_SUBSECTIONS[0]
  const activeStrategyMeta =
    PLANNING_STRATEGIES.find((strategy) => strategy.id === planningStrategy) ??
    PLANNING_STRATEGIES[0]

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <h2>UrbanSightAI</h2>
          <p>
            {user.username} ({user.role}) | {loading ? 'Updating…' : 'Live'}
          </p>
        </div>
        <div className="actions">
          <button
            onClick={() => {
              setToken('')
              setUser(null)
              setDigitalTwin(null)
              setDigitalTwinScene(null)
              setWorkflow(null)
              setSdgCard(null)
              setInteragencyPacket(null)
              setAiComponents([])
              setAuditTrail([])
              setNotificationFeed([])
              setLiveResult(null)
              setAutoLive(false)
              setWorklist([])
              setReports([])
              setScenario(null)
              setPlanningStrategy('balanced')
              setActiveSubsection('overview')
            }}
          >
            Logout
          </button>
        </div>
      </header>

      <section className="hero">
        <div className="ward-picker">
          <h3>Ward Selector</h3>
          <select
            value={selectedWard ?? ''}
            onChange={(event) => setSelectedWard(Number(event.target.value))}
          >
            {wards.map((ward) => (
              <option key={ward.id} value={ward.id}>
                {ward.code} - {ward.name}
              </option>
            ))}
          </select>
          <p className="stamp">
            Last update:{' '}
            {wardSummary?.last_updated
              ? new Date(wardSummary.last_updated).toLocaleString()
              : '-'}
          </p>
          <div className="sdg-chip">SDG-11 Score: {wardSummary?.sdg11_score?.toFixed(2)}</div>
        </div>

        <div className="twin-map">
          <h3>Ward-Level 3D Digital Twin</h3>
          <div className="map-style-toggle">
            <button
              type="button"
              className={groundMode === 'satellite-sim' ? 'active' : ''}
              onClick={() => setGroundMode('satellite-sim')}
            >
              Simulated Satellite
            </button>
            <button
              type="button"
              className={groundMode === 'stylized' ? 'active' : ''}
              onClick={() => setGroundMode('stylized')}
            >
              Stylized Terrain
            </button>
          </div>
          <TwinScene3D
            key={
              digitalTwinScene?.data_sources?.generated_at ??
              digitalTwinScene?.ward?.id ??
              selectedWard ??
              'scene'
            }
            data={digitalTwinScene}
            groundMode={groundMode}
          />
          {displayedScores && (
            <div className="metric-grid">
              <article>
                <span>Informal Area</span>
                <strong>{displayedScores.informal_area_pct.toFixed(1)}%</strong>
              </article>
              <article>
                <span>Blocked Drain Count</span>
                <strong>{displayedScores.blocked_drain_count}</strong>
              </article>
              <article>
                <span>Green Deficit Index</span>
                <strong>{displayedScores.green_deficit_index.toFixed(2)}</strong>
              </article>
              <article>
                <span>Exposed Population</span>
                <strong>{displayedScores.exposed_population.toLocaleString()}</strong>
              </article>
            </div>
          )}
          <div className="layer-badges">
            {Object.entries(layerSummary).map(([label, count]) => (
              <span key={label}>
                {LAYER_LABELS[label] ?? label.replaceAll('_', ' ')}: {count}
              </span>
            ))}
          </div>
          <div className="map-legend">
            <span className="map-key">
              <i className="map-dot road" />
              Roads
            </span>
            <span className="map-key">
              <i className="map-dot drain" />
              Drains
            </span>
            <span className="map-key">
              <i className="map-dot river" />
              Rivers/Canals
            </span>
            <span className="map-key">
              <i className="map-dot water" />
              Waterbodies
            </span>
            <span className="map-key">
              <i className="map-dot informal" />
              Informal Area
            </span>
            <span className="map-key">
              <i className="map-dot flood" />
              Flood Risk
            </span>
            <span className="map-key">
              <i className="map-dot blocked" />
              Blocked Network (Red)
            </span>
          </div>
          <p className="tiny-note">
            Risk overlays: {digitalTwinScene?.layers?.blocked_drains?.length ?? 0} blocked drains,{' '}
            {digitalTwinScene?.layers?.rivers?.length ?? 0} river/canal segments,{' '}
            {digitalTwinScene?.layers?.flood_zones?.length ?? 0} flood exposure zones,{' '}
            {digitalTwinScene?.layers?.informal_zones?.length ?? 0} informal-settlement clusters.
          </p>
          {digitalTwinScene?.data_sources && (
            <div className="source-box">
              <strong>Public Source Attribution</strong>
              <p>
                Boundaries: {digitalTwinScene.data_sources.boundaries_source} | Map features:{' '}
                {digitalTwinScene.data_sources.map_features_source}
              </p>
              <p>
                Weather: {digitalTwinScene.data_sources.weather_source} (
                {digitalTwinScene.data_sources.weather_status}) | Rainfall{' '}
                {digitalTwinScene.data_sources.rainfall_mm?.toFixed?.(2) ?? '-'} mm | Temp{' '}
                {digitalTwinScene.data_sources.temperature_c?.toFixed?.(1) ?? '-'} C
              </p>
              <p>
                River segments: {digitalTwinScene.data_sources.river_segments} | Weather observed:{' '}
                {digitalTwinScene.data_sources.weather_observed_at
                  ? new Date(digitalTwinScene.data_sources.weather_observed_at).toLocaleString()
                  : 'Unavailable'}
              </p>
            </div>
          )}
          {(digitalTwinScene?.integrity_notes?.length ?? 0) > 0 && (
            <div className="integrity-box">
              <strong>Consistency Notes</strong>
              <ul className="report-list compact-list">
                {digitalTwinScene.integrity_notes.map((note, index) => (
                  <li key={`integrity-${index}`}>
                    <p>{note}</p>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </section>

      <section className="content-grid global-assistant">
        <article className="panel">
          <h3>UrbanSightAI Expert Assistant (RAG)</h3>
          <div className="chat-window" ref={chatScrollRef} style={{ maxHeight: '250px' }}>
            {chatHistory.length === 0 && (
              <p className="chat-placeholder">Ask any question about ward risks, drainage specs, or the UrbanSightAI proposal...</p>
            )}
            {chatHistory.map((msg, idx) => (
              <div key={idx} className={`chat-bubble ${msg.role}`}>
                <div className="bubble-content">
                  <p>{msg.text}</p>
                  {msg.sources && msg.sources.length > 0 && (
                    <div className="chat-sources">
                      <strong>Sources:</strong>
                      <ul>
                        {msg.sources.map((s, i) => <li key={i}>{s.substring(0, 60)}...</li>)}
                      </ul>
                    </div>
                  )}
                </div>
                <span className="chat-time">{new Date(msg.timestamp).toLocaleTimeString()}</span>
              </div>
            ))}
            {isChatting && <div className="chat-bubble ai pulse">AI is thinking...</div>}
          </div>
          <div className="chat-input-row">
            <textarea
              rows="1"
              placeholder="Ask for policy advice, reporting an issue, or ward stats..."
              value={reportText}
              onChange={(event) => setReportText(event.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleChatQuery(); } }}
            />
            <button disabled={isChatting} onClick={handleChatQuery}>
              {isChatting ? 'Searching...' : 'Ask AI'}
            </button>
          </div>
          {classified && (
            <div className="prediction" style={{fontSize: '0.75rem', marginTop: '0.5rem'}}>
              <span>Tone: {classified.category} | Weight: {classified.priority_weight.toFixed(2)}</span>
            </div>
          )}
        </article>
      </section>

      <section className="subsection-nav">
        {DASHBOARD_SUBSECTIONS.map((section) => (
          <button
            key={section.id}
            type="button"
            className={activeSubsection === section.id ? 'active' : ''}
            onClick={() => setActiveSubsection(section.id)}
          >
            <strong>{section.label}</strong>
            <span>{section.description}</span>
          </button>
        ))}
      </section>

      <section className="subsection-head">
        <h3>{activeSubsectionMeta.label}</h3>
        <p>{activeSubsectionMeta.description}</p>
      </section>

      {activeSubsection === 'overview' && (
        <section className="content-grid">
          <article className="panel">
            <h3>⚠️ Live Risk Indicators</h3>
            <div className="risk-gauge-grid">
              <div className="risk-gauge-card" style={{borderLeftColor: '#ef4444'}}>
                <span className="risk-gauge-label">🌊 Flood Risk Index</span>
                <div className="risk-gauge-bar-wrap">
                  <div className="risk-gauge-bar" style={{width: `${Math.min((displayedScores?.flood_risk_index ?? 0) * 100, 100).toFixed(0)}%`, background: 'linear-gradient(90deg,#ef4444,#f97316)'}} />
                </div>
                <strong className="risk-gauge-val">{((displayedScores?.flood_risk_index ?? 0) * 100).toFixed(1)}%</strong>
              </div>
              <div className="risk-gauge-card" style={{borderLeftColor: '#3b82f6'}}>
                <span className="risk-gauge-label">🚰 Blocked Drains</span>
                <div className="risk-gauge-bar-wrap">
                  <div className="risk-gauge-bar" style={{width: `${Math.min((displayedScores?.blocked_drain_count ?? 0) / 30 * 100, 100).toFixed(0)}%`, background: 'linear-gradient(90deg,#3b82f6,#6366f1)'}} />
                </div>
                <strong className="risk-gauge-val">{displayedScores?.blocked_drain_count ?? 0} drains</strong>
              </div>
              <div className="risk-gauge-card" style={{borderLeftColor: '#10b981'}}>
                <span className="risk-gauge-label">🌳 Green Deficit</span>
                <div className="risk-gauge-bar-wrap">
                  <div className="risk-gauge-bar" style={{width: `${Math.min((displayedScores?.green_deficit_index ?? 0) * 100, 100).toFixed(0)}%`, background: 'linear-gradient(90deg,#10b981,#06b6d4)'}} />
                </div>
                <strong className="risk-gauge-val">{((displayedScores?.green_deficit_index ?? 0)).toFixed(3)}</strong>
              </div>
              <div className="risk-gauge-card" style={{borderLeftColor: '#f59e0b'}}>
                <span className="risk-gauge-label">🏘️ Informal Area</span>
                <div className="risk-gauge-bar-wrap">
                  <div className="risk-gauge-bar" style={{width: `${Math.min(displayedScores?.informal_area_pct ?? 0, 100).toFixed(0)}%`, background: 'linear-gradient(90deg,#f59e0b,#ef4444)'}} />
                </div>
                <strong className="risk-gauge-val">{(displayedScores?.informal_area_pct ?? 0).toFixed(1)}%</strong>
              </div>
              <div className="risk-gauge-card" style={{borderLeftColor: '#8b5cf6'}}>
                <span className="risk-gauge-label">👥 Exposed Population</span>
                <div className="risk-gauge-bar-wrap">
                  <div className="risk-gauge-bar" style={{width: `${Math.min((displayedScores?.exposed_population ?? 0) / Math.max(wardSummary?.population ?? 1, 1) * 100, 100).toFixed(0)}%`, background: 'linear-gradient(90deg,#8b5cf6,#ec4899)'}} />
                </div>
                <strong className="risk-gauge-val">{(displayedScores?.exposed_population ?? 0).toLocaleString()} people</strong>
              </div>
              <div className="risk-gauge-card" style={{borderLeftColor: '#06b6d4'}}>
                <span className="risk-gauge-label">📊 SDG-11 Score</span>
                <div className="risk-gauge-bar-wrap">
                  <div className="risk-gauge-bar" style={{width: `${Math.min((displayedScores?.sdg11_score ?? 0) * 10, 100).toFixed(0)}%`, background: 'linear-gradient(90deg,#06b6d4,#10b981)'}} />
                </div>
                <strong className="risk-gauge-val">{(displayedScores?.sdg11_score ?? 0).toFixed(2)} / 10</strong>
              </div>
            </div>
          </article>

          <article className="panel">
            <h3>🚨 Neighborhood Problems</h3>
            <div className="problem-list">
              {(digitalTwinScene?.problems ?? []).map((problem) => (
                <div key={problem.issue} className={`problem-item problem-sev-${problem.severity}`}>
                  <div className="problem-item-head">
                    <strong>{problem.issue}</strong>
                    <span className={`sev ${problem.severity}`}>{problem.severity}</span>
                  </div>
                  <p>{problem.summary}</p>
                </div>
              ))}
              {(digitalTwinScene?.problems ?? []).length === 0 && <p className="muted-text">No active problems detected.</p>}
            </div>
          </article>

          <article className="panel">
            <h3>✅ Actions Pipeline</h3>
            <div className="action-list">
              {(digitalTwinScene?.actions_taken ?? []).map((action) => (
                <div key={action.intervention_id} className="action-item">
                  <div className="action-head">
                    <strong>{action.title}</strong>
                    <span className="action-pct">{action.progress_pct}%</span>
                  </div>
                  <div className="action-progress-bar-wrap">
                    <div className="action-progress-bar" style={{width: `${action.progress_pct}%`}} />
                  </div>
                  <p>{action.agency} | {action.estimated_cost_lakh?.toFixed(2)} lakh | {action.expected_beneficiaries?.toLocaleString()} beneficiaries</p>
                </div>
              ))}
              {(digitalTwinScene?.actions_taken ?? []).length === 0 && <p className="muted-text">No actions in pipeline yet.</p>}
            </div>
          </article>

          <article className="panel">
            <h3>📋 Citizen Reports</h3>
            <div className="report-list-grid">
              {reports.slice(0, 6).map((r, i) => (
                <div key={i} className="report-card">
                  <div className="report-card-head">
                    <span className={`notif-sev ${r.category ?? 'info'}`}>{r.category ?? 'general'}</span>
                    <span className="report-weight">Priority: {r.priority_weight?.toFixed(2) ?? '—'}</span>
                  </div>
                  <p>{r.text?.substring(0, 90)}{r.text?.length > 90 ? '...' : ''}</p>
                </div>
              ))}
              {reports.length === 0 && <p className="muted-text">No reports yet for this ward.</p>}
            </div>
          </article>
        </section>
      )}
      {activeSubsection === 'prioritization' && (
        <section className="content-grid">
          <article className="panel">
            <h3>🏆 AI Policy Prioritization Output</h3>
            <p className="prescriptive-tagline">
              A trained ranking model scores micro-works by impact, cost, and feasibility.
              The list below is the optimal policy set for this ward.
            </p>
            {scenario ? (
              <div className="priority-table-wrap">
                {scenario.selected_reasoning.map((item, idx) => (
                  <div key={item.intervention_id} className="priority-row">
                    <div className="rank-badge" style={{background: idx === 0 ? '#f59e0b' : idx === 1 ? '#94a3b8' : idx === 2 ? '#b45309' : 'rgba(100,130,160,0.3)'}}>
                      #{idx + 1}
                    </div>
                    <div className="priority-row-main">
                      <div className="priority-row-title">
                        <strong>{item.title}</strong>
                        <span className="category-chip" style={{background: {
                          Drainage:'rgba(59,130,246,0.22)', Water:'rgba(6,182,212,0.22)',
                          Waste:'rgba(245,158,11,0.22)', Road:'rgba(139,92,246,0.22)',
                          Green:'rgba(16,185,129,0.22)', 'Public Safety':'rgba(239,68,68,0.22)'
                        }[item.category] ?? 'rgba(100,130,160,0.2)'}}>{item.category}</span>
                      </div>
                      <div className="priority-impact-bar-wrap">
                        <div className="priority-impact-bar" style={{
                          width: `${Math.min(item.utility_density / Math.max(...scenario.selected_reasoning.map(r => r.utility_density)) * 100, 100).toFixed(1)}%`
                        }} />
                      </div>
                      <div className="priority-row-meta">
                        <span>💰 {item.estimated_cost_lakh.toFixed(2)} lakh</span>
                        <span>👥 {item.expected_beneficiaries.toLocaleString()} beneficiaries</span>
                        <span>⚡ Score: {item.utility_density.toFixed(3)}</span>
                        <span>📅 {item.execution_months.toFixed(1)} mo</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="empty-state">
                <p>No simulation data. Run a <strong>Budget Simulation</strong> in the "Budget & Scenario" tab first.</p>
              </div>
            )}
          </article>

          <article className="panel">
            <h3>🧠 AI Decision Justifications</h3>
            <div className="reasoning-list">
              {scenario ? (
                scenario.selected_reasoning.map((item) => (
                  <article key={item.intervention_id} className="reasoning-item">
                    <div className="reasoning-head">
                      <strong>{item.title}</strong>
                      <span className="tiny-note">{item.category} | {item.execution_months.toFixed(1)} months</span>
                    </div>
                    <ul className="reason-bullets">
                      {(item.reasons ?? []).map((reason, ridx) => (
                        <li key={ridx}>✓ {reason}</li>
                      ))}
                    </ul>
                    <p className="tiny-note">Efficiency: {item.utility_density.toFixed(4)} impact per lakh</p>
                  </article>
                ))
              ) : (
                <p className="muted-text">Awaiting simulation...</p>
              )}
            </div>
          </article>
        </section>
      )}
      {activeSubsection === 'simulation' && (
        <>
          {/* ─── Budget Input Panel ─── */}
          <section className="content-grid">
            <article className="panel budget-input-panel">
              <div className="prescriptive-head">
                <h3>🤖 AI Budget Sector Allocator</h3>
                <p className="prescriptive-tagline">
                  Enter your total ward budget and the AI will analyse live ward indicators — flood risk,
                  drain blockage, green deficit, informal settlement density — to predict the optimal
                  spending distribution across all 6 urban service sectors.
                </p>
              </div>
              <div className="budget-input-row">
                <div className="budget-input-wrap">
                  <span className="budget-currency">BDT</span>
                  <input
                    id="budget-amount-input"
                    type="number"
                    className="budget-input-big"
                    min="1"
                    max="5000"
                    step="0.5"
                    value={budgetInput}
                    onChange={(e) => setBudgetInput(Number(e.target.value))}
                    onKeyDown={(e) => { if (e.key === 'Enter') runAiBudgetPlan() }}
                  />
                  <span className="budget-unit">lakh</span>
                </div>
                <button
                  className="predict-btn"
                  onClick={runAiBudgetPlan}
                  disabled={isPredicting}
                >
                  {isPredicting ? '🔄 AI Predicting...' : '✨ Predict Optimal Allocation →'}
                </button>
              </div>
            </article>

            <article className="panel">
              <h3>📋 Ward Risk Profile</h3>
              <p className="prescriptive-tagline">Live indicators feeding the AI model for this ward.</p>
              <div className="indicator-mini-grid">
                <div className="ind-mini"><span>Blocked Drains</span><strong>{displayedScores?.blocked_drain_count ?? '—'}</strong></div>
                <div className="ind-mini"><span>Flood Risk</span><strong>{((displayedScores?.flood_risk_index ?? 0)*100).toFixed(1)}%</strong></div>
                <div className="ind-mini"><span>Green Deficit</span><strong>{(displayedScores?.green_deficit_index ?? 0).toFixed(3)}</strong></div>
                <div className="ind-mini"><span>Informal Area</span><strong>{(displayedScores?.informal_area_pct ?? 0).toFixed(1)}%</strong></div>
                <div className="ind-mini"><span>Exposed Pop.</span><strong>{(displayedScores?.exposed_population ?? 0).toLocaleString()}</strong></div>
                <div className="ind-mini"><span>SDG-11 Score</span><strong>{(displayedScores?.sdg11_score ?? 0).toFixed(2)}</strong></div>
              </div>
            </article>
          </section>

          {/* ─── AI Prediction Results ─── */}
          {aiBudgetPlan && (
            <section className="content-grid simulation-wide">
              <article className="panel sector-dist-panel">
                <h3>📊 AI-Predicted Sector Distribution</h3>
                <p className="prescriptive-tagline" style={{marginBottom:'1.25rem'}}>
                  Budget: <strong>BDT {aiBudgetPlan.ward_budget_lakh} lakh</strong> across {aiBudgetPlan.sectors.length} sectors,
                  projected for <strong>{aiBudgetPlan.projected_households?.toLocaleString()}</strong> households.
                </p>
                <div className="sector-bars">
                  {aiBudgetPlan.sectors.map((sector) => (
                    <div key={sector.name} className="sector-bar-row">
                      <div className="sector-bar-label">
                        <span className="sector-icon">{sector.icon}</span>
                        <span className="sector-name">{sector.name}</span>
                      </div>
                      <div className="sector-bar-track">
                        <div
                          className="sector-bar-fill"
                          style={{
                            width: `${sector.allocation_pct}%`,
                            background: `linear-gradient(90deg, ${sector.color}cc, ${sector.color}66)`
                          }}
                        />
                      </div>
                      <div className="sector-bar-amounts">
                        <strong style={{color: sector.color}}>{sector.allocation_pct}%</strong>
                        <span>{sector.allocation_lakh.toFixed(2)} lakh</span>
                      </div>
                      <p className="sector-rationale">{sector.rationale}</p>
                      <p className="sector-outcome">→ {sector.expected_outcome}</p>
                    </div>
                  ))}
                </div>
              </article>

              <article className="panel">
                <h3>🤖 AI Summary &amp; Outcome Forecast</h3>
                <div className="ai-summary-card">
                  <div className="ai-summary-icon">🧠</div>
                  <p className="ai-summary-text">{aiBudgetPlan.ai_summary}</p>
                </div>
                <div className="summary-strip" style={{marginTop:'1.25rem'}}>
                  <div className="summary-card">
                    <span>Total Budget</span>
                    <strong>{aiBudgetPlan.ward_budget_lakh} lakh</strong>
                  </div>
                  <div className="summary-card">
                    <span>SDG-11 Gain</span>
                    <strong>+{aiBudgetPlan.estimated_sdg11_gain?.toFixed(2)}</strong>
                  </div>
                  <div className="summary-card">
                    <span>Households</span>
                    <strong>{aiBudgetPlan.projected_households?.toLocaleString()}</strong>
                  </div>
                </div>
                <div style={{marginTop:'1.25rem', display: 'flex', flexDirection: 'column', gap: '0.75rem'}}>
                  <h4 style={{margin:'0',fontSize:'0.88rem',letterSpacing:'0.02em'}}>Top Sector Breakdown</h4>
                  <div style={{display:'flex', flexDirection:'column', gap:'0.5rem'}}>
                    {aiBudgetPlan.sectors.slice(0,3).map(s => (
                      <div key={s.name} style={{display:'flex',alignItems:'center',gap:'0.6rem'}}>
                        <span style={{fontSize:'1.3rem'}}>{s.icon}</span>
                        <div style={{flex:1}}>
                          <div style={{fontSize:'0.82rem',fontWeight:700,color:s.color}}>{s.name}</div>
                          <div style={{fontSize:'0.76rem',opacity:0.75}}>{s.expected_outcome}</div>
                        </div>
                        <strong style={{color:s.color,fontSize:'0.88rem',whiteSpace:'nowrap'}}>{s.allocation_pct}%</strong>
                      </div>
                    ))}
                  </div>
                  
                  <button 
                    className="predict-btn" 
                    style={{marginTop:'0.5rem', width:'100%', justifyContent:'center'}}
                    onClick={() => downloadExport(
                      `/api/exports/wards/${selectedWard}/policy-memo.pdf?budget_lakh=${budgetInput}&strategy=${planningStrategy}`,
                      `ward_${selectedWard}_ai_policy_memo.pdf`
                    )}
                  >
                    📥 Export AI Policy Memo (PDF)
                  </button>
                </div>
              </article>
            </section>
          )}

          {/* ─── Legacy Roadmap (kept if old scenario run exists) ─── */}
          {scenario && (
            <section className="content-grid">
              <article className="panel">
                <h3>🗓️ Agency Execution Roadmap</h3>
                <div className="roadmap-list">
                  {(scenario?.implementation_roadmap ?? []).slice(0, 6).map((step) => (
                    <article key={step.intervention_id} className="roadmap-item">
                      <div className="workflow-stage-head">
                        <strong>{step.title}</strong>
                        <span className={`sev ${step.delivery_status}`}>{step.phase}</span>
                      </div>
                      <p className="roadmap-meta">
                        {step.agency} | Month {step.start_month} → {step.end_month}
                      </p>
                    </article>
                  ))}
                </div>
              </article>
              <article className="panel">
                <h3>⚡ Tradeoff Alerts</h3>
                <div className="alert-list">
                  {(scenario.tradeoff_alerts ?? []).map((alert, index) => (
                    <div key={`${alert.topic}-${index}`} className={`alert-item ${alert.severity}`}>
                      <strong>{alert.topic}</strong>
                      <p>{alert.message}</p>
                    </div>
                  ))}
                </div>
              </article>
            </section>
          )}
        </>
      )}

      {activeSubsection === 'governance' && (
        <>
          <section className="content-grid">
            <article className="panel">
              <h3>SDG-11 Governance Card</h3>
              {sdgCard ? (
                <>
                  <p>
                    Overall score: <strong>{sdgCard.overall_score.toFixed(2)}</strong>
                  </p>
                  <div className="problem-list">
                    {sdgCard.targets.map((target) => (
                      <div key={target.target} className="problem-item">
                        <strong>{target.target}</strong>
                        <span className={`sev ${target.status}`}>{target.status}</span>
                        <p>
                          Score {target.score.toFixed(1)} | {target.evidence}
                        </p>
                      </div>
                    ))}
                  </div>
                  <p>{sdgCard.priority_message}</p>
                </>
              ) : (
                <p>Loading governance card...</p>
              )}
            </article>

            <article className="panel">
              <h3>Policy Brief & Export</h3>
              <div className="row-actions">
                <button
                  disabled={user.role !== 'planner'}
                  onClick={() =>
                    downloadExport(
                      `/api/exports/wards/${selectedWard}/policy-memo.pdf?budget_lakh=${budget}&strategy=${planningStrategy}`,
                      `ward_${selectedWard}_policy_memo.pdf`
                    )
                  }
                >
                  Export PDF Policy Memo
                </button>
              </div>
            </article>
          </section>

        </>
      )}

      {error && <p className="status err">{error}</p>}
      {info && <p className="status ok">{info}</p>}
    </main>
  )
}

export default App
