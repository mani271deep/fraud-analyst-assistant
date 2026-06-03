import { useEffect, useMemo, useState } from 'react'
import './App.css'

const API_URL = 'http://127.0.0.1:8000/investigate'

function probabilityTone(probability) {
  if (probability > 0.7) return 'danger'
  if (probability >= 0.4) return 'warning'
  return 'safe'
}

function actionTone(action) {
  if (action === 'DECLINE') return 'danger'
  if (action === 'ESCALATE') return 'warning'
  return 'safe'
}

function formatPercent(value) {
  if (typeof value !== 'number' || Number.isNaN(value)) return '--'
  return `${(value * 100).toFixed(1)}%`
}

function formatNumber(value) {
  if (typeof value !== 'number' || Number.isNaN(value)) return value
  return Number.parseFloat(value.toFixed(6)).toString()
}

function App() {
  const [alerts, setAlerts] = useState([])
  const [selectedAlertId, setSelectedAlertId] = useState(null)
  const [investigation, setInvestigation] = useState(null)
  const [loadingAlerts, setLoadingAlerts] = useState(true)
  const [loadingInvestigation, setLoadingInvestigation] = useState(false)
  const [error, setError] = useState('')
  const [composerOpen, setComposerOpen] = useState(false)
  const [customJson, setCustomJson] = useState('')
  const [decision, setDecision] = useState(null)

  useEffect(() => {
    let active = true

    async function loadAlerts() {
      try {
        const response = await fetch('/sample_alerts.json')
        if (!response.ok) {
          throw new Error(`Unable to load sample alerts (${response.status})`)
        }
        const data = await response.json()
        if (!Array.isArray(data)) {
          throw new Error('Sample alerts file must contain a JSON array')
        }

        if (active) {
          setAlerts(data)
          setSelectedAlertId(data[0]?.id ?? null)
          setError('')
        }
      } catch (loadError) {
        if (active) setError(loadError.message)
      } finally {
        if (active) setLoadingAlerts(false)
      }
    }

    loadAlerts()

    return () => {
      active = false
    }
  }, [])

  const selectedAlert = useMemo(
    () => alerts.find((alert) => alert.id === selectedAlertId) ?? null,
    [alerts, selectedAlertId],
  )

  useEffect(() => {
    if (!selectedAlert) {
      return
    }

    const controller = new AbortController()

    async function investigateAlert() {
      setLoadingInvestigation(true)
      setInvestigation(null)
      setDecision(null)
      setError('')

      try {
        const response = await fetch(API_URL, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(selectedAlert.transaction),
          signal: controller.signal,
        })

        const payload = await response.json().catch(() => null)
        if (!response.ok) {
          const detail = payload?.detail
          throw new Error(
            typeof detail === 'string'
              ? detail
              : `Investigation failed (${response.status})`,
          )
        }

        setInvestigation(payload)
      } catch (investigationError) {
        if (investigationError.name !== 'AbortError') {
          setError(investigationError.message)
        }
      } finally {
        if (!controller.signal.aborted) {
          setLoadingInvestigation(false)
        }
      }
    }

    investigateAlert()

    return () => controller.abort()
  }, [selectedAlert])

  function selectAlert(alertId) {
    setSelectedAlertId(alertId)
  }

  function addCustomAlert(event) {
    event.preventDefault()

    try {
      const transaction = JSON.parse(customJson)
      if (
        !transaction ||
        Array.isArray(transaction) ||
        typeof transaction !== 'object'
      ) {
        throw new Error('Paste one transaction JSON object')
      }

      const nextId = `CUSTOM-${String(alerts.length + 1).padStart(3, '0')}`
      const nextAlert = { id: nextId, transaction }
      setAlerts((currentAlerts) => [nextAlert, ...currentAlerts])
      setSelectedAlertId(nextId)
      setCustomJson('')
      setComposerOpen(false)
      setError('')
    } catch (parseError) {
      setError(parseError.message)
    }
  }

  function renderDecisionNote() {
    if (!decision || !investigation?.recommended_action) return null

    const recommendedAction = investigation.recommended_action
    const agreed = decision === recommendedAction

    return (
      <p className={`decision-note ${agreed ? 'agreed' : 'overrode'}`}>
        {agreed
          ? 'You agreed with the assistant'
          : `You overrode the assistant's ${recommendedAction} recommendation`}
      </p>
    )
  }

  return (
    <main className="dashboard-shell">
      <header className="app-header">
        <div>
          <p className="eyebrow">Case operations</p>
          <h1>Fraud Analyst Assistant</h1>
        </div>
        <div className="header-metrics" aria-label="Alert summary">
          <span>{alerts.length} alerts</span>
          <span>{selectedAlert?.id ?? 'No alert selected'}</span>
        </div>
      </header>

      {error ? <div className="error-banner">{error}</div> : null}

      <section className="workspace-grid">
        <aside className="queue-panel" aria-label="Alert queue">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">Queue</p>
              <h2>Alerts</h2>
            </div>
            <button
              type="button"
              className="secondary-button"
              onClick={() => setComposerOpen((open) => !open)}
            >
              Paste a transaction
            </button>
          </div>

          {composerOpen ? (
            <form className="paste-panel" onSubmit={addCustomAlert}>
              <textarea
                value={customJson}
                onChange={(event) => setCustomJson(event.target.value)}
                placeholder='{"Time": 406.0, "V1": -2.31, ...}'
                spellCheck="false"
              />
              <div className="paste-actions">
                <button type="submit" className="primary-button">
                  Add alert
                </button>
                <button
                  type="button"
                  className="ghost-button"
                  onClick={() => setComposerOpen(false)}
                >
                  Cancel
                </button>
              </div>
            </form>
          ) : null}

          <div className="alert-list">
            {loadingAlerts ? (
              <div className="empty-state">Loading sample alerts...</div>
            ) : null}

            {!loadingAlerts && alerts.length === 0 ? (
              <div className="empty-state">No alerts available.</div>
            ) : null}

            {alerts.map((alert) => (
              <button
                type="button"
                key={alert.id}
                className={`alert-card ${
                  alert.id === selectedAlertId ? 'selected' : ''
                }`}
                onClick={() => selectAlert(alert.id)}
              >
                <span className="alert-id">{alert.id}</span>
                <span className="alert-meta">
                  {Object.keys(alert.transaction ?? {}).length} features
                </span>
              </button>
            ))}
          </div>
        </aside>

        <section className="detail-panel" aria-label="Investigation detail">
          {!selectedAlert ? (
            <div className="empty-detail">
              Select an alert to begin the investigation.
            </div>
          ) : null}

          {selectedAlert ? (
            <>
              <div className="detail-header">
                <div>
                  <p className="eyebrow">Investigation</p>
                  <h2>{selectedAlert.id}</h2>
                </div>
                {investigation ? (
                  <span
                    className={`probability-badge ${probabilityTone(
                      investigation.fraud_probability,
                    )}`}
                  >
                    {formatPercent(investigation.fraud_probability)}
                  </span>
                ) : null}
              </div>

              {loadingInvestigation ? (
                <div className="loading-card">
                  <span className="loader" />
                  Investigating transaction...
                </div>
              ) : null}

              {!loadingInvestigation && investigation ? (
                <div className="investigation-stack">
                  <section className="summary-section">
                    <div
                      className={`recommendation ${actionTone(
                        investigation.recommended_action,
                      )}`}
                    >
                      <span>Recommended action</span>
                      <strong>{investigation.recommended_action}</strong>
                    </div>
                    <p className="explanation">{investigation.explanation}</p>
                  </section>

                  <section className="action-section">
                    <div className="action-buttons">
                      {['APPROVE', 'DECLINE', 'ESCALATE'].map((action) => (
                        <button
                          type="button"
                          key={action}
                          className={`action-button ${actionTone(action)} ${
                            decision === action ? 'chosen' : ''
                          }`}
                          onClick={() => setDecision(action)}
                        >
                          {action.charAt(0) + action.slice(1).toLowerCase()}
                        </button>
                      ))}
                    </div>
                    {renderDecisionNote()}
                  </section>

                  <section className="evidence-section">
                    <div className="section-title">
                      <p className="eyebrow">Evidence</p>
                      <h3>SHAP features</h3>
                    </div>
                    <div className="feature-table">
                      <div className="feature-row table-head">
                        <span>Feature</span>
                        <span>Value</span>
                        <span>Contribution</span>
                      </div>
                      {investigation.evidence?.shap_features?.map((feature) => (
                        <div className="feature-row" key={feature.feature}>
                          <span>{feature.feature}</span>
                          <span>{formatNumber(feature.value)}</span>
                          <span>{formatNumber(feature.shap_contribution)}</span>
                        </div>
                      ))}
                    </div>
                  </section>

                  <section className="evidence-section">
                    <div className="section-title">
                      <p className="eyebrow">Evidence</p>
                      <h3>Retrieved policies</h3>
                    </div>
                    <div className="policy-list">
                      {investigation.evidence?.retrieved_policies?.map(
                        (policy) => (
                          <article className="policy-card" key={policy.id}>
                            <strong>{policy.id}</strong>
                            <p>{policy.text}</p>
                          </article>
                        ),
                      )}
                    </div>
                  </section>
                </div>
              ) : null}
            </>
          ) : null}
        </section>
      </section>
    </main>
  )
}

export default App
