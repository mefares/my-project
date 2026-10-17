import { useEffect, useState } from 'react'
import { errorText, getProviders, getRuns, sendRun, sendRunStream } from './api.js'
import SettingsPanel from './components/SettingsPanel.jsx'
import AnswerView from './components/AnswerView.jsx'
import RunsTable from './components/RunsTable.jsx'

// An empty string means "not set": the setting is not sent and the provider uses its default.
const START_SETTINGS = {
  provider: 'mistral',
  systemPrompt: '',
  temperature: '',
  topP: '',
  maxTokens: '',
}

function numberOrNull(text) {
  return text === '' ? null : Number(text)
}

export default function App() {
  const [providers, setProviders] = useState({})
  const [settings, setSettings] = useState(START_SETTINGS)
  const [question, setQuestion] = useState('')
  const [stream, setStream] = useState(true) // show the answer as it is written
  const [loading, setLoading] = useState(false)
  const [partial, setPartial] = useState('') // the answer so far, while streaming
  const [run, setRun] = useState(null)
  const [error, setError] = useState('')
  const [runs, setRuns] = useState([])

  function loadRuns() {
    getRuns().then((result) => result.ok && setRuns(result.data))
  }

  // Runs once when the page opens.
  useEffect(() => {
    getProviders()
      .then((result) => result.ok && setProviders(result.data))
      .catch(() => setError('Cannot reach the backend. Is uvicorn running on port 8000?'))
    loadRuns()
  }, [])

  function changeSetting(name, value) {
    setSettings({ ...settings, [name]: value })
  }

  async function send() {
    setLoading(true)
    setError('')
    setRun(null)
    setPartial('')
    try {
      const body = {
        provider: settings.provider,
        system_prompt: settings.systemPrompt,
        question,
        temperature: numberOrNull(settings.temperature),
        top_p: numberOrNull(settings.topP),
        max_tokens: numberOrNull(settings.maxTokens),
      }
      const result = stream
        ? await sendRunStream(body, (text) => setPartial((soFar) => soFar + text))
        : await sendRun(body)
      if (result.ok) {
        setRun(result.data)
      } else {
        setError(errorText(result.data))
      }
    } catch {
      setError('Cannot reach the backend. Is uvicorn running on port 8000?')
    }
    setLoading(false)
    loadRuns() // a failed send is saved too, so the table is refreshed either way
  }

  return (
    <main>
      <h1>LLM playground</h1>
      <div className="columns">
        <SettingsPanel providers={providers} settings={settings} onChange={changeSetting} />
        <div className="main-column">
          <section className="panel">
            <h2>Question</h2>
            <textarea
              rows={4}
              placeholder="Type your question..."
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
            />
            <div className="send-row">
              <button onClick={send} disabled={loading || question.trim() === ''}>
                {loading ? 'Sending...' : 'Send'}
              </button>
              <label className="inline">
                <input type="checkbox" checked={stream} onChange={(event) => setStream(event.target.checked)} />
                Stream the answer
              </label>
            </div>
          </section>
          <AnswerView loading={loading} partial={partial} run={run} error={error} />
        </div>
      </div>
      <RunsTable runs={runs} />
    </main>
  )
}
