// The only place that talks to the backend. The UI never calls a model provider.

async function request(url, options) {
  const response = await fetch(url, options)
  const data = await response.json()
  return { ok: response.ok, data }
}

export function getProviders() {
  return request('/api/providers')
}

export function getRuns() {
  return request('/api/runs?limit=50')
}

export function sendRun(body) {
  return request('/api/run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
}

// An error answer has either a text "detail" or (for bad input) a list of problems.
export function errorText(data) {
  if (typeof data.detail === 'string') return data.detail
  if (Array.isArray(data.detail)) return data.detail.map((problem) => problem.msg).join('; ')
  return 'Unknown error'
}
