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

// Same as sendRun, but the answer arrives piece by piece: onDelta(text) is called for each piece.
// The server sends one JSON event per line: {"type":"delta"}, then {"type":"done"} or {"type":"error"}.
export async function sendRunStream(body, onDelta) {
  const response = await fetch('/api/run/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!response.ok) {
    // Refused before the stream started (for example a setting out of range).
    return { ok: false, data: await response.json() }
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let lastEvent = null
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() // the last piece may be an unfinished line: keep it for the next read
    for (const line of lines) {
      if (line === '') continue
      const event = JSON.parse(line)
      if (event.type === 'delta') {
        onDelta(event.text)
      } else {
        lastEvent = event
      }
    }
  }

  if (lastEvent && lastEvent.type === 'done') return { ok: true, data: lastEvent.run }
  if (lastEvent && lastEvent.type === 'error') return { ok: false, data: { detail: lastEvent.detail } }
  return { ok: false, data: { detail: 'The answer stopped before the end.' } }
}

// An error answer has either a text "detail" or (for bad input) a list of problems.
export function errorText(data) {
  if (typeof data.detail === 'string') return data.detail
  if (Array.isArray(data.detail)) return data.detail.map((problem) => problem.msg).join('; ')
  return 'Unknown error'
}
