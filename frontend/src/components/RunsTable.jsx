// The latest saved runs. Empty cells mean the setting was not sent (provider default).

function show(value) {
  return value === null || value === undefined ? '' : value
}

function shorten(text, length) {
  if (!text) return ''
  return text.length > length ? text.slice(0, length) + '...' : text
}

export default function RunsTable({ runs }) {
  return (
    <section className="panel">
      <h2>Saved runs</h2>
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>Time (UTC)</th>
              <th>Provider</th>
              <th>Temp</th>
              <th>Top p</th>
              <th>Max tok</th>
              <th>In</th>
              <th>Out</th>
              <th>ms</th>
              <th>Finish</th>
              <th>Question</th>
              <th>Answer / error</th>
            </tr>
          </thead>
          <tbody>
            {runs.map((run) => (
              <tr key={run.id} className={run.error ? 'failed' : ''}>
                <td>{run.id}</td>
                <td>{run.created_at.replace('T', ' ').slice(0, 19)}</td>
                <td>{run.provider}</td>
                <td>{show(run.temperature)}</td>
                <td>{show(run.top_p)}</td>
                <td>{show(run.max_tokens)}</td>
                <td>{show(run.input_tokens)}</td>
                <td>{show(run.output_tokens)}</td>
                <td>{show(run.latency_ms)}</td>
                <td>{show(run.finish_reason)}</td>
                <td title={run.question}>{shorten(run.question, 40)}</td>
                <td title={run.error || run.answer}>{shorten(run.error || run.answer, 60)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {runs.length === 0 && <p className="note">No runs yet.</p>}
      </div>
    </section>
  )
}
