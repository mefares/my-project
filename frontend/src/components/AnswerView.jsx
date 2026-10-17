// Shows the last send: the answer so far while it streams, then the final answer (or the error)
// and its measurements.

export default function AnswerView({ loading, partial, run, error }) {
  const finished = run && run.answer !== null

  return (
    <section className="panel">
      <h2>Answer</h2>

      {loading && partial === '' && <p className="note">Waiting for the model...</p>}
      {error && <p className="error">{error}</p>}

      {/* While streaming (or after a failure midway) show what arrived; once done, the saved answer. */}
      {!finished && partial !== '' && <pre className="answer">{partial}</pre>}
      {finished && <pre className="answer">{run.answer || '(empty answer)'}</pre>}

      {finished && !loading && (
        <p className="stats">
          run #{run.id} | {run.model} | tokens in {run.input_tokens}, out {run.output_tokens} |{' '}
          {run.ttft_ms !== null && <>first token after {run.ttft_ms} ms | </>}
          total {run.latency_ms} ms | finish: {run.finish_reason}
        </p>
      )}

      {!loading && !run && !error && partial === '' && (
        <p className="note">Send a question to see the answer here.</p>
      )}
    </section>
  )
}
