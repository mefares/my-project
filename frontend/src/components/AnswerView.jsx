// Shows the result of the last send: the answer (or the error) and its measurements.

export default function AnswerView({ loading, run, error }) {
  return (
    <section className="panel">
      <h2>Answer</h2>

      {loading && <p className="note">Waiting for the model...</p>}
      {error && <p className="error">{error}</p>}

      {!loading && run && run.answer !== null && (
        <>
          <pre className="answer">{run.answer || '(empty answer)'}</pre>
          <p className="stats">
            run #{run.id} | {run.model} | tokens in {run.input_tokens}, out {run.output_tokens} |{' '}
            {run.latency_ms} ms | finish: {run.finish_reason}
          </p>
        </>
      )}

      {!loading && !run && !error && <p className="note">Send a question to see the answer here.</p>}
    </section>
  )
}
