// The settings form. It only shows and edits values; App.jsx owns the state.
// An empty number box means "do not send this setting": the provider uses its own default.

export default function SettingsPanel({ providers, settings, onChange }) {
  const info = providers[settings.provider]
  const ignored = info ? info.ignored : []

  function hint(name) {
    return ignored.includes(name) ? ` (ignored by ${info.model})` : ''
  }

  return (
    <section className="panel">
      <h2>Settings</h2>

      <label>
        Provider
        <select
          value={settings.provider}
          onChange={(event) => onChange('provider', event.target.value)}
        >
          {Object.keys(providers).map((name) => (
            <option key={name} value={name}>
              {name} ({providers[name].model})
            </option>
          ))}
        </select>
      </label>

      <label>
        System prompt
        <textarea
          rows={4}
          placeholder="Optional. Example: Answer in exactly two sentences."
          value={settings.systemPrompt}
          onChange={(event) => onChange('systemPrompt', event.target.value)}
        />
      </label>

      <label>
        Temperature{hint('temperature')}
        <input
          type="number"
          step="0.1"
          min="0"
          max={info ? info.temperature_max : undefined}
          placeholder={`default (0 to ${info ? info.temperature_max : '?'})`}
          value={settings.temperature}
          onChange={(event) => onChange('temperature', event.target.value)}
        />
      </label>

      <label>
        Top p{hint('top_p')}
        <input
          type="number"
          step="0.1"
          min="0"
          max="1"
          placeholder="default (0 to 1)"
          value={settings.topP}
          onChange={(event) => onChange('topP', event.target.value)}
        />
      </label>

      <label>
        Max tokens
        <input
          type="number"
          step="1"
          min="1"
          placeholder="default"
          value={settings.maxTokens}
          onChange={(event) => onChange('maxTokens', event.target.value)}
        />
      </label>

      {settings.provider === 'mistral' && (
        <p className="note">
          Mistral advises changing temperature or top p, not both. With temperature 0, top p must be
          empty or 1.
        </p>
      )}
      {settings.provider === 'gemini' && (
        <p className="note">
          On Gemini, thinking tokens count toward max tokens, so a small value can give a cut-off or
          empty answer.
        </p>
      )}
    </section>
  )
}
