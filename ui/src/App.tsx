import { useState } from 'react'
import './App.css'

interface Citation { source: number; quote: string }
interface Source { id: number; header: string; page: number; acronym: string; version?: string }
interface Answer { answerable: boolean; answer: string; citations: Citation[]; sources: Source[] }

const EXAMPLES = [
  "What is required for XMI Schema Compliance?",
  "What is a Case File Item in CMMN?",
  "What is the MOF Extent?",
  "What is DMN?",
  "What does OCL say about null values?",
]

export default function App() {
  const [question, setQuestion] = useState('')
  const [result, setResult] = useState<Answer | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  async function ask(q?: string) {
    const text = (q ?? question).trim()
    if (!text) return
    setQuestion(text)
    setLoading(true)
    setError('')
    setResult(null)
    try {
      const res = await fetch('https://omg-specs-rag.onrender.com/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: text }),
      })
      if (!res.ok) throw new Error('Request failed')
      setResult(await res.json())
    } catch {
      setError('Could not reach the API. Is it running on port 8000?')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <header>
        <h1>OMG Specs Assistant</h1>
        <p className="subtitle">Answers with citations from XMI, BPMN, CMMN, MOF, OCL, SysML, DMN, and fUML</p>
      </header>

      <main>
        <div className="input-box">
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && (e.preventDefault(), ask())}
            rows={3}
            placeholder="Ask a question about an OMG specification..."
          />
          <button onClick={() => ask()} disabled={loading || !question.trim()}>
            {loading ? 'Searching...' : 'Ask'}
          </button>
        </div>

        <div className="examples">
          <span>Try:</span>
          {EXAMPLES.map((ex) => (
            <button key={ex} className="example" onClick={() => ask(ex)} disabled={loading}>
              {ex}
            </button>
          ))}
        </div>

        {error && <div className="error">{error}</div>}

        {result && (
          <div className="result">
            <section className="answer-card">
              <h2>Answer</h2>
              {!result.answerable ? (
                <p className="not-found">Not found in the indexed specifications.</p>
              ) : (
                <p className="answer-text">{result.answer}</p>
              )}
            </section>

            {result.citations?.length > 0 && (
              <section>
                <h3>Citations</h3>
                <div className="citations">
                  {result.citations.map((c, i) => (
                    <blockquote key={i} className="citation">
                      <span className="cite-id">[S{c.source}]</span>
                      <p>{c.quote}</p>
                    </blockquote>
                  ))}
                </div>
              </section>
            )}

            {result.sources?.length > 0 && (
              <section>
                <h3>Sources</h3>
                <ul className="sources">
                  {result.sources.map((s) => (
                    <li key={s.id}>
                      <span className="spec">{s.acronym} {s.version || ''}</span>
                      <span className="header">{s.header}</span>
                      <span className="page">p.{s.page}</span>
                    </li>
                  ))}
                </ul>
              </section>
            )}
          </div>
        )}
      </main>

      <footer>
        Local Mode A · Citations link to indexed OMG specifications · Answers refuse when not found
      </footer>
    </div>
  )
}
