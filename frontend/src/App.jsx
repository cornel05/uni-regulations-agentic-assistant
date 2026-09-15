import React, { useEffect, useMemo, useRef, useState } from 'react'

import { clearSession, fetchDomains, sendMessage } from './api.js'

const BAND_LABELS = {
  HIGH: 'Độ tin cậy cao / High confidence',
  MEDIUM: 'Độ tin cậy trung bình / Medium confidence',
  LOW: 'Độ tin cậy thấp / Low confidence',
}

const EXAMPLES = [
  'Một học kỳ được đăng ký tối đa bao nhiêu tín chỉ?',
  'GPA tối thiểu để giữ học bổng là bao nhiêu?',
  'How do I apply for a leave of absence?',
]

export default function App() {
  const [domains, setDomains] = useState([])
  const [domain, setDomain] = useState(null)
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [sessionId, setSessionId] = useState(null)
  const [pending, setPending] = useState(false)
  const [error, setError] = useState('')
  const endRef = useRef(null)

  useEffect(() => {
    fetchDomains()
      .then(setDomains)
      .catch(() => setDomains([]))
  }, [])

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, pending])

  const canSend = useMemo(() => input.trim().length > 0 && !pending, [input, pending])

  async function submit(text) {
    const message = (text ?? input).trim()
    if (!message || pending) return

    setMessages((current) => [...current, { role: 'user', content: message }])
    setInput('')
    setError('')
    setPending(true)

    try {
      const answer = await sendMessage({ message, sessionId, domain })
      setSessionId(answer.session_id)
      setMessages((current) => [
        ...current,
        {
          role: 'assistant',
          content: answer.answer,
          confidence: answer.confidence,
          sources: answer.sources,
          disclaimer: answer.disclaimer,
          language: answer.language,
        },
      ])
    } catch (exc) {
      setError(exc.message)
    } finally {
      setPending(false)
    }
  }

  async function reset() {
    if (sessionId) {
      // Best effort: the local transcript clears either way.
      await clearSession(sessionId).catch(() => {})
    }
    setSessionId(null)
    setMessages([])
    setError('')
  }

  function onKeyDown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      submit()
    }
  }

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          Trợ lý Quy chế
          <small>Uni-Regulations Assistant</small>
        </div>

        <div>
          <div className="group-label">Lĩnh vực / Domain</div>
          <div className="domain-list">
            <button
              type="button"
              className="domain-option"
              aria-pressed={domain === null}
              onClick={() => setDomain(null)}
            >
              Tất cả
              <span>All domains</span>
            </button>
            {domains.map((item) => (
              <button
                key={item.value}
                type="button"
                className="domain-option"
                aria-pressed={domain === item.value}
                onClick={() => setDomain(item.value)}
              >
                {item.label_vi}
                <span>{item.label_en}</span>
              </button>
            ))}
          </div>
        </div>

        <div className="sidebar-footer">
          <button
            type="button"
            className="secondary"
            onClick={reset}
            disabled={!messages.length && !sessionId}
          >
            Xoá hội thoại / Clear history
          </button>
          <a href="#admin">Quản trị / Admin</a>
        </div>
      </aside>

      <main className="main">
        <div className="transcript">
          {messages.length === 0 && !pending ? (
            <div className="empty-state">
              <h1>Hỏi về quy chế của trường</h1>
              <p>
                Trả lời kèm trích dẫn điều khoản, số trang và mức độ tin cậy. Hỏi bằng
                tiếng Việt hoặc tiếng Anh.
                <br />
                Ask in Vietnamese or English — every answer cites its source.
              </p>
              <div className="examples">
                {EXAMPLES.map((example) => (
                  <button
                    key={example}
                    type="button"
                    className="example"
                    onClick={() => submit(example)}
                  >
                    {example}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            messages.map((message, index) =>
              message.role === 'user' ? (
                <div key={index} className="turn user">
                  <div className="bubble">{message.content}</div>
                </div>
              ) : (
                <Answer key={index} message={message} />
              ),
            )
          )}
          {pending && <div className="typing">Đang tra cứu quy chế… / Searching…</div>}
          <div ref={endRef} />
        </div>

        {error && <div className="error">{error}</div>}

        <form
          className="composer"
          onSubmit={(event) => {
            event.preventDefault()
            submit()
          }}
        >
          <textarea
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={onKeyDown}
            placeholder="Nhập câu hỏi của bạn… / Type your question…"
            maxLength={2000}
            rows={1}
            aria-label="Câu hỏi / Question"
          />
          <button type="submit" className="primary" disabled={!canSend}>
            Gửi / Send
          </button>
        </form>
      </main>
    </div>
  )
}

function Answer({ message }) {
  const band = message.confidence?.band
  const percent = Math.round((message.confidence?.score ?? 0) * 100)

  return (
    <div className="turn">
      <div className="meta-row">
        {band && (
          <span className={`badge ${band}`} title={BAND_LABELS[band]}>
            {band} · {percent}%
          </span>
        )}
        {message.language && <span className="lang-tag">{message.language}</span>}
      </div>

      <div className="bubble">{message.content}</div>

      {message.disclaimer && <div className="disclaimer">{message.disclaimer}</div>}

      {message.sources?.length > 0 && (
        <div className="sources">
          <h3>Nguồn tham khảo / Sources</h3>
          {message.sources.map((source, index) => (
            <div className="source" key={`${source.source_url}-${index}`}>
              <a href={source.source_url} target="_blank" rel="noreferrer">
                {source.doc_title || source.source_url}
              </a>
              <div className="source-meta">
                {source.section && <>{source.section} · </>}
                trang/page {source.page_start}
                {source.page_end > source.page_start && `–${source.page_end}`}
                {source.doc_updated_at && <> · cập nhật/updated {source.doc_updated_at}</>}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
