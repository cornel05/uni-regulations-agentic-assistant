import React, { useCallback, useEffect, useState } from 'react'

import {
  fetchAdminDocuments,
  fetchAdminStatus,
  fetchDomains,
  triggerCrawl,
  uploadDocument,
} from './api.js'

const TOKEN_KEY = 'unireg.adminToken'

export default function Admin() {
  // sessionStorage, not localStorage: the token dies with the tab.
  const [token, setToken] = useState(() => sessionStorage.getItem(TOKEN_KEY) ?? '')
  const [status, setStatus] = useState(null)
  const [documents, setDocuments] = useState([])
  const [domains, setDomains] = useState([])
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    fetchDomains()
      .then(setDomains)
      .catch(() => setDomains([]))
  }, [])

  const load = useCallback(async () => {
    if (!token) return
    setError('')
    setBusy(true)
    try {
      const [nextStatus, nextDocuments] = await Promise.all([
        fetchAdminStatus(token),
        fetchAdminDocuments(token),
      ])
      setStatus(nextStatus)
      setDocuments(nextDocuments)
      sessionStorage.setItem(TOKEN_KEY, token)
    } catch (exc) {
      setError(exc.message)
      setStatus(null)
      setDocuments([])
    } finally {
      setBusy(false)
    }
  }, [token])

  useEffect(() => {
    load()
  }, [load])

  async function onCrawl() {
    setBusy(true)
    setError('')
    setNotice('')
    try {
      const result = await triggerCrawl(token)
      setNotice(
        `Crawl xong: ${result.changed_count} tài liệu thay đổi, ` +
          `${result.skipped_count} không đổi, ${result.indexed_chunks} đoạn được lập chỉ mục.`,
      )
      await load()
    } catch (exc) {
      setError(exc.message)
    } finally {
      setBusy(false)
    }
  }

  async function onUpload(event) {
    event.preventDefault()
    const form = new FormData(event.currentTarget)
    const file = form.get('file')
    const title = String(form.get('title') ?? '').trim()
    const domain = String(form.get('domain') ?? '')

    if (!file || !file.size || !title) {
      setError('Cần chọn tệp PDF và nhập tiêu đề. / A PDF file and a title are required.')
      return
    }

    setBusy(true)
    setError('')
    setNotice('')
    try {
      const result = await uploadDocument(token, { file, title, domain: domain || null })
      setNotice(
        `Đã lập chỉ mục "${result.title}" (${result.domain}): ${result.indexed_chunks} đoạn.`,
      )
      event.target.reset()
      await load()
    } catch (exc) {
      setError(exc.message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="admin">
      <header>
        <h1>Quản trị chỉ mục / Index admin</h1>
        <a href="#">← Trở lại chat / Back to chat</a>
      </header>

      <div className="card">
        <h2>Xác thực / Authentication</h2>
        <div className="field">
          <label htmlFor="admin-token">
            ADMIN_TOKEN — gửi qua header X-Admin-Token, lưu trong tab này
          </label>
          <input
            id="admin-token"
            type="password"
            value={token}
            autoComplete="off"
            placeholder="Nhập admin token…"
            onChange={(event) => setToken(event.target.value)}
          />
        </div>
        <button type="button" className="secondary" onClick={load} disabled={!token || busy}>
          {busy ? 'Đang tải…' : 'Tải lại / Reload'}
        </button>
      </div>

      {error && <div className="error">{error}</div>}
      {notice && (
        <div className="card" style={{ borderLeft: '3px solid var(--high)' }}>
          {notice}
        </div>
      )}

      {status && (
        <>
          <div className="card">
            <h2>Trạng thái / Status</h2>
            <div className="stat-grid">
              <Stat value={status.total_documents} label="Tài liệu / Documents" />
              <Stat value={status.indexed_documents} label="Đã lập chỉ mục / Indexed" />
              <Stat value={status.failed_documents} label="Lỗi / Failed" />
              <Stat value={status.total_chunks} label="Đoạn / Chunks" />
            </div>
            <p className="muted" style={{ marginBottom: 0 }}>
              Crawl gần nhất: {status.last_crawled_at || '—'} · Lập chỉ mục gần nhất:{' '}
              {status.last_indexed_at || '—'}
            </p>
          </div>

          <div className="card">
            <h2>Theo lĩnh vực / Per domain</h2>
            {Object.keys(status.documents_per_domain).length === 0 ? (
              <p className="muted">Chưa có tài liệu nào.</p>
            ) : (
              <div className="stat-grid">
                {Object.entries(status.documents_per_domain).map(([name, count]) => (
                  <Stat key={name} value={count} label={name} />
                ))}
              </div>
            )}
          </div>

          <div className="card">
            <h2>Tải lên tài liệu / Upload a document</h2>
            <form onSubmit={onUpload}>
              <div className="field">
                <label htmlFor="upload-file">Tệp PDF</label>
                <input id="upload-file" name="file" type="file" accept="application/pdf" />
              </div>
              <div className="field">
                <label htmlFor="upload-title">Tiêu đề / Title</label>
                <input id="upload-title" name="title" type="text" maxLength={500} />
              </div>
              <div className="field">
                <label htmlFor="upload-domain">
                  Lĩnh vực / Domain — để trống để hệ thống tự phân loại
                </label>
                <select id="upload-domain" name="domain" defaultValue="">
                  <option value="">Tự động phân loại / Classify automatically</option>
                  {domains.map((item) => (
                    <option key={item.value} value={item.value}>
                      {item.label_vi} · {item.label_en}
                    </option>
                  ))}
                </select>
              </div>
              <div className="row">
                <button type="submit" className="primary" disabled={busy}>
                  Tải lên & lập chỉ mục / Upload
                </button>
                <button type="button" className="secondary" onClick={onCrawl} disabled={busy}>
                  Chạy crawl ngay / Crawl now
                </button>
              </div>
            </form>
          </div>

          <div className="card">
            <h2>Lần chạy gần đây / Recent runs</h2>
            {status.recent_runs.length === 0 ? (
              <p className="muted">Chưa có lần chạy nào.</p>
            ) : (
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Bắt đầu</th>
                      <th>Nguồn</th>
                      <th>Trạng thái</th>
                      <th>Thay đổi</th>
                      <th>Đoạn</th>
                      <th>Lỗi</th>
                    </tr>
                  </thead>
                  <tbody>
                    {status.recent_runs.map((run) => (
                      <tr key={run.id}>
                        <td>{run.started_at}</td>
                        <td>{run.trigger}</td>
                        <td>
                          <span
                            className={`pill ${run.status === 'completed' ? 'indexed' : run.status === 'failed' ? 'failed' : ''}`}
                          >
                            {run.status}
                          </span>
                        </td>
                        <td>{run.changed_count}</td>
                        <td>{run.indexed_chunks}</td>
                        <td className="wrap muted">{run.error || '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          <div className="card">
            <h2>Tài liệu / Documents</h2>
            {documents.length === 0 ? (
              <p className="muted">Chưa có tài liệu nào được lập chỉ mục.</p>
            ) : (
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Tiêu đề</th>
                      <th>Lĩnh vực</th>
                      <th>Đoạn</th>
                      <th>Trạng thái</th>
                      <th>Cập nhật</th>
                    </tr>
                  </thead>
                  <tbody>
                    {documents.map((document) => (
                      <tr key={document.source_url}>
                        <td className="wrap">
                          <a href={document.source_url} target="_blank" rel="noreferrer">
                            {document.title}
                          </a>
                        </td>
                        <td>{document.domain}</td>
                        <td>{document.chunk_count}</td>
                        <td>
                          <span className={`pill ${document.status}`}>{document.status}</span>
                        </td>
                        <td>{document.doc_updated_at || '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </>
      )}

      {!status && !error && !token && (
        <p className="muted">Nhập ADMIN_TOKEN để xem trạng thái chỉ mục.</p>
      )}
    </div>
  )
}

function Stat({ value, label }) {
  return (
    <div className="stat">
      <b>{value}</b>
      <span>{label}</span>
    </div>
  )
}
