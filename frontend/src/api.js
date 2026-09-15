const JSON_HEADERS = { 'Content-Type': 'application/json' }

async function request(path, options = {}) {
  let response
  try {
    response = await fetch(path, options)
  } catch {
    throw new Error('Không kết nối được tới server. / Cannot reach the server.')
  }

  if (!response.ok) {
    let detail = `Yêu cầu thất bại (${response.status}). / Request failed (${response.status}).`
    try {
      const body = await response.json()
      if (typeof body.detail === 'string') detail = body.detail
    } catch {
      // Non-JSON error body; the status-based message stands.
    }
    throw new Error(detail)
  }

  return response.status === 204 ? null : response.json()
}

const adminHeaders = (token) => ({ 'X-Admin-Token': token })

export const fetchDomains = () => request('/api/domains')

export const sendMessage = ({ message, sessionId, domain }) =>
  request('/api/chat', {
    method: 'POST',
    headers: JSON_HEADERS,
    body: JSON.stringify({ message, session_id: sessionId ?? null, domain: domain ?? null }),
  })

export const clearSession = (sessionId) =>
  request(`/api/sessions/${encodeURIComponent(sessionId)}`, { method: 'DELETE' })

export const fetchAdminStatus = (token) =>
  request('/api/admin/status', { headers: adminHeaders(token) })

export const fetchAdminDocuments = (token) =>
  request('/api/admin/documents', { headers: adminHeaders(token) })

export const triggerCrawl = (token) =>
  request('/api/admin/crawl', { method: 'POST', headers: adminHeaders(token) })

export const uploadDocument = (token, { file, title, domain }) => {
  const form = new FormData()
  form.append('file', file)
  form.append('title', title)
  if (domain) form.append('domain', domain)
  // No Content-Type header: the browser sets the multipart boundary.
  return request('/api/admin/documents', {
    method: 'POST',
    headers: adminHeaders(token),
    body: form,
  })
}
