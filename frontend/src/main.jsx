import React, { useEffect, useState } from 'react'
import { createRoot } from 'react-dom/client'

import App from './App.jsx'
import Admin from './Admin.jsx'
import './styles.css'

// Hash routing: two screens do not justify a router dependency.
function Root() {
  const [hash, setHash] = useState(window.location.hash)

  useEffect(() => {
    const onHashChange = () => setHash(window.location.hash)
    window.addEventListener('hashchange', onHashChange)
    return () => window.removeEventListener('hashchange', onHashChange)
  }, [])

  return hash === '#admin' ? <Admin /> : <App />
}

createRoot(document.getElementById('root')).render(<Root />)
