import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// In dev the SPA runs on Vite's port and proxies /api to the FastAPI process.
// In production `npm run build` writes dist/, which FastAPI serves itself.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
})
