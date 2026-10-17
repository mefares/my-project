import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// In development the page is served on port 5173 and every /api request is
// forwarded to the FastAPI backend on port 8000, so no CORS setup is needed.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': 'http://127.0.0.1:8000',
    },
  },
})
