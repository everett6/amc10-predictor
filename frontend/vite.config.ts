import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// `base: './'` keeps asset URLs relative so the built UI works when the
// backend serves it from any path. In dev, /api is proxied to the backend.
export default defineConfig({
  plugins: [react()],
  base: './',
  server: { proxy: { '/api': 'http://127.0.0.1:8000' } },
})
