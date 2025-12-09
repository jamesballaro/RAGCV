import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    // Optional: Proxy API requests if backend runs on different port locally to avoid CORS
    // proxy: {
    //   '/query': 'http://localhost:8000'
    // }
  }
})