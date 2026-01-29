import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/query': 'http://localhost:8000',
      '/compile_latex': 'http://localhost:8000',
      '/logs': 'http://localhost:8000',
      '/docs': 'http://localhost:8000'
    }
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
    chunkSizeWarningLimit: 1000,
  }
})