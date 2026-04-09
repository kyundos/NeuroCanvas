import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    watch: {
      // WSL2 can miss native FS events; polling keeps HMR reliable.
      usePolling: true,
      interval: 100,
    },
    hmr: {
      overlay: true,
    },
  },
})
