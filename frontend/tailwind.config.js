/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        // LangGraph Studio-inspired colors
        'studio-bg': '#0f0f0f',
        'studio-panel': '#1a1a1a',
        'studio-border': '#2a2a2a',
        'studio-text': '#e5e5e5',
        'studio-muted': '#888888',
        'studio-accent': '#3b82f6',
        'studio-success': '#22c55e',
        'studio-warning': '#f59e0b',
        'studio-error': '#ef4444',
      },
    },
  },
  plugins: [],
}
