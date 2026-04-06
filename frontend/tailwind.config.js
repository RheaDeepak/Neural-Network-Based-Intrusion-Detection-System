/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,jsx}"
  ],
  theme: {
    extend: {
      colors: {
        background: "#3D405B",
        foreground: "#F4F1DE",
        card: "#2F3148",
        muted: "#F2CC8F",
        primary: "#E07A5F",
        primaryForeground: "#F4F1DE",
        secondary: "#81B29A",
        accent: "#F2CC8F"
      },
      boxShadow: {
        glow: "0 0 30px rgba(224, 122, 95, 0.35)",
        glowCyan: "0 0 30px rgba(129, 178, 154, 0.3)"
      }
    }
  },
  plugins: []
}
