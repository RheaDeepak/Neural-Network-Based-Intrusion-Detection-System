/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,jsx}"
  ],
  theme: {
    extend: {
      colors: {
        background: "hsl(222 47% 11%)",
        foreground: "hsl(210 40% 98%)",
        card: "hsl(222 47% 13%)",
        muted: "hsl(215 20% 65%)",
        primary: "hsl(198 93% 60%)",
        primaryForeground: "hsl(210 40% 5%)"
      },
      boxShadow: {
        glow: "0 0 30px rgba(56, 189, 248, 0.25)"
      }
    }
  },
  plugins: []
}
