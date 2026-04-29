import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        midnight: "#06111f",
        ocean: "#0d1f36",
        cyanGlow: "#4fd1ff",
        medical: {
          50: "#eefbff",
          100: "#d6f3ff",
          200: "#afe8ff",
          300: "#74d8ff",
          400: "#39c2ff",
          500: "#0ea5e9",
          600: "#0284c7",
          700: "#05638f",
          800: "#0a3f5e",
          900: "#08263b"
        }
      },
      fontFamily: {
        display: ["Space Grotesk", "ui-sans-serif", "system-ui"],
        body: ["Manrope", "ui-sans-serif", "system-ui"]
      },
      boxShadow: {
        glow: "0 20px 80px rgba(14, 165, 233, 0.20)",
        panel: "0 24px 70px rgba(2, 8, 23, 0.45)"
      },
      backgroundImage: {
        grid: "linear-gradient(rgba(79,209,255,0.08) 1px, transparent 1px), linear-gradient(90deg, rgba(79,209,255,0.08) 1px, transparent 1px)"
      }
    }
  },
  plugins: []
};

export default config;
