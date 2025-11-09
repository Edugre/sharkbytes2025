/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['"Inter"', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'monospace']
      },
      colors: {
        space: {
          900: '#05060A',
          800: '#0A0F1A',
          700: '#0D1117',
          600: '#161B22'
        },
        neon: {
          cyan: '#35DDF9',
          blue: '#3D8BFF'
        }
      },
      boxShadow: {
        glow: '0 0 12px -2px rgba(53,221,249,0.6), 0 0 32px -6px rgba(94,154,255,0.4)'
      },
      backgroundImage: {
        'space-gradient': 'radial-gradient(circle at 20% 30%, rgba(20,35,60,0.35), transparent 60%), radial-gradient(circle at 80% 70%, rgba(53,221,249,0.12), transparent 65%), linear-gradient(135deg, #05060A 0%, #0A0F1A 55%, #0D1117 100%)',
        'panel-glass': 'linear-gradient(145deg, rgba(255,255,255,0.10), rgba(255,255,255,0.03))'
      },
      animation: {
        'pulse-soft': 'pulseSoft 5s ease-in-out infinite',
        'float': 'float 6s ease-in-out infinite'
      },
      keyframes: {
        pulseSoft: {
          '0%,100%': { opacity: 0.85 },
          '50%': { opacity: 1 }
        },
        float: {
          '0%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-6px)' },
          '100%': { transform: 'translateY(0px)' }
        }
      }
    },
  },
  plugins: [],
}
