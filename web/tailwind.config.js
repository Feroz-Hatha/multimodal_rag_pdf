/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        inter: ['Inter', 'sans-serif'],
        garamond: ['"EB Garamond"', 'Georgia', 'serif'],
      },
      colors: {
        sage: {
          50:  '#f0f7f0',
          100: '#dcebdc',
          200: '#c3d9c3',
          300: '#99be99',
          400: '#6b9b6b',
          500: '#4a7c59',
          600: '#3d6b4f',
          700: '#2f5540',
          800: '#1c3326',
          900: '#0f1f16',
        },
      },
    },
  },
  plugins: [],
}
