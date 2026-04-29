# Frontend Prototype

High-fidelity frontend prototype for the maternal risk medical decision support system.

## Stack

- React
- Vite
- TypeScript
- Tailwind CSS
- Apache ECharts
- Framer Motion
- Lucide React

## Current mode

This frontend is intentionally running in demo mode:

- the patient form is fully interactive
- the "Analyze Patient" action currently triggers mocked outputs
- charts, rules, optimization history, and analytics are simulated
- the structure is ready to be connected to the Python backend later

## Run locally

```powershell
npm.cmd install
npm.cmd run dev
```

## Verification notes

- TypeScript compilation was verified with:

```powershell
.\node_modules\.bin\tsc.cmd -b
```

- A full `vite build` may require running outside restrictive sandboxes because Vite uses `esbuild` subprocesses.
