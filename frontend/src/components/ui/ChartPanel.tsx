import type { ReactNode } from "react";
import { GlassPanel } from "./GlassPanel";

interface ChartPanelProps {
  title: string;
  subtitle: string;
  children: ReactNode;
  className?: string;
}

export function ChartPanel({ title, subtitle, children, className }: ChartPanelProps) {
  return (
    <GlassPanel className={`h-full p-5 sm:p-6 ${className ?? ""}`}>
      <div className="mb-4">
        <h3 className="text-lg font-semibold text-slate-900">{title}</h3>
        {subtitle.trim().length > 0 ? (
          <p className="mt-1 text-sm text-slate-600">{subtitle}</p>
        ) : null}
      </div>
      {children}
    </GlassPanel>
  );
}
