import { motion } from "framer-motion";
import {
  Activity,
  Droplets,
  HeartPulse,
  LoaderCircle,
  Stethoscope,
  Thermometer,
  UserRound,
  type LucideIcon,
} from "lucide-react";
import type { ChangeEvent, FormEvent } from "react";
import type { PatientFormData } from "../../data/mockData";
import { getFieldRange, numericFieldSpecs, type NumericFormField } from "../../lib/riesgoMaterno";
import { GlassPanel } from "../ui/GlassPanel";
import { SectionHeader } from "../ui/SectionHeader";

interface PatientDataSectionProps {
  formData: PatientFormData;
  isAnalyzing: boolean;
  onFieldChange: (
    field: keyof PatientFormData,
    event: ChangeEvent<HTMLInputElement>,
  ) => void;
  onAnalyze: () => void;
}

const iconByField: Record<NumericFormField, LucideIcon> = {
  age: UserRound,
  systolicBP: Stethoscope,
  diastolicBP: Activity,
  bloodGlucose: Droplets,
  bodyTemperature: Thermometer,
  heartRate: HeartPulse,
};

export function PatientDataSection({
  formData,
  isAnalyzing,
  onFieldChange,
  onAnalyze,
}: PatientDataSectionProps) {
  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    onAnalyze();
  }

  return (
    <section className="section-anchor pt-10" id="patient-entry">
      <SectionHeader
        eyebrow="Ingreso clinico"
        title="Variables del paciente"
        description="Ingrese los 6 indicadores clinicos para ejecutar la inferencia difusa."
      />
      <GlassPanel className="overflow-hidden">
        <form className="p-6 sm:p-8" onSubmit={handleSubmit}>
          <div className="grid gap-5 md:grid-cols-2 xl:grid-cols-3">
            {numericFieldSpecs.map((field, index) => {
              const Icon = iconByField[field.formKey];
              return (
                <motion.label
                  key={field.formKey}
                  animate={{ opacity: 1, y: 0 }}
                  className="rounded-3xl border border-sky-100 bg-white/78 p-4 transition hover:border-cyan-300/35 hover:bg-cyan-50"
                  initial={{ opacity: 0, y: 16 }}
                  transition={{ duration: 0.4, delay: index * 0.06, ease: [0.22, 1, 0.36, 1] }}
                >
                  <div className="mb-3 flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div className="rounded-2xl border border-cyan-300/30 bg-cyan-50 p-2.5 text-cyan-700">
                        <Icon className="h-4 w-4" />
                      </div>
                      <div>
                        <span className="block text-sm font-semibold text-slate-900">
                          {field.label}
                        </span>
                        <span className="block text-[11px] text-slate-400">
                          {getFieldRange(field.formKey)}
                        </span>
                      </div>
                    </div>
                    <span className="rounded-full border border-sky-100 bg-sky-50 px-2.5 py-1 text-xs font-medium text-slate-600">
                      {field.unit}
                    </span>
                  </div>
                  <input
                    className="w-full rounded-2xl border border-sky-100 bg-white px-4 py-3 text-base text-slate-900 outline-none transition placeholder:text-slate-400 focus:border-cyan-400/60 focus:ring-2 focus:ring-cyan-200"
                    inputMode="decimal"
                    max={field.max}
                    min={field.min}
                    placeholder={field.placeholder}
                    step={field.step}
                    type="number"
                    value={formData[field.formKey]}
                    onChange={(event) => onFieldChange(field.formKey, event)}
                  />
                </motion.label>
              );
            })}
          </div>

          <div className="mt-6">
            <button
              className="inline-flex items-center gap-2 rounded-full bg-gradient-to-r from-cyan-600 to-sky-600 px-6 py-3 text-sm font-semibold text-white transition hover:scale-[1.01] hover:brightness-105 disabled:cursor-wait disabled:opacity-90"
              disabled={isAnalyzing}
              type="submit"
            >
              {isAnalyzing ? (
                <>
                  <LoaderCircle className="h-4 w-4 animate-spin" />
                  Ejecutando inferencia...
                </>
              ) : (
                "Ejecutar inferencia"
              )}
            </button>
          </div>
        </form>
      </GlassPanel>
    </section>
  );
}
