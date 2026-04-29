export interface PatientFormData {
  age: string;
  systolicBP: string;
  diastolicBP: string;
  bloodGlucose: string;
  bodyTemperature: string;
  heartRate: string;
}

export const initialPatientForm: PatientFormData = {
  age: "25",
  systolicBP: "130",
  diastolicBP: "80",
  bloodGlucose: "15",
  bodyTemperature: "98",
  heartRate: "86",
};
