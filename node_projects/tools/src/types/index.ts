export interface Scenario {
  name: string;
  description: string;
  steps: Step[];
}

export interface Step {
  name: string;
  action: (modules: any) => Promise<void>;
}
