export interface InferenceResult {
  predicted_label: number;
  probabilities: number[];
  layer_activations: number[][];
}

export interface ModelApiResponse {
  version: string;
  model: unknown;
}

export interface LabelSampleResponse {
  saved: boolean;
  sample_id: number;
}

export interface DeleteSamplesResponse {
  deleted_count: number;
}

export interface LabelCount {
  label: number;
  count: number;
}

export interface DatasetStatsResponse {
  total_samples: number;
  counts_by_label: LabelCount[];
}
