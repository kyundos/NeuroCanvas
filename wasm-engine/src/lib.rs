use serde::{Deserialize, Serialize};
use std::fmt;
use wasm_bindgen::prelude::*;

// Structure to map the JSON file of weights
#[derive(Deserialize)]
struct ModelWeights {
    w1: Vec<Vec<f32>>, // [784][128]
    b1: Vec<f32>,      // [128]
    w2: Vec<Vec<f32>>, // [128][64]
    b2: Vec<f32>,      // [64]
    w3: Vec<Vec<f32>>, // [64][10]
    b3: Vec<f32>,      // [10]
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct InferenceResult {
    pub predicted_label: u8,
    pub probabilities: Vec<f32>,
    pub layer_activations: Vec<Vec<f32>>,
}

#[derive(Debug)]
pub enum EngineError {
    JsonParse(String),
    InvalidModelShape(String),
    InvalidInput(String),
    Numerical(String),
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EngineError::JsonParse(msg) => write!(f, "JSON parse error: {msg}"),
            EngineError::InvalidModelShape(msg) => write!(f, "Invalid model shape: {msg}"),
            EngineError::InvalidInput(msg) => write!(f, "Invalid input: {msg}"),
            EngineError::Numerical(msg) => write!(f, "Numerical error: {msg}"),
        }
    }
}

#[wasm_bindgen]
pub struct NeuralEngine {
    model: ModelWeights,
}

#[wasm_bindgen]
impl NeuralEngine {
    // The constructor now accepts the JSON as a string from the frontend
    #[wasm_bindgen(constructor)]
    pub fn new(json_data: &str) -> Result<NeuralEngine, JsValue> {
        let model: ModelWeights = serde_json::from_str(json_data)
            .map_err(|e| JsValue::from_str(&format!("Errore parsing JSON: {}", e)))?;
        validate_model(&model).map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(NeuralEngine { model })
    }

    pub fn predict(&self, input_pixels: &[f32]) -> Result<JsValue, JsValue> {
        let result = self
            .predict_inference(input_pixels)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(serde_wasm_bindgen::to_value(&result)?)
    }
}

impl NeuralEngine {
    pub fn from_json(json_data: &str) -> Result<NeuralEngine, EngineError> {
        let model: ModelWeights =
            serde_json::from_str(json_data).map_err(|e| EngineError::JsonParse(e.to_string()))?;
        validate_model(&model)?;
        Ok(NeuralEngine { model })
    }

    pub fn predict_inference(&self, input_pixels: &[f32]) -> Result<InferenceResult, EngineError> {
        if input_pixels.len() != self.model.w1.len() {
            return Err(EngineError::InvalidInput(format!(
                "expected {} input features, got {}",
                self.model.w1.len(),
                input_pixels.len()
            )));
        }

        // Forward pass: hidden layer 1.
        let mut a1 = dense_forward(input_pixels, &self.model.w1, &self.model.b1)?;
        relu(&mut a1);

        // Forward pass: hidden layer 2.
        let mut a2 = dense_forward(&a1, &self.model.w2, &self.model.b2)?;
        relu(&mut a2);

        // Forward pass: output layer.
        let logits = dense_forward(&a2, &self.model.w3, &self.model.b3)?;
        let probabilities = softmax(&logits)?;

        let (predicted_label, _) = probabilities
            .iter()
            .copied()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or_else(|| EngineError::Numerical("empty probabilities".to_string()))?;

        Ok(InferenceResult {
            predicted_label: predicted_label as u8,
            probabilities,
            layer_activations: vec![a1, a2], // Exposed for frontend visualisation.
        })
    }
}

// --- CORE MATHEMATICAL FUNCTIONS ---

// Matrix x Vector multiplication + Bias
fn dense_forward(
    inputs: &[f32],
    weights: &[Vec<f32>],
    biases: &[f32],
) -> Result<Vec<f32>, EngineError> {
    let in_features = inputs.len();
    let out_features = biases.len();
    if weights.len() != in_features {
        return Err(EngineError::InvalidModelShape(format!(
            "weights rows {} do not match inputs {}",
            weights.len(),
            in_features
        )));
    }

    if out_features == 0 {
        return Err(EngineError::InvalidModelShape(
            "dense layer has zero output neurons".to_string(),
        ));
    }

    for (row_idx, row) in weights.iter().enumerate() {
        if row.len() != out_features {
            return Err(EngineError::InvalidModelShape(format!(
                "weights row {row_idx} has size {}, expected {out_features}",
                row.len()
            )));
        }
    }

    let mut out = vec![0.0; out_features];

    for i in 0..in_features {
        for j in 0..out_features {
            out[j] += inputs[i] * weights[i][j];
        }
    }

    for j in 0..out_features {
        out[j] += biases[j];
    }

    Ok(out)
}

// ReLU activation function (Rectified Linear Unit): turns off neurons with negative values
fn relu(layer: &mut [f32]) {
    for val in layer.iter_mut() {
        *val = val.max(0.0);
    }
}

// Converts raw outputs to percentages (0.0 - 1.0)
fn softmax(logits: &[f32]) -> Result<Vec<f32>, EngineError> {
    if logits.is_empty() {
        return Err(EngineError::Numerical(
            "softmax logits cannot be empty".to_string(),
        ));
    }
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum <= f32::EPSILON {
        return Err(EngineError::Numerical(
            "softmax denominator is too small".to_string(),
        ));
    }
    Ok(exps.into_iter().map(|x| x / sum).collect())
}

fn validate_dense_layer(
    input_size: usize,
    weights: &[Vec<f32>],
    biases: &[f32],
    layer_name: &str,
) -> Result<(), EngineError> {
    if weights.len() != input_size {
        return Err(EngineError::InvalidModelShape(format!(
            "{layer_name} expects {input_size} input rows, got {}",
            weights.len()
        )));
    }
    if biases.is_empty() {
        return Err(EngineError::InvalidModelShape(format!(
            "{layer_name} has empty biases"
        )));
    }
    for (idx, row) in weights.iter().enumerate() {
        if row.len() != biases.len() {
            return Err(EngineError::InvalidModelShape(format!(
                "{layer_name} row {idx} has {}, expected {}",
                row.len(),
                biases.len()
            )));
        }
    }
    Ok(())
}

fn validate_model(model: &ModelWeights) -> Result<(), EngineError> {
    validate_dense_layer(model.w1.len(), &model.w1, &model.b1, "layer w1->b1")?;
    validate_dense_layer(model.b1.len(), &model.w2, &model.b2, "layer w2->b2")?;
    validate_dense_layer(model.b2.len(), &model.w3, &model.b3, "layer w3->b3")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-5;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPSILON
    }

    #[test]
    fn given_dense_layer_when_forward_then_applies_matmul_and_bias() {
        let inputs = vec![1.0, 2.0];
        let weights = vec![
            vec![0.5, -1.0, 2.0],
            vec![1.5, 3.0, -0.5],
        ];
        let biases = vec![0.1, -0.2, 0.3];

        let out = match dense_forward(&inputs, &weights, &biases) {
            Ok(value) => value,
            Err(err) => panic!("dense_forward failed unexpectedly: {err}"),
        };

        assert_eq!(out.len(), 3);
        assert!(approx_eq(out[0], 3.6));
        assert!(approx_eq(out[1], 4.8));
        assert!(approx_eq(out[2], 1.3));
    }

    #[test]
    fn given_relu_when_layer_has_negative_values_then_clamps_to_zero() {
        let mut layer = vec![-3.0, 0.0, 2.5, -0.001];
        relu(&mut layer);
        assert_eq!(layer, vec![0.0, 0.0, 2.5, 0.0]);
    }

    #[test]
    fn given_softmax_when_logits_have_large_values_then_stays_numerically_stable() {
        let logits = vec![1000.0, 999.0, 998.0];
        let probs = match softmax(&logits) {
            Ok(value) => value,
            Err(err) => panic!("softmax failed unexpectedly: {err}"),
        };
        let sum: f32 = probs.iter().sum();

        assert_eq!(probs.len(), 3);
        assert!(approx_eq(sum, 1.0));
        assert!(probs[0] > probs[1] && probs[1] > probs[2]);
        assert!(probs.iter().all(|p| *p >= 0.0 && *p <= 1.0));
    }

    #[test]
    fn given_two_layer_pipeline_when_forward_then_output_probabilities_are_valid() {
        // Small deterministic network used to validate the whole math pipeline.
        let input = vec![2.0, -1.0];

        let w1 = vec![
            vec![1.0, -2.0],
            vec![0.5, 1.0],
        ];
        let b1 = vec![0.0, 0.5];

        let w2 = vec![
            vec![1.0, -1.0],
            vec![0.5, 2.0],
        ];
        let b2 = vec![0.1, -0.2];

        let mut hidden = match dense_forward(&input, &w1, &b1) {
            Ok(value) => value,
            Err(err) => panic!("layer 1 forward failed unexpectedly: {err}"),
        };
        relu(&mut hidden);
        let logits = match dense_forward(&hidden, &w2, &b2) {
            Ok(value) => value,
            Err(err) => panic!("layer 2 forward failed unexpectedly: {err}"),
        };
        let probs = match softmax(&logits) {
            Ok(value) => value,
            Err(err) => panic!("softmax failed unexpectedly: {err}"),
        };

        assert_eq!(hidden, vec![1.5, 0.0]);
        assert_eq!(logits.len(), 2);
        assert!(probs[0] > probs[1]);
        assert!(approx_eq(probs.iter().sum(), 1.0));
    }

    #[test]
    fn given_invalid_dense_dimensions_when_forward_then_returns_error() {
        let inputs = vec![1.0, 2.0];
        let weights = vec![vec![1.0], vec![2.0, 3.0]];
        let biases = vec![0.5];

        match dense_forward(&inputs, &weights, &biases) {
            Ok(_) => panic!("dense_forward should have failed"),
            Err(err) => assert!(matches!(err, EngineError::InvalidModelShape(_))),
        }
    }

    #[test]
    fn given_empty_logits_when_softmax_then_returns_error() {
        match softmax(&[]) {
            Ok(_) => panic!("softmax should have failed"),
            Err(err) => assert!(matches!(err, EngineError::Numerical(_))),
        }
    }
}
