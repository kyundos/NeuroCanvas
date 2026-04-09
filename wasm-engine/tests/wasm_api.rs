#![cfg(target_arch = "wasm32")]

use wasm_bindgen_test::*;
use wasm_engine::{InferenceResult, NeuralEngine};

fn tiny_model_json() -> &'static str {
    r#"{
      "w1": [[0.5, -0.2], [1.0, 0.3]],
      "b1": [0.1, 0.0],
      "w2": [[1.0, -0.5], [0.2, 0.7]],
      "b2": [0.0, 0.2],
      "w3": [[0.6, 0.1], [0.4, 0.9]],
      "b3": [0.0, 0.0]
    }"#
}

#[wasm_bindgen_test]
fn given_valid_json_when_constructing_engine_then_succeeds() {
    let engine = NeuralEngine::new(tiny_model_json());
    assert!(engine.is_ok());
}

#[wasm_bindgen_test]
fn given_valid_input_when_predicting_then_returns_serializable_payload() {
    let engine = match NeuralEngine::new(tiny_model_json()) {
        Ok(value) => value,
        Err(err) => panic!("engine creation failed unexpectedly: {:?}", err),
    };

    let raw = match engine.predict(&[0.8, 0.2]) {
        Ok(value) => value,
        Err(err) => panic!("prediction failed unexpectedly: {:?}", err),
    };

    let result: InferenceResult = match serde_wasm_bindgen::from_value(raw) {
        Ok(value) => value,
        Err(err) => panic!("deserialization failed unexpectedly: {err}"),
    };

    assert_eq!(result.probabilities.len(), 2);
    assert_eq!(result.layer_activations.len(), 2);
    assert!(result.predicted_label <= 1);
}

#[wasm_bindgen_test]
fn given_invalid_input_when_predicting_then_returns_error() {
    let engine = match NeuralEngine::new(tiny_model_json()) {
        Ok(value) => value,
        Err(err) => panic!("engine creation failed unexpectedly: {:?}", err),
    };

    let result = engine.predict(&[0.8]);
    assert!(result.is_err());
}
