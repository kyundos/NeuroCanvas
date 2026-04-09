use wasm_engine::NeuralEngine;

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

#[test]
fn given_valid_model_and_input_when_predicting_then_returns_consistent_result() {
    let engine = match NeuralEngine::from_json(tiny_model_json()) {
        Ok(value) => value,
        Err(err) => panic!("engine creation failed unexpectedly: {err}"),
    };

    let result = match engine.predict_inference(&[0.8, 0.2]) {
        Ok(value) => value,
        Err(err) => panic!("prediction failed unexpectedly: {err}"),
    };

    assert_eq!(result.probabilities.len(), 2);
    assert_eq!(result.layer_activations.len(), 2);
    assert_eq!(result.layer_activations[0].len(), 2);
    assert_eq!(result.layer_activations[1].len(), 2);
    assert!(result.predicted_label <= 1);

    let sum: f32 = result.probabilities.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

#[test]
fn given_invalid_input_size_when_predicting_then_returns_error() {
    let engine = match NeuralEngine::from_json(tiny_model_json()) {
        Ok(value) => value,
        Err(err) => panic!("engine creation failed unexpectedly: {err}"),
    };

    let result = engine.predict_inference(&[0.8]);
    assert!(result.is_err());
}

#[test]
fn given_invalid_model_shape_when_loading_then_returns_error() {
    let invalid_model = r#"{
      "w1": [[0.5, -0.2], [1.0, 0.3]],
      "b1": [0.1, 0.0],
      "w2": [[1.0, -0.5]],
      "b2": [0.0, 0.2],
      "w3": [[0.6, 0.1], [0.4, 0.9]],
      "b3": [0.0, 0.0]
    }"#;

    let result = NeuralEngine::from_json(invalid_model);
    assert!(result.is_err());
}
