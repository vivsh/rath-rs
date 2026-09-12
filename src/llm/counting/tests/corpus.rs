use super::super::estimate;

/// Reports estimation error against sourced examples, without treating them as safety bounds.
#[test]
fn reports_published_count_comparison() {
    let rows: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/provider-counts.json")).unwrap();
    let rows = rows.as_array().unwrap();
    let mut underestimated = 0;
    let mut absolute_error = 0;
    for row in rows {
        let actual = row["input_tokens"].as_u64().unwrap();
        let count = estimate(
            &row["prompt"],
            row["wire_messages"].as_u64().unwrap() as usize,
            row["tool_definitions"].as_u64().unwrap() as usize,
        )
        .unwrap()
        .input_tokens;
        let error = i128::from(count) - i128::from(actual);
        absolute_error += error.abs();
        underestimated += usize::from(count < actual);
        println!(
            "{}: reference={actual}, estimated={count}, error={error:+}, relative_error={:.1}%",
            row["id"],
            100.0 * error as f64 / actual as f64
        );
        assert!(row["source"].as_str().unwrap().starts_with("https://"));
    }
    println!(
        "Published examples only: underestimates={underestimated}/{}, mean_absolute_error={:.1} tokens. Not a live calibration or context-safety guarantee.",
        rows.len(),
        absolute_error as f64 / rows.len() as f64
    );
}
