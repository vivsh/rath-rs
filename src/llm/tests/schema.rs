use super::*;
use serde_json::json;

/// Root and nested schema annotations are removed.
#[test]
fn strips_schema_and_title() {
    let input = json!({
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "MyTool",
        "type": "object",
        "properties": {
            "name": { "title": "Name", "type": "string" }
        }
    });
    let out = sanitize_strict(input).unwrap();
    assert!(out.get("$schema").is_none());
    assert!(out.get("title").is_none());
    assert!(out["properties"]["name"].get("title").is_none());
}

/// Root-local `$ref` values are inlined.
#[test]
fn inlines_defs_ref() {
    let input = json!({
        "type": "object",
        "properties": {
            "tasks": {
                "type": "array",
                "items": { "$ref": "#/definitions/TaskDraft" }
            }
        },
        "definitions": {
            "TaskDraft": {
                "type": "object",
                "properties": {
                    "title": { "type": "string" }
                }
            }
        }
    });
    let out = sanitize_strict(input).unwrap();
    assert!(out.get("definitions").is_none());
    let items = &out["properties"]["tasks"]["items"];
    assert_eq!(items["type"], "object");
    assert!(items.get("$ref").is_none());
}

/// Nullable type arrays are normalized.
#[test]
fn normalises_nullable_type() {
    let input = json!({
        "type": "object",
        "properties": {
            "note": { "type": ["string", "null"] }
        }
    });
    let out = sanitize_strict(input).unwrap();
    assert_eq!(out["properties"]["note"]["type"], "string");
}

/// Property keys named `title` must survive sanitization.
#[test]
fn preserves_property_named_title() {
    let input = json!({
        "type": "object",
        "title": "TaskDraft",
        "required": ["id", "title"],
        "properties": {
            "id":    { "type": "string" },
            "title": { "title": "Title", "type": "string" }
        }
    });
    let out = sanitize_strict(input).unwrap();
    assert!(out.get("title").is_none());
    assert!(
        out["properties"].get("title").is_some(),
        "title property was incorrectly removed"
    );
    assert!(out["properties"]["title"].get("title").is_none());
}

/// `additionalProperties` is removed at every level.
#[test]
fn removes_additional_properties() {
    let input = json!({
        "type": "object",
        "additionalProperties": false,
        "properties": {}
    });
    let out = sanitize_strict(input).unwrap();
    assert!(out.get("additionalProperties").is_none());
}

/// Unsafe reference transformations fail instead of recursing indefinitely or dropping constraints.
#[test]
fn rejects_recursive_unresolved_and_sibling_references() {
    for value in [
        json!({"$defs":{"Node":{"$ref":"#/$defs/Node"}},"$ref":"#/$defs/Node"}),
        json!({"$ref":"#/$defs/missing"}),
        json!({"$ref":"https://example.com/schema"}),
        json!({"$defs":{"X":{"type":"string"}},"$ref":"#/$defs/X","maxLength":2}),
    ] {
        assert_eq!(
            sanitize_strict(value).unwrap_err().kind(),
            crate::ErrorKind::Validation
        );
    }
}

/// Local root references and literal `$ref` property names retain their intended schema meaning.
#[test]
fn root_references_and_named_properties_are_supported() {
    let value = json!({"$defs":{"X":{"type":"object","properties":{"$ref":{"type":"string"}}}},"$ref":"#/$defs/X"});
    let normalized = sanitize_strict(value).unwrap();
    assert_eq!(normalized["properties"]["$ref"]["type"], "string");
}
