use serde_json::Value;

/// Tool metadata exposed to an LLM.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ToolDefinition {
    /// Registered tool name as seen by the model.
    pub name: String,
    /// Human-readable description of what the tool does.
    pub description: String,
    /// JSON Schema describing the tool input.
    pub parameters: Value,
}

impl ToolDefinition {
    /// Constructs required values; optional controls start unset.
    pub fn new(name: String, description: String, parameters: Value) -> Self {
        Self {
            name,
            description,
            parameters,
        }
    }
}
