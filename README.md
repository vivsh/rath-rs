# Rath

[![Crates.io](https://img.shields.io/crates/v/rath-rs)](https://crates.io/crates/rath-rs)
[![docs.rs](https://img.shields.io/docsrs/rath-rs)](https://docs.rs/rath-rs)
[![License](https://img.shields.io/crates/l/rath-rs)](LICENSE)

_Rath_ (रथ, _ruth-uh_) means "chariot" in Sanskrit.

Rath is a provider-agnostic Rust API layer for AI applications. It exposes
capability-focused modules for LLM calls, embeddings, image APIs, video APIs,
and audio APIs while keeping provider adapters behind stable traits.

Rath is not another AI provider SDK. It is a stable capability layer over
provider SDKs/APIs.

## Modules

- `rath::core`: shared provider types, `ModelUrl`, `RathError`, and token usage
- `rath::llm`: text-generation clients, messages, tools, structured output, and provider adapters
- `rath::embeddings`: embedding request/response types and `EmbeddingClient`
- `rath::images`: image request/response types and `ImageClient`
- `rath::video`: async video job request/response types and `VideoClient`
- `rath::audio`: text-to-speech and speech-to-text traits and types

Internally, provider adapters live under
`providers/{openai,openrouter,gemini,anthropic,ollama,fal}.rs`. That module is not
part of the public API; consumers should import capability traits and types
instead.

## Model URLs

Rath uses one model locator format across capabilities. The path is always the
provider-native model id or endpoint slug; custom HTTP endpoints are configured
with `base_url`.

```text
provider:///provider-native-model-id[?params]
```

Examples:

```text
openai:///gpt-4o
openrouter:///openai/gpt-5.2
openai:///text-embedding-3-large
openai:///gpt-image-1
openai:///tts-1
fal:///fal-ai/flux/schnell
fal:///fal-ai/wan/v2.2-a14b/text-to-video
gemini:///gemini-2.5-flash
ollama:///qwen3:8b?base_url=http://localhost:11434
openai:///gpt-4o?base_url=https://api.example.com/v1
```

Use `rath::core::ModelUrl` for parsed URLs. `rath::llm::LlmUrl` remains as a
compatibility alias.

OpenRouter model slugs keep their provider prefix in the URL path, for example
`openrouter:///anthropic/claude-sonnet-4.5`.

Fal model slugs also keep the full path, for example
`fal:///fal-ai/flux/schnell`.

Model locators are not provider HTTP URLs. `openai:///gpt-4o` means "use the
OpenAI adapter with model `gpt-4o`"; `base_url` is the only place for a custom
provider endpoint.

Capability options parse the same URL format and dispatch to the provider
implementation that supports that capability.

## Credentials

Rath reads provider API keys from environment variables. The default variables
are:

- `OPENAI_API_KEY` for OpenAI
- `OPENROUTER_API_KEY` for OpenRouter
- `ANTHROPIC_API_KEY` for Anthropic
- `GEMINI_API_KEY` for Gemini
- `FAL_KEY` for Fal
- `OLLAMA_API_KEY` for Ollama only when the Ollama server requires auth

Set the relevant variable before creating a client:

```sh
export OPENAI_API_KEY="..."
export FAL_KEY="..."
```

Provider clients read their default environment variable when the client is
created. To use a different variable, pass its name with `api_key_env` in the
model locator:

```text
openai:///gpt-4o?api_key_env=MY_OPENAI_KEY
fal:///fal-ai/flux/schnell?api_key_env=MY_FAL_KEY
```

When `api_key_env` is present, `ModelUrl::parse` reads that environment variable
immediately and stores the resolved key on the parsed model locator. When it is
not present, Rath falls back to the provider default variable listed above.
Missing required keys return an error before making a provider request. Rath does
not accept raw API keys in constructors or inline credentials in model locators;
keep secrets in environment variables.

## Embedding Usage

```rust
use rath::embeddings::{EmbedRequest, EmbeddingOptions};

# async fn run() -> Result<(), Box<dyn std::error::Error>> {
let client = EmbeddingOptions::default().create("openai:///text-embedding-3-small")?;
let response = client.embed(&EmbedRequest {
    input: "Rust workflows with provider-agnostic AI clients".to_string(),
    ..EmbedRequest::default()
}).await?;

println!("{} dimensions", response.values.len());
# Ok(())
# }
```

## Image Usage

```rust
use rath::images::{ImageOptions, ImageRequest};

# async fn run() -> Result<(), Box<dyn std::error::Error>> {
let client = ImageOptions::default().create("fal:///fal-ai/flux/schnell")?;
let response = client.generate_image(&ImageRequest {
    prompt: "A walnut desk lamp in warm studio light".to_string(),
    size: Some("landscape_4_3".to_string()),
    ..ImageRequest::default()
}).await?;

println!("{} image(s)", response.images.len());
# Ok(())
# }
```

## Video Usage

```rust
use rath::video::{VideoJobStatus, VideoOptions, VideoRequest};

# async fn run() -> Result<(), Box<dyn std::error::Error>> {
let client = VideoOptions::default().create("fal:///fal-ai/wan/v2.2-a14b/text-to-video")?;
let job = client.submit_video(&VideoRequest {
    prompt: "A slow cinematic push-in on a brass astrolabe".to_string(),
    ..VideoRequest::default()
}).await?;

match client.get_video(&job.id).await? {
    VideoJobStatus::Succeeded { response } => println!("{} video(s)", response.videos.len()),
    VideoJobStatus::Failed { message, .. } => println!("video failed: {message}"),
    VideoJobStatus::Queued { .. } | VideoJobStatus::Running { .. } => println!("still rendering"),
}
# Ok(())
# }
```

## Audio Usage

```rust
use rath::audio::tts::{TtsOptions, TtsRequest};

# async fn run() -> Result<(), Box<dyn std::error::Error>> {
let client = TtsOptions::default().create("openai:///tts-1")?;
let response = client.synthesize_speech(&TtsRequest {
    input: "Rath is a stable capability layer for AI applications.".to_string(),
    voice: Some("alloy".to_string()),
    format: Some("mp3".to_string()),
    ..TtsRequest::default()
}).await?;

println!("{} bytes of {}", response.data.len(), response.mime_type);
# Ok(())
# }
```

## LLM Usage

```rust
use rath::llm::{LlmOptions, LlmOutput, Message};

# async fn run() -> Result<(), Box<dyn std::error::Error>> {
let client = LlmOptions::default().create("openai:///gpt-4o")?;
let response = client.execute(&[Message::user("Write one sentence about Rust.")]).await?;

match response.output {
    LlmOutput::Output(value) => println!("{value}"),
    LlmOutput::ToolCalls { .. } => println!("model requested tools"),
}
# Ok(())
# }
```

## LLM Provider Config

Use `provider_config` for provider-specific request knobs that Rath does not
model directly. Common options such as temperature, thinking, tools, schemas,
system prompt, and cache should still use Rath's typed fields.

For example, Gemini safety settings can be provided with `safetySettings`:

```rust
use serde_json::json;
use rath::llm::LlmOptions;

# fn run() -> Result<(), Box<dyn std::error::Error>> {
let client = LlmOptions::default()
    .with_provider_config(json!({
        "safetySettings": [
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
            }
        ]
    }))
    .create("gemini:///gemini-2.5-flash")?;
# Ok(())
# }
```

## Exit Tool Quirk

Rath uses native structured output where providers support it cleanly. For
Ollama, and Gemini models before `gemini-3.1`, Rath may convert structured
output into a required synthetic tool call. This is internal to Rath: callers
still receive `LlmOutput::Output(...)` when the exit tool is called successfully.

Use `LlmClient::uses_exit_tool()` to check whether a client uses this strategy.

## License

Licensed under either the MIT License or Apache License 2.0, at your option.
