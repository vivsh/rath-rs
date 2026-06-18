# Rath

[![Crates.io](https://img.shields.io/crates/v/rath-rs)](https://crates.io/crates/rath-rs)
[![docs.rs](https://img.shields.io/docsrs/rath-rs)](https://docs.rs/rath-rs)
[![License](https://img.shields.io/crates/l/rath-rs)](LICENSE-MIT)

_Rath_ (रथ, _ruth-uh_) means "chariot" in Sanskrit.

Rath is a provider-agnostic Rust API layer for AI applications. It exposes
capability-focused modules for LLM calls, embeddings, image APIs, video APIs,
and audio APIs while keeping provider adapters behind stable traits.

The public API is organized by capability. Provider-specific implementations are
kept in a private `providers` module so one provider file can share auth,
transport, payload mapping, and response parsing across the capabilities it
supports.

## Modules

- `rath::core`: shared provider types, `ModelUrl`, `RathError`, and token usage
- `rath::llm`: text-generation clients, messages, tools, structured output, and provider adapters
- `rath::embeddings`: embedding request/response types and `EmbeddingClient`
- `rath::images`: image request/response types and `ImageClient`
- `rath::video`: video request/response types and `VideoClient`
- `rath::audio`: text-to-speech and speech-to-text traits and types

Internally, provider adapters live under
`providers/{openai,openrouter,gemini,anthropic,ollama,fal}.rs`. That module is not
part of the public API; consumers should import capability traits and types
instead.

## Model URLs

Rath uses one model URL format across capabilities:

```text
provider[+transport]://[host][/path/]model[?params]
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
ollama://localhost:11434/qwen3:8b
openai+https://openrouter.ai/api/v1/gpt-4o
```

Use `rath::core::ModelUrl` for parsed URLs. `rath::llm::LlmUrl` remains as a
compatibility alias.

OpenRouter model slugs keep their provider prefix in the URL path, for example
`openrouter:///anthropic/claude-sonnet-4.5`.

Fal model slugs also keep the full path, for example
`fal:///fal-ai/flux/schnell`. Rath uses Fal's REST API; set `FAL_KEY` or pass
`?api_key_env=YOUR_ENV_VAR`.

Capability options parse the same URL format and dispatch to the provider
implementation that supports that capability.

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
use rath::video::{VideoOptions, VideoRequest};

# async fn run() -> Result<(), Box<dyn std::error::Error>> {
let client = VideoOptions::default().create("fal:///fal-ai/wan/v2.2-a14b/text-to-video")?;
let response = client.generate_video(&VideoRequest {
    prompt: "A slow cinematic push-in on a brass astrolabe".to_string(),
    ..VideoRequest::default()
}).await?;

println!("{} video(s)", response.videos.len());
# Ok(())
# }
```

## LLM Usage

```rust
use rath::llm::{ClientOptions, ClientOutput, Message};

# async fn run() -> Result<(), Box<dyn std::error::Error>> {
let client = ClientOptions::default().create("openai:///gpt-4o")?;
let response = client.execute(&[Message::user("Write one sentence about Rust.")]).await?;

match response.output {
    ClientOutput::Output(value) => println!("{value}"),
    ClientOutput::ToolCalls { .. } => println!("model requested tools"),
}
# Ok(())
# }
```

## License

Licensed under either the MIT License or Apache License 2.0, at your option.
