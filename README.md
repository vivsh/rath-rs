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

### Fal speech

Fal supports these explicitly mapped endpoints:

| Capability | Model URL |
|---|---|
| TTS | `fal:///fal-ai/kokoro/american-english` |
| TTS | `fal:///fal-ai/elevenlabs/tts/turbo-v2.5` |
| TTS | `fal:///fal-ai/elevenlabs/tts/eleven-v3` |
| STT | `fal:///fal-ai/wizper` |
| STT | `fal:///fal-ai/elevenlabs/speech-to-text/scribe-v2` |

Set `FAL_KEY`, or select another credential variable with `api_key_env`.

Eleven v3 accepts inline audio tags in `TtsRequest.input`, for example
`[whispers] Stay close. [laughs] I was only teasing.` Rath passes these unchanged;
it does not generate delivery cues. Use `voice: Some("Rachel".into())` for an
explicit female voice, or omit it for the endpoint default. Native v3 settings
such as `stability` go in `provider_config`; no settings are added implicitly.
Audio tags are model-specific and should not be assumed to work with Turbo.

```rust
use rath::audio::tts::{TtsOptions, TtsRequest};
use rath::audio::stt::{SttOptions, SttRequest};

# async fn run() -> Result<(), rath::core::RathError> {
let tts = TtsOptions::default().create("fal:///fal-ai/kokoro/american-english")?;
let speech = tts.synthesize_speech(&TtsRequest {
    input: "Hello from Rath.".into(),
    voice: Some("af_heart".into()),
    ..Default::default()
}).await?;

let stt = SttOptions::default()
    .create("fal:///fal-ai/elevenlabs/speech-to-text/scribe-v2")?;
let transcript = stt.transcribe_audio(&SttRequest {
    mime_type: speech.mime_type,
    data: speech.data,
    model: None,
    provider_config: Some(serde_json::json!({"language_code": "en"})),
}).await?;
println!("{}", transcript.text);
# Ok(())
# }
```

Both operations submit once to Fal's queue, poll every five seconds and have a
five-minute I/O deadline. Rath does not automatically resubmit failures. Timing
out or dropping the future stops local waiting; the remote job may continue.
There is no persistent job tracking or resume API for audio.

TTS downloads the generated audio into `TtsResponse.data`. STT sends the supplied
bytes as a base64 data URI, without a separate upload. This uses additional memory
for encoding; provider file-size limits still apply. Use an `audio/*` MIME type
without parameters. Rath does not transcode files. Fal results are preserved in
`raw_metadata`, including timestamps or speaker information when available.

`provider_config` must be an object. Request configuration overrides client
configuration, then typed input/audio and any supplied voice take precedence.
Other settings retain the endpoint's native names and defaults: for example,
Wizper uses `language` (default `en`, explicit `null` for detection), while Scribe
uses `language_code`. `request.model` selects a supported endpoint for that call
without changing the client. Unknown endpoints fail locally. The supported TTS
endpoints do not expose format selection, so leave `format` unset.

The default queue is `https://queue.fal.run`; a custom `base_url` is used as the
queue base without rerouting to Fal. Status/result URLs must share that origin.
Audio downloads never receive the API key. Redirects are rejected, so custom
services must provide direct queue and download URLs. Fal failures use Rath's shared
error contract: useful provider messages and status remain visible, while sanitized
response bytes require explicit access through `response_body()`. Failed jobs retain
their message and metadata. These diagnostics and `raw_metadata` can contain private
content; they are not automatically suitable for public HTTP responses.

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

For example, Gemini safety settings can be provided with conservative
`safetySettings`:

```rust
use serde_json::json;
use rath::llm::LlmOptions;

# fn run() -> Result<(), Box<dyn std::error::Error>> {
let client = LlmOptions::default()
    .with_provider_config(json!({
        "safetySettings": [
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_LOW_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }
        ]
    }))
    .create("gemini:///gemini-2.5-flash")?;
# Ok(())
# }
```

## Structured Output Strategy

Rath uses native structured output where providers support it cleanly. For
Ollama, and Gemini models before `gemini-3.1`, Rath may convert structured
output into a required synthetic tool call. This is internal to Rath: callers
still receive `LlmOutput::Output(...)` when the exit tool is called successfully.

Use `LlmClient::uses_exit_tool()` to check whether a client uses this strategy.

## License

Licensed under either the MIT License or Apache License 2.0, at your option.

## Request measurement and response limits

All five LLM adapters (Gemini, OpenAI, Anthropic, OpenRouter and Ollama) support
local text-request estimates and `LlmOptions::with_max_output_tokens`.

```rust,no_run
use rath::llm::{LlmOptions, Message, RathError};

# async fn example(model_url: &str, retained_messages: &[Message]) -> Result<(), RathError> {
let client = LlmOptions::default()
    .with_preamble("Answer using the supplied conversation.")
    .with_max_output_tokens(1024)
    .create(model_url)?;

// Includes persona, retained history, current input, schemas and enabled tools.
let estimate = client.estimate_tokens(retained_messages)?;
println!("estimated input: {}", estimate.input_tokens);

// Same selected response model; excludes configured persona, history and schemas.
// Includes minimal one-message request framing: compare directly to a summary budget.
let summary_size = client.estimate_content_tokens("A standalone summary.")?;
println!("summary budget usage: {}", summary_size.input_tokens);

// Explicit network request. Unsupported adapters/endpoints return an error.
let reported = client.count_tokens(retained_messages).await?;
println!("provider-reported input: {}", reported.input_tokens);
# Ok(())
# }
```

Local estimates support inexpensive monitoring and early compaction. **They do
not establish that a request fits a model's context window.** Admission requiring
model-specific counts must explicitly use a provider counting API or a matching
deployment tokenizer. Rath does not bundle matching deployment tokenizers;
budgeting for arbitrary local models is best effort.

Estimates use a fixed `cl100k_base` reference vocabulary and ordinary text
encoding, not a guessed tokenizer based on the model name. The prompt projection
includes adapter formatting and schema/tool instructions. The allowance is
`ceil(1.25 * (reference_tokens + 256 + 16 * (wire_messages + tool_definitions)))`.
This deliberately padded heuristic can overestimate substantially, especially
for short inputs, and can still underestimate unfamiliar models or workloads.
It is not a guaranteed upper bound. Media attachments return unsupported locally.

| Adapter | Explicit `count_tokens` / `count_content_tokens` | Generation cap |
|---|---|---|
| Gemini | Full `generateContentRequest` through `countTokens` | `maxOutputTokens` |
| OpenAI | `/responses/input_tokens` | `max_output_tokens` |
| Anthropic | `/messages/count_tokens` | `max_tokens` |
| OpenRouter | Unsupported; local estimate available | `max_tokens` |
| Ollama | Unsupported; local estimate available | `max_tokens` |

Native counting uses the configured endpoint, model and credentials. It never
falls back to generation or estimation. Provider counts have source
`TokenCountSource::ProviderReported`; local estimates use `Estimated`. Provider
counts can differ from actual generation usage and are not labelled exact.
Unsupported media/source forms are rejected rather than silently omitted.

`count_content_tokens` is the native counterpart to `estimate_content_tokens`.
Both exclude persona, schemas, tools, history, thinking and output caps, but
include minimal-request framing. Never measure summary content by subtracting
two full-request estimates. Use the client for the model that will consume it.

The caller owns admission, context limits, output reservation and compaction.
`execute` never automatically counts or modifies history. Do not record estimates
as `TokenUsage`: that type continues to represent generation usage from providers.

Output caps are validated at construction and request preparation. Zero and
values outside the adapter's representable range fail. Provider-specific model
maximums are validated by the deployment. A cap includes reasoning tokens where
the provider includes them, so it does not promise that many visible answer
tokens. When unset, previous defaults remain, including Anthropic's 4096.

Use the typed cap exclusively. Rath rejects reserved `provider_config` keys
`max_tokens`, `max_completion_tokens`, `max_output_tokens`, `maxOutputTokens`,
`num_predict`, `generationConfig.maxOutputTokens`,
`generation_config.max_output_tokens`, and `options.num_predict`, even when the
typed field is absent or contains the same value. Schema properties with those
names are not treated as configuration. Rejected caps are never dropped or retried.

When the provider reports token-limit termination, Rath returns
an error with `kind() == ErrorKind::OutputLimitReached` before parsing generated JSON
or exposing tool calls. Partial output is not successful structured output. The
sanitized response remains explicitly accessible through `response_body()`, while
ordinary `Display`, `Debug` and tracing of Rath errors omit body contents.

Measurement methods still default to unsupported for custom `LlmClient` implementations.
Exhaustive `LlmOptions` literals must add `max_output_tokens: None` or use
`..Default::default()`. The structured error redesign is a breaking API change;
custom error construction and enum matches must migrate as described below.
`LlmResponse` and `TokenUsage` are unchanged.

For an offline comparison against sourced documentation examples, run
`cargo test reports_published_count_comparison -- --nocapture`. Each fixture records
its model, request projection, published count and source. These illustrative
examples are not live captures or a representative calibration corpus; their
error/underestimation report does not establish context safety.

## Application message keys

Attach an optional opaque identity to a message for storage and correlation:

```rust
use rath::llm::Message;

let message = Message::user("Hello").with_key("message:42");
assert_eq!(message.key.as_deref(), Some("message:42"));
```

`Message::key` is `Option<String>`. Constructors leave it unset, and absent keys
are omitted from message serialization. Older stored messages remain readable;
keys survive serialization, cloning and other message builders. The caller owns
assignment and uniqueness: Rath does not generate keys or deduplicate messages.

Keys are application metadata. Provider adapters omit them from generation and
counting requests, so they consume no prompt tokens. They are separate from the
tool-call IDs used to correlate calls and results. Exhaustive Rust `Message`
literals must add `key: None` (or an application key).


## Error reporting and migration

`RathError` is now a structured diagnostic with private fields. Import `RathError`,
`ErrorKind` and `ErrorBody` from `rath` (also available through `core` and `llm`).
Replace enum-pattern matching with `kind()` and borrowing accessors:

```rust
use rath::{ErrorKind, RathError};

fn inspect(error: &RathError) {
    eprintln!("{error}"); // Context, provider message and distinct causes; never the body.
    if error.kind() == ErrorKind::OutputLimitReached {
        // Partial output is diagnostic data, never a successful result or tool call.
        if let Some(body) = error.response_body() {
            let bytes: &[u8] = body.bytes();
            let complete = body.is_complete();
            // Inspect bytes explicitly in an access-controlled diagnostic workflow.
            // `complete` describes the transfer, not whether generation finished.
            let _ = (bytes, complete);
        }
    }
    let _metadata = (
        error.provider(), error.operation(), error.http_status(),
        error.provider_code(), error.request_id(), error.retry_after(),
    );
    let mut level = Some(error);
    while let Some(cause) = level {
        // Each message is separate; std::error::Error::source() also traverses this chain.
        let _message = cause.message();
        level = cause.source();
    }
}
```

`ErrorKind` classifies validation, invalid URLs, unsupported capabilities, transport,
timeout, HTTP rejection, provider-reported failure, serialization/deserialization,
invalid responses, output limits, token counting and otherwise unclassified causes.
Network counting failures retain their HTTP/transport classification; endpoint absence
uses `UnsupportedCapability` without discarding evidence. Model-not-found remains a
provider HTTP failure rather than an unsupported endpoint. Do not parse display text
to make retry decisions; Rath does not add retries or fallbacks because of diagnostics.

`ErrorBody::Complete(Vec<u8>)` retains a complete response (or serialized SDK output).
`Incomplete(Vec<u8>)` retains bytes received before a failed transfer. Neither is
implicitly decoded to UTF-8. Old `Deserialize { raw, source }` access becomes
`response_body()` and `source()`. Old `OutputLimitReached { provider, response }`
access becomes `provider()` and `response_body()`; explicitly deserialize its bytes
if the provider response is JSON. Native SDK/HTTP error downcasts are replaced by a
stable chain of Rath errors. Body bytes reflect credential sanitization, and metadata
unavailable at a provider/SDK boundary is represented by `None`.

Custom clients create errors with the same contract:

```rust
use rath::{ErrorKind, Provider, RathError};

let error = RathError::new(ErrorKind::Transport, "request dispatch failed")
    .with_context(Provider::OpenAi, "generation")
    .with_source(RathError::new(ErrorKind::Other, "connection refused"));
assert_eq!(error.source().unwrap().message(), "connection refused");
```

`with_context` fills missing context; different existing operation context is retained
as a cause rather than overwritten. `with_source` appends without dropping an existing
chain. `Display` includes distinct messages in cause order; `Debug` includes metadata
and body presence/completeness but never body contents. Rath no longer automatically
logs raw model output on structured-output parse failures.

Built-in adapters remove known credentials, common URL/JSON-escaped credential forms,
authorization/cookie/secret fields and URL user information, query values and fragments
before returning diagnostics. Replacements use `[REDACTED]`; remaining body bytes are
not intentionally truncated. This is deterministic protection for known credentials
and credential-bearing locations, not a detector of arbitrary private prose or unknown
secrets. Provider messages can quote user content and remain visible. Custom clients
must avoid passing credential values in arbitrary message text; `new` sanitizes known
credential locations but cannot know a custom client's credentials.

### Gemini SDK boundary

Rath uses the official `gemini-rust` 2.1.0 release with its existing `generateContent`
API. Rath retains everything that boundary exposes, but it cannot recover response
headers, unknown decoded JSON fields or original bytes discarded by the SDK. HTTP
failures expose status and an optional UTF-8 description; failed body reads and
incomplete bytes are discarded internally, and text decoding can replace invalid
UTF-8. Successful-response JSON decode failures expose causes without the raw body.
Partial generation output is serialized from the SDK's typed response, not the
original wire bytes.

The SDK's `check_response` has tracing `err` instrumentation which can log its raw
HTTP error description before Rath sanitizes it; request spans can also include URLs.
Rath does not install global logging filters or modify the SDK. Therefore Rath's safe
error-formatting guarantee does not cover the SDK's own logging. For default Gemini
credentials Rath uses the current environment value when normalizing errors; if the
environment changes after construction, the SDK does not expose its previously used
credential for Rath to redact. Explicit `api_key_env` resolution stays available in
the existing model URL. No extra credential copy is retained for diagnostics.
