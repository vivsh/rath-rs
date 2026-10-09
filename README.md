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
- `rath::audio`: speech and voice clients, backend traits and types
- `rath::registry`: application-owned factories for all eight capabilities

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
gemini:///gemini-3.8-flash-tts
gemini:///gemini-3.1-flash-image
ollama:///qwen3:8b?base_url=http://localhost:11434
openai:///gpt-4o?base_url=https://api.example.com/v1
```

Use `rath::core::ModelUrl` for parsed URLs. `rath::llm::LlmUrl` remains as a
compatibility alias.

OpenRouter model slugs keep their provider prefix in the URL path, for example
`openrouter:///anthropic/claude-sonnet-4.5`.

OpenRouter accepts thinking through `.with_thinking(Some(ThinkingLevel::High))`
or a URL such as `openrouter:///vendor/model?thinking=high`. An explicit URL value
overrides the builder. Rath sends `reasoning.effort`: Off maps to `none`, and
Low/Medium/High/XHigh map to `low`/`medium`/`high`/`xhigh`. Unspecified thinking
omits the control and leaves the provider default intact. This requests generation
behavior rather than merely hiding reasoning text. Supported levels depend on the
selected model and upstream provider; rejections retain typed diagnostics, without
a retry at a weaker setting. See [OpenRouter's reasoning documentation](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens).

Fal model slugs also keep the full path, for example
`fal:///fal-ai/flux/schnell`.

Model locators are not provider HTTP URLs. `openai:///gpt-4o` means "use the
OpenAI adapter with model `gpt-4o`"; `base_url` is the only place for a custom
provider endpoint.

Capability options parse the same URL format and dispatch to the provider
implementation that supports that capability.

## Provider factories and external adapters

All eight capabilities use one pattern: an application-owned `ProviderFactory`
constructs a concrete capability client, which executes through a backend.
`ProviderRegistry::new()` is external-only; `with_builtins()` also enables Rath's
built-in adapters. Registration is synchronous and consuming; construction is async.

```rust
use rath::RathError;
use rath::audio::tts::{TtsClient, TtsOptions};
use rath::registry::{ProviderFactory, ProviderRegistry};

async fn speech_client(provider: impl ProviderFactory + 'static) -> Result<TtsClient, RathError> {
    let providers = ProviderRegistry::with_builtins().register("acme", provider)?;
    providers
        .tts("acme:///speech-model", TtsOptions::default())
        .await
}
```

Use `llm`, `embeddings`, `image`, `video`, `tts`, `stt`, `voice_design` or
`voice_clone`. Factories own construction dependencies; clients survive registry
disposal. Share either explicitly through `Arc`. Rath retains no application history,
media, job state or global registrations. There is no discovery, retry or fallback
introduced by the registry.

Registration names match `[a-z][a-z0-9.-]*`. Duplicates and built-in names/aliases
(including `fastembed`, even without the Cargo feature) fail. External URLs support
nested model paths, `api_key_env` and `base_url`; external LLM URLs additionally
support the same temperature/thinking/cache controls and URL-over-options precedence.
Other capabilities reject LLM-only controls. Model-path percent sequences retain
Rath's literal escaping convention. Missing routes/capabilities are
`UnsupportedCapability`; malformed locators are `InvalidUrl`.

### Writing an extension

Implement a public `…Backend` trait using ordinary async methods and return
`…Client::from_backend(backend)` from `ProviderFactory`. Public traits use RPITIT
with `Send` futures, with no `async_trait` annotation or public boxed-future API.
The private erased bridge boxes once per asynchronous dispatch. Direct generic
backend calls have no Rath future-boxing requirement. `from_backend` performs no I/O.
HTTP factories do not probe providers or generate; local inference construction
may prepare resources as explicitly configured.

Every factory method defaults to unsupported. A transparent decorator must override
changed methods and explicitly forward all seven others. See the fully forwarding,
compile-tested [factory decorator](examples/factory_decorator.rs).

Use `ProviderRegistry::with_builtin_factory(factory)` to replace or decorate
construction for all built-in schemes, including their aliases:

```rust
use rath::RathError;
use rath::llm::{LlmClient, LlmOptions};
use rath::registry::{ProviderFactory, ProviderRegistry};

async fn customized_llm(factory: impl ProviderFactory + 'static) -> Result<LlmClient, RathError> {
    let providers = ProviderRegistry::with_builtin_factory(factory);
    providers.llm("ollama:///model", LlmOptions::default()).await
}
```

Pass a decorator around `BuiltinProviderFactory` to retain built-in behavior for
forwarded capabilities. The registry validates URLs and resolves LLM URL-over-options
precedence before invoking your factory. An unsupported result is returned directly;
there is no automatic fallback. External registrations still select their own factories,
and standalone `Options::create()` continues to use Rath's default factory.

Use `ModelUrl` accessors (`provider()`, `model()`, `api_key()`, `base_url()` and LLM
controls). Its fields are private and Debug redacts credentials and URL query data.
Use `RathError::from_error(kind, &cause, secrets)` and credential-aware body/metadata
builders to preserve causes. Construction redacts URL-resolved credentials; extension
backends remain responsible for their own private credentials and execution errors.
No wrapper intercepts third-party execution. Ordinary diagnostics never print bodies.

`Provider::External(String)` retains `{"External":"acme"}` Serde representation,
including video receipts. Built-in representations are unchanged. `as_str()` borrows
`&str`, not a static lifetime. Extensible enum matches need a wildcard. Older persisted
readers still need migration before reading unknown identities.

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

## Embedding usage

Every model is bound at construction. A homogeneous batch produces one nonempty,
finite vector per input, in input order, or an error for the whole call. Remote
adapters send one batch request, without implicit fan-out. Usage is aggregate and
provider-reported. Rath never slices vectors to emulate requested dimensions.

```rust
use rath::embeddings::{EmbedRequest, EmbeddingOptions};
async fn run() -> Result<(), rath::RathError> {
    let client = EmbeddingOptions::default()
        .create("openai:///text-embedding-3-small")
        .await?;
    let response = client
        .embed(&EmbedRequest::batch([
            "first document".into(),
            "second document".into(),
        ]))
        .await?;
    assert_eq!(response.embeddings.len(), 2);
    Ok(())
}
```

`EmbedRequest::new(text)` builds a single-input batch. Empty batches/strings, zero
dimensions and blank instructions fail. Supplied titles require RetrievalDocument;
instructions require a supported RetrievalQuery task. Native configuration is shallow
client defaults → request settings → supplied typed fields; unknown embedding native
keys and typed aliases fail locally.

| Control | OpenAI | Gemini | Ollama | FastEmbed |
| --- | --- | --- | --- | --- |
| Task hints | rejected | all typed tasks | Qwen3 query/document | BGESmallENV15 query/document |
| Title | rejected | retrieval document | rejected | rejected |
| Instructions | rejected | rejected | required for Qwen queries | optional BGE query replacement |
| Dimensions | forwarded | forwarded | forwarded | native dimension only |
| Explicit truncation | rejected | autoTruncate | truncate | actual tokenizer; defaults off |
| Native settings | user | empty only | keep_alive, object options | empty only |

Without task hints, raw input is preserved. Recognized `qwen3-embedding[:tag]`
queries use `Instruct: {instructions}\nQuery: {text}`. Custom aliases are not guessed;
callers can submit already prepared raw text. BGE queries receive its documented
retrieval prefix unless explicitly replaced. Documents remain unchanged.

Gemini uses direct batch HTTP to retain current `embedContentConfig` and usage metadata;
its generation SDK is unchanged. See [Gemini embeddings](https://ai.google.dev/api/embeddings).

### Optional FastEmbed CPU inference

```rust
use rath::embeddings::{EmbedRequest, EmbeddingOptions, FastEmbedOptions};
async fn run() -> Result<(), rath::RathError> {
    let client = EmbeddingOptions::default()
        .with_fastembed(FastEmbedOptions::new("./models").with_intra_threads(2))
        .create("fastembed:///BGESmallENV15")
        .await?;
    let response = client.embed(&EmbedRequest::new("hello")).await?;
    Ok(())
}
```

Enable Cargo feature `fastembed` for CPU ONNX inference. The typed setup remains
available without it, but construction then reports UnsupportedCapability. Defaults
are downloads off, revision `main`, max length 512, one intra-op thread and batch size
32. Cache location is mandatory. Other embedding adapters reject FastEmbed setup;
there is no JSON setup path. Native dimensions cannot be changed locally.

The cache follows hf-hub layout at the explicit directory. Offline construction fails
on missing files without network requests. `.with_download(true)` explicitly permits
retrieving missing artifacts and pins them to one revision, with retrieval retries
disabled. Only explicit `api_key_env` supplies a download credential; ambient HF tokens
are not read. Optional `base_url` selects a compatible hub endpoint. Files and cleanup
belong to the application; there is no global model/session cache.

Enabling the feature can acquire ONNX Runtime binaries at **build time** independently
of runtime model-download permission. FastEmbed and ONNX are absent from the default
dependency graph; no Candle, GPU, sparse embeddings or reranking features are enabled.

Initialization and inference use Tokio blocking workers. Calls share an authoritative
async session mutex. Dropping a future stops waiting but cannot interrupt native work
or downloads; it never permits overlapping session use or resubmission. A Tokio runtime
is required. hf-hub/FastEmbed may discard native diagnostic details internally; Rath
retains their exposed cause chains but cannot recover discarded response bodies.

Normal tests use an original tiny ONNX/tokenizer fixture, not downloaded model weights.
Opt-in real-model probes take explicit arguments and print initialization/inference
latency, dimensions and batch size:

```sh
cargo run --features fastembed --example embedding_probe -- fastembed:///BGESmallENV15 ./models main
cargo run --example embedding_probe -- ollama:///qwen3-embedding:0.6b
```

Add `--download` as the final FastEmbed argument only when model retrieval is intended.
The Ollama model must already be installed; Rath does not pull or restart it.

## Image Usage

```rust
use rath::images::{ImageOptions, ImageRequest};

async fn run() -> Result<(), rath::RathError> {
    let client = ImageOptions::default()
        .create("fal:///fal-ai/flux/schnell")
        .await?;
    let response = client
        .generate_image(
            &ImageRequest::new("A walnut desk lamp in warm studio light".to_string())
                .with_size("landscape_4_3".to_string()),
        )
        .await?;

    println!("{} image(s)", response.images.len());
    Ok(())
}
```

### Gemini images

Gemini image generation reuses `GEMINI_API_KEY`, including when chat, STT and TTS
already use that key. Select a dedicated image model:

| Model URL | Explicit `size` |
| --- | --- |
| `gemini:///gemini-2.5-flash-image` | Omit; provider chooses resolution |
| `gemini:///gemini-3.1-flash-lite-image` | `1K` |
| `gemini:///gemini-3.1-flash-image` | `512`, `1K`, `2K`, `4K` |
| `gemini:///gemini-3-pro-image` | `1K`, `2K`, `4K` |
| `gemini:///gemini-nano-banana-2.1` | `1K`, `2K`, `4K` |

```rust,no_run
use rath::images::{ImageOptions, ImageRequest};
use serde_json::json;

async fn run() -> Result<(), rath::RathError> {
    let client = ImageOptions::default()
        .create("gemini:///gemini-3.1-flash-image")
        .await?;
    let response = client.generate_image(
        &ImageRequest::new("A walnut desk lamp in warm studio light".into())
            .with_size("2K".into())
            .with_provider_config(json!({
                "generationConfig": {"imageConfig": {"aspectRatio": "16:9"}}
            })),
    ).await?;
    println!("{} image(s)", response.images.len());
    Ok(())
}
```

Results use `ImageData::Base64` with the provider's PNG, JPEG or WebP MIME type.
Rath checks base64 and file signatures; it does not fully decode image pixels.
Native configuration supports `generationConfig.imageConfig` (`aspectRatio`,
`imageSize`) and `safetySettings`. Client defaults merge shallowly with request
settings; typed `size` wins. Ordinary aspect ratios are `1:1`, `2:3`, `3:2`,
`3:4`, `4:3`, `4:5`, `5:4`, `9:16`, `16:9`, `21:9`. Flash Image and Nano Banana 2.1
also accept `1:4`, `4:1`, `1:8`, `8:1`. Unknown controls, unsupported sizes/models,
empty prompts and missing/blocked/unfinished images fail without retries.
Text accompanying images remains in `raw_metadata`; thought images are excluded.

This adapter provides text-to-image generation. The existing `ImageRequest` has
no typed reference-image input, so reference editing, multi-turn editing and
grounding tools are not exposed through provider JSON. Model availability and
visual fidelity require separately authorized live validation. Wire mappings use
Google's [GenerateContent API](https://ai.google.dev/api/generate-content) and
[image-generation documentation](https://ai.google.dev/gemini-api/docs/image-generation).

## Video Usage

`VideoOptions::create(url)` selects the provider and native endpoint just like
the LLM and voice APIs. Gemini/Veo and Fal have built-in video adapters. Construction
does not discover models or start generation.

Typed Fal inputs have verified mappings for these endpoints:

| Input | Fal endpoint |
| --- | --- |
| `TextToVideo` | `fal-ai/kling-video/v3/pro/text-to-video` |
| `TextToVideo` | `fal-ai/wan/v2.2-a14b/text-to-video` |
| `ImageToVideo` | `fal-ai/kling-video/v3/pro/image-to-video` |
| `MotionTransfer` | `fal-ai/kling-video/v3/pro/motion-control` |

A mismatched typed operation returns `UnsupportedCapability` before submission.
Other Fal queue endpoints require explicit `VideoInput::Native`.

### Gemini / Veo generation and optional callbacks

Veo uses the same `GEMINI_API_KEY` as Gemini chat, speech and images. Supported
model URLs are `gemini:///veo-3.1-generate-preview`,
`gemini:///veo-3.1-fast-generate-preview` and
`gemini:///veo-3.1-lite-generate-preview`. Text-to-video and image-to-video use
the existing submit, get, wait and generate methods. Image-to-video requires
`ImageData::Base64` with standard base64 PNG/JPEG bytes, optionally with an end
frame; input URLs are not fetched. Encoded images must be nonempty and at most
28 MB each. Motion transfer, Native payloads, reference images and extension
are unsupported by this adapter.

```rust,no_run
use rath::video::{VideoInput, VideoOptions, VideoRequest};
use serde_json::json;

async fn generate() -> Result<(), rath::RathError> {
    let client = VideoOptions::default()
        .with_provider_config(json!({"parameters": {
            "durationSeconds": 4,
            "aspectRatio": "16:9",
            "resolution": "720p"
        }}))
        .create("gemini:///veo-3.1-fast-generate-preview")
        .await?;
    let job = client.submit_video(&VideoRequest::new(
        "A bluebird flies through a sunlit garden. Gentle birdsong.".into(),
        VideoInput::TextToVideo,
    )).await?;
    let result = client.wait_video(&job.id).await?;
    // Download and persist result.videos in your application.
    assert!(!result.videos.is_empty());
    Ok(())
}
```

Native controls belong in `provider_config.parameters`: `durationSeconds`
(integer 4/6/8), `aspectRatio` (`16:9`/`9:16`), `resolution`
(`720p`/`1080p`/`4k`), `sampleCount` (1), `personGeneration`
(`allow_all`/`allow_adult`), `negativePrompt` (string), and `enhancePrompt`
(boolean). Request-level `parameters` replaces the client-level object.
1080p/4k requires eight seconds; Lite excludes 4k. Image inputs exclude
`personGeneration: "allow_all"`; Google enforces regional and safety rules.
Audio is generated by Veo; there is no separate audio toggle.

`VideoData::Url` holds Google's generated video URI, which requires API-key
authentication to download. Validate that the download destination is Google's
trusted service before attaching a key, and do not forward credentials to
another origin on redirects. Save videos within Google's two-day retention
window. Rath returns locations and never downloads or persists videos.

To receive optional **project-registered** completion callbacks, register your
HTTPS endpoint with Google's Webhook Service for `video.generated`. Keep the
returned signing secret in an environment variable and configure its name:

```rust,no_run
use rath::video::{VideoClient, VideoEvent, VideoOptions};
use serde_json::json;

async fn webhook_client() -> Result<VideoClient, rath::RathError> {
    VideoOptions::default()
        .with_provider_config(json!({
            "webhook_signing_secret_env": "GEMINI_WEBHOOK_SIGNING_SECRET"
        }))
        .create("gemini:///veo-3.1-fast-generate-preview")
        .await
}

async fn receive(client: &VideoClient, headers: &http::HeaderMap, original_body: &[u8])
    -> Result<VideoEvent, rath::RathError>
{
    client.parse_webhook(headers, original_body).await
}
```

The parser verifies Standard Webhooks HMAC-SHA256 signatures over the original
body, rejects duplicate/missing signed headers and timestamps outside ±300
seconds, then decodes `video.generated`. It returns the operation ID and a
`Succeeded` status with the output URI. Successful response metadata contains
`event` (the authenticated envelope) and `webhook_id` (the signed delivery ID)
for application-owned deduplication. Registration, HTTP serving, durable
acceptance, job/user authorization and downloads belong to the application.
Missing verification configuration returns `UnsupportedCapability`;
`webhook_signing_secret_env` is never sent to Gemini or overridden by a request.
The parser reads that variable when called, so provision the returned secret
through your application's configuration. It does not register or rotate secrets.

This adapter supports registered callbacks; request-level `webhook_url` and
dynamic JWT callbacks return `UnsupportedCapability`. Polling remains available
for reconciliation and failures without a documented video failure event.
See Google's [Veo guide](https://ai.google.dev/gemini-api/docs/veo) and
[webhook guide](https://ai.google.dev/gemini-api/docs/webhooks), and the
compile-checked [Gemini video example](examples/gemini_video.rs).

### Fal typed generation and polling

```rust,no_run
use rath::images::ImageData;
use rath::video::{VideoInput, VideoOptions, VideoRequest};

async fn run() -> Result<(), rath::RathError> {
    let client = VideoOptions::default()
        .create("fal:///fal-ai/kling-video/v3/pro/image-to-video")
        .await?;
    let job = client
        .submit_video(&VideoRequest::new(
            "The character turns and waves.".into(),
            VideoInput::ImageToVideo {
                start_image: ImageData::Url {
                    url: "https://assets.example/approved-scene.png".into(),
                },
                end_image: None,
            },
        ))
        .await?;

    // Persist the receipt and provider/model/account association in your application.
    let status = client.get_video(&job.id).await?;
    let response = client.wait_video(&job.id).await?;
    Ok(())
}
```

For T2V, use `VideoInput::TextToVideo` (the default) and a compatible T2V endpoint.
Typed T2V/I2V require a nonblank prompt. The I2V image establishes the opening
shot; it is not interchangeable with a plain-background character reference.

`generate_video(&request)` submits once and waits. Polling defaults to five-second
intervals. Each Fal video HTTP request has a 60-second timeout, but total waiting
has no deadline. Apply your own async deadline if needed. Dropping a future does
not cancel the remote job, and an uncertain submission must not be blindly retried.

### Motion transfer

```rust,no_run
use rath::images::ImageData;
use rath::video::{VideoData, VideoInput, VideoOptions, VideoRequest};

async fn run() -> Result<(), rath::RathError> {
    let client = VideoOptions::default()
        .create("fal:///fal-ai/kling-video/v3/pro/motion-control")
        .await?;
    let response = client
        .generate_video(&{
            let mut value = VideoRequest::default();
            value.input = VideoInput::MotionTransfer {
                character_image: ImageData::Url {
                    url: "https://assets.example/character.png".into(),
                },
                driving_video: VideoData::Url {
                    url: "https://assets.example/performance.mp4".into(),
                },
            };
            value.provider_config = Some(serde_json::json!({
                "character_orientation": "video",
                "keep_original_sound": false
            }));
            value
        })
        .await?;
    Ok(())
}
```

The motion prompt is optional. Kling requires an explicit `character_orientation`
of `"image"` or `"video"`; Rath does not invent that choice. Model-specific controls
such as duration, audio generation, and identity elements remain native settings.

Inputs may also use `ImageData::Base64` or `VideoData::Base64` with a matching,
parameter-free MIME type and nonempty standard base64. Rath validates and encodes
data URIs without uploading or downloading input files. The provider validates
actual media dimensions, duration, and content.

### Explicit native input

```rust,no_run
use rath::video::{VideoInput, VideoOptions, VideoRequest};

async fn run() -> Result<(), rath::RathError> {
    let client = VideoOptions::default()
        .create("fal:///fal-ai/kling-video/v3/pro/image-to-video")
        .await?;
    let job = client
        .submit_video(&{
            let mut value = VideoRequest::default();
            value.input = VideoInput::Native {
                payload: serde_json::json!({
                    "start_image_url": "https://assets.example/scene.png",
                    "multi_prompt": [
                        { "prompt": "A slow push-in.", "duration": "3" },
                        { "prompt": "The character smiles.", "duration": "3" }
                    ],
                    "duration": "6"
                }),
            };
            value
        })
        .await?;
    Ok(())
}
```

Native mode also accepts arbitrary/custom Fal video endpoint paths. Its payload
must be an object, and the top-level request prompt must be empty. Rath performs
no model-specific input validation or field translation in this mode.

Merge order is client defaults → request `provider_config` → explicit typed inputs
or Native payload. Typed inputs clear competing native prompt/media aliases;
typed T2V/I2V reject `multi_prompt`. Native mode preserves its supplied fields.
Both paths require results matching Rath's existing video URL/base64 shapes;
arbitrary non-video output is not a successful video response.

### Authenticated webhooks

Set `VideoRequest.webhook_url` to a final HTTPS destination without embedded
credentials or a fragment. Rath sends it as Fal's `fal_webhook` query parameter,
not a model input. Polling is still available for the same job.

```rust,no_run
use rath::video::{VideoClient, VideoEvent};

async fn decode_callback(
    client: &VideoClient,
    headers: &http::HeaderMap,
    original_body: &[u8],
) -> Result<VideoEvent, rath::RathError> {
    client.parse_webhook(headers, original_body).await
}
```

Pass the original request bytes, not reserialized JSON. Fal verification rejects
missing/duplicate/malformed signed headers, invalid signatures, and timestamps
outside ±300 seconds. It fetches public keys from Fal's fixed JWKS endpoint per
call with a ten-second timeout, no credentials, no redirects, and no key cache.
Unsupported adapters return `UnsupportedCapability`.

After successful parsing, the application must:

1. Match `event.job_id` against its stored provider/model/account and owning user.
2. Durably accept the event and apply idempotent state transitions.
3. Acknowledge delivery only after durable acceptance; expect duplicate callbacks.

Authentication alone does not authorize a job for an application user. Rath owns
neither the HTTP receiver nor durable replay/deduplication state. Keep existing
terminal results from being overwritten by stale polling or callbacks.

Both polling and callbacks return `VideoJobStatus`. A valid Fal `ERROR` notification
is `Failed`; malformed or incomplete `OK` output, including `payload_error`, returns
an error rather than a fabricated generation failure. The parser does not follow
result URLs or resubmit; reconcile the already-submitted job through `get_video`
using your stored receipt. Results are provider-retained only temporarily.

Returned metadata/error bodies may contain private inputs or signed media URLs.
Rath does not log those bodies. Download and persist any required media in your
application; Rath has no file store, job registry, or background worker.

### Video API migration

`VideoRequest.image_url` is removed. Replace it with
`VideoInput::ImageToVideo { start_image: ImageData::Url { url }, end_image: None }`.
Use `Native { payload }` for previously untyped endpoints instead of silently
assuming typed support. Existing prompt-only Rust construction using
`..VideoRequest::default()` still works on the documented Wan T2V endpoint.
Serialized requests must include the new tagged `input` field.

Existing job/response types and polling method signatures are unchanged.
`parse_webhook` has a default unsupported implementation, so existing downstream
`VideoBackend` implementations do not need a new required method.

## Audio Usage

```rust
use rath::audio::tts::{TtsOptions, TtsRequest};
use rath::audio::voice::Voice;

async fn run() -> Result<(), rath::RathError> {
    let client = TtsOptions::default().create("openai:///tts-1").await?;
    let response = client
        .synthesize_speech(
            &TtsRequest::new("Rath is a stable capability layer for AI applications.".to_string())
                .with_voice(Voice::id("openai", "alloy"))
                .with_format("mp3".to_string()),
        )
        .await?;

    println!("{} bytes of {}", response.data.len(), response.mime_type);
    Ok(())
}
```

### Gemini speech and one shared key

Configure `GEMINI_API_KEY` once in the consumer's environment. All four capability
clients below resolve that same key without an STT- or TTS-specific credential:

```rust,no_run
use rath::llm::LlmOptions;
use rath::audio::stt::{SttOptions, SttRequest};
use rath::audio::tts::{TtsOptions, TtsRequest};
use rath::audio::voice::Voice;
use rath::images::ImageOptions;

async fn run(audio: Vec<u8>) -> Result<(), rath::RathError> {
    let _chat = LlmOptions::default().create("gemini:///gemini-3.8-flash").await?;
    let stt = SttOptions::default().create("gemini:///gemini-3.8-flash").await?;
    let tts = TtsOptions::default().create("gemini:///gemini-3.8-flash-tts").await?;
    let _images = ImageOptions::default().create("gemini:///gemini-3.1-flash-image").await?;
    let transcript = stt.transcribe_audio(
        &SttRequest::new("audio/webm;codecs=opus".into(), audio),
    ).await?;
    let speech = tts.synthesize_speech(
        &TtsRequest::new(transcript.text)
            .with_voice(Voice::id("gemini", "Kore"))
            .with_language("en-IN".into())
            .with_instructions("Speak warmly and clearly".into())
            .with_format("wav".into()),
    ).await?;
    println!("{} bytes of {}", speech.data.len(), speech.mime_type);
    Ok(())
}
```

An explicit `?api_key_env=MY_GEMINI_KEY` works identically for chat, STT, TTS and
images and overrides the default key. It never silently falls back when that
variable is missing. Clients resolve credentials at construction and retain their
bound model independently. Construction performs no generation.

STT uses Gemini audio understanding with a verbatim, original-language, text-only
transcription prompt. Use an audio-input model such as `gemini-3.8-flash`; ordinary
chat models are not presumed to support speech output. Encoded inputs support WAV,
MP3/MPEG, AIFF, AAC, OGG, FLAC, M4A, Opus and WebM. Common MIME aliases are normalized;
browser `codecs` parameters are stripped from the MIME sent to Gemini. Empty audio,
other MIME parameters, raw PCM input and nonempty native STT configuration fail
locally. Inline audio is limited to 14,000,000 bytes to stay below the documented
20 MB total request boundary after base64 encoding. Files API uploads, streaming,
diarization and timestamp schemas are not provided. Empty or malformed model text
is rejected. Prompt-based transcription accuracy needs live evaluation.

Supported TTS models are `gemini-3.8-flash-tts`, `gemini-3.8-flash-lite-tts`,
`gemini-3.1-flash-tts-preview`, `gemini-2.5-flash-preview-tts` and
`gemini-2.5-pro-preview-tts`. Current 3.8 models use structured style metadata;
legacy preview models use prompt-based delivery instructions. Rath supports the
30 documented preset IDs with scope `gemini` (for example Kore, Puck, Zephyr,
Charon, Aoede); omission selects Kore. Unsupported voices, cloning/reference
material, multi-speaker controls and unsupported model URLs fail locally.

Default output and `format="wav"` return complete RIFF/WAVE, 24 kHz mono signed
16-bit little-endian PCM. Headerless PCM receives one valid WAV header; returned
WAV containers are validated and preserved. `format="pcm"` returns the same
sample bytes without a header, with Gemini's `audio/L16;codec=pcm;rate=24000`
MIME descriptor. Other formats are rejected. Language uses the exact BCP-47 values
documented by Google's `SpeechConfig`, including `en-IN`, `en-US` and `hi-IN`.
Instructions are per-utterance style directions; they do not guarantee fidelity.

TTS native configuration supports `safetySettings` and `generationConfig` with
`temperature`, `topP`, `topK`, `seed`, and single-voice `speechConfig` using
`voiceConfig.prebuiltVoiceConfig.voiceName` and `languageCode`. Defaults merge
shallowly with request configuration, then typed fields win. Routing, credentials,
response modalities, formats, output caps and other unhandled controls are rejected.
HTTP media calls have a 300-second timeout, reject redirects and do not retry.
Failures retain redacted provider/HTTP diagnostics and response evidence; no private
media or response bodies are automatically logged. Cancellation stops local waiting
and cannot guarantee that Gemini stops remote work.

Mappings were checked against Google's [audio guide](https://ai.google.dev/gemini-api/docs/audio),
[speech guide](https://ai.google.dev/gemini-api/docs/speech-generation) and
[GenerateContent API](https://ai.google.dev/api/generate-content). Model/account
availability, transcription quality and speech quality have not been live-tested.
See [gemini_media.rs](examples/gemini_media.rs) for a compile-checked example.

### Fal speech

Fal supports these explicitly mapped endpoints:

| Capability | Model URL |
|---|---|
| TTS | `fal:///fal-ai/kokoro/american-english` |
| TTS | `fal:///fal-ai/elevenlabs/tts/turbo-v2.5` |
| TTS | `fal:///fal-ai/elevenlabs/tts/eleven-v3` |
| Voice design | `fal:///fal-ai/qwen-3-tts/voice-design/1.7b` |
| Speaker embedding | `fal:///fal-ai/qwen-3-tts/clone-voice/1.7b` |
| Custom voice TTS | `fal:///fal-ai/qwen-3-tts/text-to-speech/1.7b` |
| STT | `fal:///fal-ai/wizper` |
| STT | `fal:///fal-ai/elevenlabs/speech-to-text/scribe-v2` |

Set `FAL_KEY`, or select another credential variable with `api_key_env`.

Custom voices use the independent design and cloning APIs below. Design and clone
endpoints are no longer accepted by `TtsOptions`; cloning is no longer a method on
`TtsClient`. Every operation has the Fal audio deadline. Rath does not persist files
or play audio.

Eleven v3 accepts inline audio tags in `TtsRequest.input`, for example
`[whispers] Stay close. [laughs] I was only teasing.` Rath passes these unchanged;
it does not generate delivery cues. Use `voice: Some(Voice::id("fal/elevenlabs", "Rachel"))` for an
explicit female voice, or omit it for the endpoint default. Native v3 settings
such as `stability` go in `provider_config`; no settings are added implicitly.
Audio tags are model-specific and should not be assumed to work with Turbo.

```rust
use rath::audio::stt::{SttOptions, SttRequest};
use rath::audio::tts::{TtsOptions, TtsRequest};
use rath::audio::voice::Voice;

async fn run() -> Result<(), rath::core::RathError> {
    let tts = TtsOptions::default()
        .create("fal:///fal-ai/kokoro/american-english")
        .await?;
    let speech = tts
        .synthesize_speech(
            &TtsRequest::new("Hello from Rath.".into())
                .with_voice(Voice::id("fal/kokoro/american-english", "af_heart")),
        )
        .await?;

    let stt = SttOptions::default()
        .create("fal:///fal-ai/elevenlabs/speech-to-text/scribe-v2")
        .await?;
    let transcript = stt
        .transcribe_audio(&{
            let mut value = SttRequest::new(speech.mime_type, speech.data);
            value.provider_config = Some(serde_json::json!({"language_code": "en"}));
            value
        })
        .await?;
    println!("{}", transcript.text);
    Ok(())
}
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
uses `language_code`. The construction URL binds the endpoint for all calls.
Unknown endpoints fail locally. The supported TTS
endpoints do not expose format selection, so leave `format` unset.

The default queue is `https://queue.fal.run`; a custom `base_url` is used as the
queue base without rerouting to Fal. Status/result URLs must share that origin.
Audio downloads never receive the API key. Redirects are rejected, so custom
services must provide direct queue and download URLs. Fal failures use Rath's shared
error contract: useful provider messages and status remain visible, while sanitized
response bytes require explicit access through `response_body()`. Failed jobs retain
their message and metadata. These diagnostics and `raw_metadata` can contain private
content; they are not automatically suitable for public HTTP responses.

### Storage-free custom voices

`VoiceDesignClient`, `VoiceCloneClient` and `TtsClient` are independent capabilities.
Each has an `Options { provider_config }` factory taking a model URL. Factory construction
does not generate media or register a remote voice. Inputs and results are caller-owned.
There are no file paths, persistence callbacks, registries, caches or automatic pipelines.

```rust
use rath::audio::{
    tts::{TtsOptions, TtsRequest},
    voice_clone::{VoiceCloneOptions, VoiceCloneRequest},
    voice_design::{VoiceDesignOptions, VoiceDesignRequest},
};

async fn run() -> Result<(), rath::core::RathError> {
    let designer = VoiceDesignOptions::default()
        .create("fal:///fal-ai/qwen-3-tts/voice-design/1.7b")
        .await?;
    let previews = designer
        .design_voice(&{
            let mut value = VoiceDesignRequest::default();
            value.description = "A warm adult voice with a natural Indian English accent.".into();
            value.text =
                "Hello. It is good to hear from you. What would you like to talk about?".into();
            value.language = Some("en".into());
            value
        })
        .await?;

    // The application selects/auditions a sample. Rath never selects one automatically.
    let sample = previews
        .previews
        .into_iter()
        .next()
        .ok_or_else(|| {
            rath::core::RathError::new(rath::core::ErrorKind::InvalidResponse, "no previews")
        })?
        .sample;
    let cloner = VoiceCloneOptions::default()
        .create("fal:///fal-ai/qwen-3-tts/clone-voice/1.7b")
        .await?;
    let voice = cloner
        .clone_voice(&VoiceCloneRequest::new(vec![sample]))
        .await?;

    // Save voice in the application if desired. It is serializable, not stored by Rath.
    let tts = TtsOptions::default()
        .create("fal:///fal-ai/qwen-3-tts/text-to-speech/1.7b")
        .await?;
    let audio = tts
        .synthesize_speech(&TtsRequest::new("Welcome back.".into()).with_voice(voice))
        .await?;
    Ok(())
}
```

`Voice { scope, data }` distinguishes an `Id`, an opaque `Embedding { format, data,
reference_text }`, and `ReferenceAudio { samples }`. A scope is a compatibility
namespace, never a credential or routing URL. It must match the selected adapter/model;
provider IDs may additionally be account-specific. Reference-audio identities are
representable, but current adapters reject them rather than pretending to support them.
Changing a scope string does not convert an embedding or transfer a remote voice.

Qwen clones accept exactly one recording with an exact nonblank transcript and no
`name`. Returned embeddings use scope `fal/qwen3-tts-1.7b` and format
`qwen3-tts-1.7b/safetensors`. Native embedding fields and data-URI encoding belong to
the Fal adapter, not the application. Default `max_new_tokens` is 2048 for design
and 4096 for synthesis; native client/request settings can override these defaults.
Typed `language` accepts `en`, `zh`, `es`, `fr`, `de`, `it`, `ja`, `ko`, `pt`, `ru`.
Regional tags are not silently reduced to a base language. Typed `instructions`
work with Qwen presets, but fail for embeddings because that endpoint ignores them.
Accent descriptions are requests, not guarantees of audible accuracy.

### Native ElevenLabs voices

Set `ELEVENLABS_API_KEY`. Native endpoints are separate from Fal's ElevenLabs gateway:

| Capability | Model URLs |
| --- | --- |
| Design / explicit registration | `elevenlabs:///eleven_ttv_v3`, `elevenlabs:///eleven_multilingual_ttv_v2` |
| Instant cloning | `elevenlabs:///ivc` |
| TTS | `elevenlabs:///eleven_v3`, `elevenlabs:///eleven_multilingual_v2`, `elevenlabs:///eleven_turbo_v2_5` |

Design requires a 20–1000 character description and 100–1000 character sample text.
All returned previews include audio, transcript, scope `elevenlabs/voice-design`,
and a `registration_token`. That token is **not** a TTS ID and may expire.
Calling `designer.register_voice(&preview, "Nadia", Some(&json!({
"voice_description": "A warm and softly spoken adult voice"})))` explicitly registers
the selected preview. The native description is required; Rath neither remembers
the earlier request nor manufactures missing metadata.

Alternatively, `VoiceCloneOptions::default().create("elevenlabs:///ivc")` accepts
one or more `VoiceSample` recordings, an explicit nonblank `name`, and native
settings. Transcripts are optional for this adapter. Both cloning and registration
return `Voice::id("elevenlabs", ...)` for use in native TTS. A verification-required
result is an error with retained response evidence, not a ready voice.

Native TTS accepts MP3 formats `mp3_44100_128` (default) and `mp3_22050_32`.
Typed language/instructions are currently unsupported for native ElevenLabs and
the existing OpenAI adapter; explicit requests fail locally. Native settings remain
available through `provider_config`. Fal Kokoro/ElevenLabs also reject these typed
controls. Rath does not translate arbitrary descriptions into audio tags.

Remote registration/cloning can create provider-owned resources. Rath performs no
automatic registration, retries, deletion, verification, training or local storage.
The application owns remote cleanup decisions. Calls have a five-minute HTTP deadline;
a timeout/cancellation may leave a remote resource created. Inspect retained error
evidence before retrying. No credentials are serialized into Voice/VoicePreview.

Migration: replace string `TtsRequest.voice` values with scoped `Voice::id(...)`;
move old `TtsClient::clone_voice` calls to `VoiceCloneClient`; pass returned `Voice`
to TTS instead of constructing model-specific JSON. Public request changes are
breaking; no automatic conversion or provider fallback is performed.

## LLM Usage

```rust
use rath::llm::{LlmOptions, LlmOutput, Message};

async fn run() -> Result<(), rath::RathError> {
    let client = LlmOptions::default().create("openai:///gpt-4o").await?;
    let response = client
        .execute(&[Message::user("Write one sentence about Rust.")])
        .await?;

    match response.output {
        LlmOutput::Output(value) => println!("{value}"),
        LlmOutput::ToolCalls { .. } => println!("model requested tools"),
    }
    Ok(())
}
```

### Streaming text and tool calls

Use the same client, messages and options with `execute_stream`. Add
`futures-util = "0.3"` to your application to consume the standard Rust stream.

```rust
use futures_util::StreamExt;
use rath::llm::{LlmClient, LlmEvent, LlmResponse, Message};

async fn stream_answer(client: &LlmClient) -> Result<LlmResponse, rath::RathError> {
    let mut events = client.execute_stream(&[Message::user("Hello")]).await?;
    while let Some(event) = events.next().await {
        match event? {
            LlmEvent::TextDelta { text } => print!("{text}"),
            LlmEvent::ToolCallDelta { index, name_delta, arguments_delta, .. } => {
                // Append these provisional fragments to this index's tool preview.
                println!("Tool {index}: {name_delta:?} {arguments_delta}");
            }
            LlmEvent::Completed { response } => return Ok(response),
            _ => {}
        }
    }
    Err(rath::RathError::new(
        rath::ErrorKind::InvalidResponse,
        "stream ended without completion",
    ))
}
```

OpenAI, Anthropic, OpenRouter, Ollama and Gemini support this API through their
existing endpoints. Tool fragments may interleave: correlate them by the stable
`index`, which may be sparse and is independent of the final calls vector.
An available `id` is a complete ID and may repeat unchanged. Names and argument
strings are append-only; argument fragments need not be valid JSON. The current
Gemini SDK supplies complete function-call parts, so its tool event can contain
the entire argument JSON in one fragment. Rath's private final-output tool stays
hidden from progress events.

Text and tool events are provisional. Use `Completed.response` for validated
JSON, final tool calls, continuation signatures and reported usage. Execute tools
only after completion. Built-in streams yield exactly one completion or one
error, then end. An interrupted or token-limited response returns an error even
after progress was displayed. Dropping the stream releases Rath's local reads;
remote generation and billing may continue. Applications own retries and timeouts.
Existing custom backends inherit an explicit unsupported-capability error until
they implement `LlmBackend::execute_stream`.

The [streaming example](examples/llm_streaming.rs) offers a weather tool and
prints its progress and final calls. With no arguments it prints usage without
making a generation request; supplying a model URL and prompt requests one
generation using the usual provider credentials.

### OpenRouter structured output

OpenRouter's schema support depends on the provider endpoint serving a model.
Use its native `require_parameters` setting to restrict routing to endpoints
that support your requested parameters:

```rust
use rath::llm::LlmOptions;
use serde_json::json;

let options = LlmOptions::default()
    .with_output_schema(json!({
        "type": "object",
        "properties": { "answer": { "type": "string" } },
        "required": ["answer"],
        "additionalProperties": false
    }))
    .with_provider_config(json!({
        "provider": { "require_parameters": true }
    }));
```

Rath parses generated JSON strictly and does not repair malformed JSON, remove
markdown wrappers or extract JSON from prose. Invalid generated JSON returns
`Deserialize` with output-specific context and the parser cause. Tool argument
errors have separate context. The original HTTP response or received SSE prefix
remains available through `response_body()`, without appearing in error formatting.
Choice-level provider errors, refusals and `error`/`content_filter` finish reasons
return `Provider`; token-limit termination returns `OutputLimitReached` before
partial output is parsed. Partial content from a failed choice is never accepted
as a completed response.

Schema support and enforcement vary by endpoint, and parsing is not local schema
validation. See [OpenRouter structured outputs](https://openrouter.ai/docs/guides/features/structured-outputs).
OpenRouter also offers an optional [response-healing plugin](https://openrouter.ai/docs/guides/features/plugins/response-healing),
enabled explicitly through `provider_config.plugins`. It applies to non-streaming
JSON requests only; Rath does not enable it automatically.

## Ollama thinking

Use either `.with_thinking(Some(ThinkingLevel::Off))` or a model URL such as
`ollama:///qwen3:8b?thinking=off`. An explicit URL `thinking` value overrides the
programmatic option; without a URL value, the programmatic setting applies.
Leaving both unspecified omits the control and preserves Ollama's default.

On `/v1/chat/completions`, Rath maps `Off`, `Low`, `Medium`, `High`, and `XHigh` to
`reasoning_effort` values `none`, `low`, `medium`, `high`, and `max`, respectively.
Rath does not inject `/no_think` or send the native API's `think` field. Support
depends on the model and Ollama version; provider rejections remain typed errors
without retries or downgraded settings. Some models cannot disable thinking.

Explicit Off supports the existing JSON-output mode. Unspecified thinking also
retains that mode; JSON formatting itself does not disable reasoning. Explicit
enabled levels retain schema instructions without forced JSON mode.

Ollama responses expose `finish_reason`, `reasoning`, and raw `usage` alongside
`id` in `LlmResponse::raw_metadata`. Missing fields are null. An absent reasoning
trace or missing reasoning-token count does not establish zero reasoning tokens.
These fields may contain private model output and are caller-controlled.

Opt-in local tests use Rath's public client with a synthetic extraction task and
explicit installed models (`qwen3:0.6b`, `granite4.2:3b`, `lfm2.5-thinking:1.2b`):

```sh
cargo test --test ollama_thinking_live -- --ignored --nocapture --test-threads=1
```

The tests target `127.0.0.1:11434`, never download models, and report output,
finish status, available reasoning evidence, token usage and elapsed time for
three requests per model. Normal test runs skip them. Server support follows
[Ollama's compatibility API](https://docs.ollama.com/api/openai-compatibility).

## Ollama tool calls and JSON

Use `with_tools(...)` to offer tools. `ToolChoice::Auto` permits a normal answer;
`Required` needs at least one offered tool and rejects responses without a call.
`Disabled` omits offered tools, preserves tool-like text as ordinary text, and
rejects protocol-level calls. Rath returns calls for the application to execute.
Returned calls must name an offered tool and have distinct nonempty IDs and
JSON-object arguments. Complete legacy `<function=...>` blocks remain supported
when tools are enabled; malformed blocks return errors.

Use `Message::from_json(Role::User, &value)` for JSON input and
`with_output_schema(...)` or `with_response_format(ResponseFormat::Json)` for
JSON output. `with_input_schema(...)` adds model instructions; it does not parse
or validate manually supplied message text. Schema options do not perform local
JSON Schema validation. Validate application-specific shapes after decoding.
Valid JSON values, including literal markup inside strings or keys, are preserved.

Failures remain inspectable through `RathError`:

- Provider-declared unsupported tools/JSON mode: `Http`, with status, provider
  message and available code/type. Rath never removes the setting and retries.
- Missing required calls, malformed call shapes or disallowed calls:
  `InvalidResponse`. A missing call does not prove the model lacks tool support.
- Invalid JSON output/arguments: `Deserialize`, with a useful explanation and
  the original decoder cause, including line/column details when available.
- Truncated output: `OutputLimitReached`, before parsing or tool extraction.

Response bodies are explicitly accessible and absent from ordinary error
formatting. `Message::from_json` returns `RathError` with a Serialize classification and the original cause.

Run installed-model checks with:

```sh
cargo test --test ollama_tools_json_live -- --ignored --nocapture --test-threads=1
```

The cases exercise JSON input, tool-call/result replay and JSON output on local
Granite/Qwen models. Granite's case explicitly disables tools for the final JSON
turn after the lookup; Qwen3 8B retains Auto. Mixed tool/JSON behavior varies by
model and server. Invalid mixed output remains an error, never a partial success.

## LLM Provider Config

Use `provider_config` for provider-specific request knobs that Rath does not
model directly. Common options such as temperature, thinking, tools, schemas,
system prompt, and cache should still use Rath's typed fields.

For example, Gemini safety settings can be provided with conservative
`safetySettings`:

```rust
use rath::llm::LlmOptions;
use serde_json::json;

async fn run() -> Result<(), rath::RathError> {
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
        .create("gemini:///gemini-2.5-flash")
        .await?;
    Ok(())
}
```

## Structured Output Strategy

Rath uses native structured output where providers support it cleanly. For
Ollama, and Gemini models before `gemini-3.1`, Rath may convert structured
output into a required synthetic tool call. This is internal to Rath: callers
still receive `LlmOutput::Output(...)` when the exit tool is called successfully.

The strategy is private and operation-local; `options()` remains unchanged. The reserved
`__rath_final_output` name cannot be used by application tools. Mixed final-output and
ordinary tool calls are errors. Disabled user-tool policy is preserved.

`ResponseFormat::{Text, Json, JsonSchema { schema }}` is the sole output choice.
`with_output_schema` selects JsonSchema; setting another format replaces the entire
choice. JSON parsing does not prove schema conformance. Input schemas are prompt hints.
Applications supply reminders as ordinary messages.

## License

Licensed under either the MIT License or Apache License 2.0, at your option.

## Request measurement and response limits

All five LLM adapters (Gemini, OpenAI, Anthropic, OpenRouter and Ollama) support
local text-request estimates and `LlmOptions::with_max_output_tokens`.

```rust,no_run
use rath::llm::{LlmOptions, Message, RathError};

async fn example(model_url: &str, retained_messages: &[Message]) -> Result<(), RathError> {
    let client = LlmOptions::default()
        .with_preamble("Answer using the supplied conversation.")
        .with_max_output_tokens(1024)
        .create(model_url)
        .await?;

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
    Ok(())
}
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

Measurement methods still default to unsupported for custom `LlmBackend` implementations.
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

Since 0.2.8, `RathError` stores its diagnostic details in an internal `Box`, making
the error value pointer-sized. Keep using `Result<T, RathError>`; callers do not
need to box it to reduce its size. Constructors, accessors, cause traversal and
formatting are unchanged. `LlmResponse` remains returned by value. Each error
level owns one allocation for its details; cloning still copies its snapshot.

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
        error.provider(),
        error.operation(),
        error.http_status(),
        error.provider_code(),
        error.request_id(),
        error.retry_after(),
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

## API stabilization migration

Version 0.3.0 introduces these breaking API changes.

- Implement `…Backend` instead of the old `…Client` traits; use native async methods.
- Replace `Box<dyn …Client>` with concrete client handles and await construction.
- Replace MediaProvider and LlmClientFactory/Layer with ProviderFactory. Decorators
  explicitly forward untouched capabilities; unsupported defaults do not delegate.
- Construct extensible records with their constructors/builders, then assign public
  fields when needed. Add fallback arms only for extensible enums.
- Use ModelUrl accessors. Model overrides on image/TTS/STT requests are removed.
- Replace output_schema/output_type_name with ResponseFormat::JsonSchema. Turn budgets,
  reminder helpers and exit-tool inspection are removed; applications own those policies.
- Read ToolCalls.text in place of thought. Embeddings now use ordered input/vector batches.
- Use typed FastEmbedOptions, with downloads explicitly enabled when intended.

Provider, ErrorKind, EmbedTaskType, ThinkingLevel, CacheControl, Role, Attachment,
image/video/voice data, VideoInput and VideoJobStatus are extensible. Options, requests,
responses, Message, ToolCall, ToolDefinition, TokenUsage, Voice, VoicePreview, VideoJob
and VideoEvent are extensible records. TokenCount and VoiceSample remain fixed records.
ToolChoice, ResponseFormat, LlmOutput, TokenCountSource and ErrorBody are closed enums.

Serde representations remain compatible except for documented field migrations:
embedding `input` is now an array; request-level `model` no longer controls routing;
new optional controls may be absent in older payloads. Non-exhaustiveness does not
make unknown persisted variants readable by older applications. Review storage migrations
separately from Rust API changes.
