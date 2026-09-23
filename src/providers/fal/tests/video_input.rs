use super::*;
use crate::images::ImageData;
use crate::video::VideoData;
use serde_json::json;

fn image() -> ImageData {
    ImageData::Url {
        url: "https://assets.example/start.png".into(),
    }
}
fn video() -> VideoData {
    VideoData::Url {
        url: "https://assets.example/motion.mp4".into(),
    }
}
fn i2v() -> VideoRequest {
    VideoRequest {
        prompt: "Walk".into(),
        input: VideoInput::ImageToVideo {
            start_image: image(),
            end_image: None,
        },
        ..Default::default()
    }
}
fn motion() -> VideoRequest {
    VideoRequest {
        input: VideoInput::MotionTransfer {
            character_image: image(),
            driving_video: video(),
        },
        provider_config: Some(json!({"character_orientation":"video"})),
        ..Default::default()
    }
}

/// Maps the verified T2V endpoints without changing caller-selected model parameters.
#[test]
fn text_endpoints_map_prompt_and_native_controls() {
    for model in [KLING_T2V, WAN_T2V] {
        let req = VideoRequest {
            prompt: "Walk".into(),
            provider_config: Some(json!({"duration":"3"})),
            ..Default::default()
        };
        let result = payload(model, &None, &req).unwrap();
        assert_eq!(result["prompt"], "Walk");
        assert_eq!(result["duration"], "3");
    }
}

/// Typed input wins over native aliases and client defaults without leaving competing image fields.
#[test]
fn maps_start_and_end_frames_with_precedence() {
    let mut req = i2v();
    req.input = VideoInput::ImageToVideo {
        start_image: image(),
        end_image: Some(image()),
    };
    req.provider_config = Some(json!({"duration":"3","prompt":"wrong","image_url":"wrong"}));
    let defaults =
        Some(json!({"duration":"5","start_image_url":"wrong","video_url":"wrong","cfg_scale":0.4}));
    let result = payload(KLING_I2V, &defaults, &req).unwrap();
    assert_eq!(
        result["start_image_url"],
        "https://assets.example/start.png"
    );
    assert_eq!(result["end_image_url"], "https://assets.example/start.png");
    assert_eq!(result["prompt"], "Walk");
    assert_eq!(result["duration"], "3");
    assert_eq!(result["cfg_scale"], 0.4);
    assert!(!result.contains_key("image_url") || result["image_url"].is_null());
    assert!(!result.contains_key("video_url"));
}

/// Removing an optional typed end frame also removes a conflicting configured end frame.
#[test]
fn absent_end_frame_clears_native_alias() {
    let result = payload(KLING_I2V, &Some(json!({"end_image_url":"wrong"})), &i2v()).unwrap();
    assert!(!result.contains_key("end_image_url"));
}

/// Motion transfer requires an explicit valid orientation while retaining other native controls.
#[test]
fn maps_motion_and_validates_orientation() {
    let mut req = motion();
    req.provider_config =
        Some(json!({"character_orientation":"image","keep_original_sound":false}));
    let result = payload(KLING_MOTION, &None, &req).unwrap();
    assert_eq!(result["image_url"], "https://assets.example/start.png");
    assert_eq!(result["video_url"], "https://assets.example/motion.mp4");
    assert_eq!(result["keep_original_sound"], false);
    assert!(!result.contains_key("prompt"));
    for config in [
        None,
        Some(json!({"character_orientation":"auto"})),
        Some(json!({"character_orientation":null})),
    ] {
        req.provider_config = config;
        assert_eq!(
            payload(KLING_MOTION, &None, &req).unwrap_err().kind(),
            ErrorKind::Validation
        );
    }
}

/// URL-routed typed operations never silently switch endpoints.
#[test]
fn incompatible_and_unknown_endpoints_fail() {
    for (model, req) in [
        (KLING_T2V, i2v()),
        (KLING_I2V, motion()),
        ("owner/custom", i2v()),
    ] {
        assert_eq!(
            payload(model, &None, &req).unwrap_err().kind(),
            ErrorKind::UnsupportedCapability
        );
    }
}

/// Non-object config, blank required prompts, and multi-shot conflicts fail locally.
#[test]
fn invalid_typed_requests_fail() {
    assert!(payload(KLING_I2V, &Some(json!([])), &i2v()).is_err());
    let mut req = i2v();
    req.provider_config = Some(json!("wrong"));
    assert!(payload(KLING_I2V, &None, &req).is_err());
    req.provider_config = None;
    req.prompt = "  ".into();
    assert!(payload(KLING_I2V, &None, &req).is_err());
    req.prompt = "Walk".into();
    req.provider_config = Some(json!({"multi_prompt":[]}));
    assert!(payload(KLING_I2V, &None, &req).is_err());
}

/// Explicit native mode preserves unknown model inputs with deterministic merge precedence.
#[test]
fn native_escape_hatch_preserves_payload() {
    let req = VideoRequest {
        input: VideoInput::Native {
            payload: json!({"prompt":"native","custom":[1,2],"duration":"7"}),
        },
        provider_config: Some(json!({"duration":"3","setting":true})),
        ..Default::default()
    };
    let output = payload(
        "owner/custom",
        &Some(json!({"duration":"5","other":false})),
        &req,
    )
    .unwrap();
    assert_eq!(
        Value::Object(output),
        json!({"prompt":"native","custom":[1,2],"duration":"7","setting":true,"other":false})
    );
}

/// Native requests cannot mix top-level prompt semantics or non-object payloads.
#[test]
fn native_input_is_explicit() {
    let mut req = VideoRequest {
        input: VideoInput::Native { payload: json!({}) },
        prompt: "typed".into(),
        ..Default::default()
    };
    assert!(payload("owner/custom", &None, &req).is_err());
    req.prompt.clear();
    req.input = VideoInput::Native { payload: json!([]) };
    assert!(payload("owner/custom", &None, &req).is_err());
}

/// Standard base64 maps to the appropriate image/video data URI without remote uploads.
#[test]
fn base64_media_is_encoded() {
    let mut req = motion();
    req.input = VideoInput::MotionTransfer {
        character_image: ImageData::Base64 {
            mime_type: "image/png".into(),
            data: "YWJj".into(),
        },
        driving_video: VideoData::Base64 {
            mime_type: "video/mp4".into(),
            data: "YWJj".into(),
        },
    };
    let result = payload(KLING_MOTION, &None, &req).unwrap();
    assert_eq!(result["image_url"], "data:image/png;base64,YWJj");
    assert_eq!(result["video_url"], "data:video/mp4;base64,YWJj");
}

/// Invalid media never reaches the paid endpoint and its content is not exposed in errors.
#[test]
fn malformed_media_fails() {
    for (mime, data) in [
        ("audio/wav", "YWJj"),
        ("image/", "YWJj"),
        ("image/png;bad", "YWJj"),
        ("image/png", ""),
        ("image/png", "!"),
    ] {
        assert!(
            video_media::image(&ImageData::Base64 {
                mime_type: mime.into(),
                data: data.into()
            })
            .is_err()
        );
    }
    for value in [
        "",
        "file:///tmp/image.png",
        "https://u:p@example.com/a",
        "https://example.com/a#b",
        " https://example.com/a",
    ] {
        assert!(video_media::image(&ImageData::Url { url: value.into() }).is_err());
    }
    assert!(
        video_media::video(&VideoData::Base64 {
            mime_type: "image/png".into(),
            data: "YWJj".into()
        })
        .is_err()
    );
}

/// Every explicit input variant serializes with the stable snake-case operation discriminator.
#[test]
fn input_serialization_roundtrips() {
    for (input, tag) in [
        (VideoInput::default(), "text_to_video"),
        (i2v().input, "image_to_video"),
        (motion().input, "motion_transfer"),
        (
            VideoInput::Native {
                payload: json!({"arbitrary":true}),
            },
            "native",
        ),
    ] {
        let raw = serde_json::to_value(&input).unwrap();
        assert_eq!(raw["type"], tag);
        let decoded: VideoInput = serde_json::from_value(raw.clone()).unwrap();
        assert_eq!(serde_json::to_value(decoded).unwrap(), raw);
    }
}
