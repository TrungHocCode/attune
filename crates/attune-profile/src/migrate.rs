//! Upgrade older profile documents before parsing (M4 slice 3).
//!
//! Slice 3 is the first profile writer, so `v0` is the shape an early
//! prototype or a hand-written file may have: no `profile_version`, no
//! `fallbacks` and no `expected_performance.confidence`. Every field is
//! already `serde(default)`, so parsing usually works even without this
//! helper; the migration step exists to make the upgraded document explicit
//! (stamp the version, materialise the fallback block) instead of relying on
//! implicit defaults.

use serde_json::{json, Map, Value};

use crate::{Profile, DEFAULT_FALLBACK_N_CTX, PROFILE_VERSION};

/// Upgrade `value` to [`PROFILE_VERSION`] and parse it.
///
/// The function cannot fail by contract: a non-object or unparseable document
/// yields [`Profile::default`]. Callers that must reject bad input should use
/// [`Profile::from_json_str`] directly.
pub fn migrate_value(mut value: Value) -> Profile {
    if let Some(object) = value.as_object_mut() {
        migrate_object(object);
    }
    serde_json::from_value(value).unwrap_or_default()
}

fn migrate_object(object: &mut Map<String, Value>) {
    let version = object
        .get("profile_version")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    if version >= u64::from(PROFILE_VERSION) {
        return;
    }

    object.insert("profile_version".to_string(), Value::from(PROFILE_VERSION));

    if object.get("fallbacks").is_some_and(Value::is_object) {
        if let Some(fallbacks) = object.get_mut("fallbacks").and_then(Value::as_object_mut) {
            fallbacks
                .entry("if_ram_insufficient")
                .or_insert_with(|| json!({ "n_ctx": DEFAULT_FALLBACK_N_CTX }));
        }
    } else {
        object.insert(
            "fallbacks".to_string(),
            json!({ "if_ram_insufficient": { "n_ctx": DEFAULT_FALLBACK_N_CTX } }),
        );
    }

    match object
        .get_mut("expected_performance")
        .and_then(Value::as_object_mut)
    {
        Some(performance) => {
            performance
                .entry("confidence")
                .or_insert_with(|| json!(0.0));
        }
        None => {
            object.insert(
                "expected_performance".to_string(),
                json!({ "confidence": 0.0 }),
            );
        }
    }

    // `draft_model` is `Option` and already defaults to `None`; no key is
    // inserted so an upgraded v0 document keeps omitting it from JSON.
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pre-slice-3 shape: version 0, no fallbacks, no confidence, no draft.
    const V0_JSON: &str = r#"{
        "profile_version": 0,
        "name": "legacy",
        "created_at": 1700000000,
        "hardware_fingerprint": "cpu|4|sse|8000000000",
        "engine": { "name": "llama.cpp", "version": "b4000" },
        "model": { "path": "models/legacy.gguf", "quant": "q4_k_m", "size_bytes": 82600000 },
        "load_config": {
            "n_ctx": 2048,
            "n_batch": 512,
            "n_ubatch": 128,
            "n_gpu_layers": 0,
            "mmap": true,
            "mlock": false,
            "flash_attn": true,
            "cache_type_k": "q8_0",
            "cache_type_v": "q8_0",
            "numa": "local"
        },
        "compute_config": { "n_threads": 4, "n_threads_batch": 4 },
        "expected_performance": { "ttft_ms": 10.0, "tpot_ms": 4.0, "tok_s": 250.0, "ram_peak_gb": 0.1 }
    }"#;

    #[test]
    fn migrates_the_v0_shape_and_fills_the_gaps() {
        let value: Value = serde_json::from_str(V0_JSON).expect("parse v0 fixture");
        let profile = migrate_value(value);

        assert_eq!(profile.profile_version, PROFILE_VERSION);
        assert_eq!(profile.name, "legacy");
        assert_eq!(profile.load_config.n_ctx, 2_048);
        assert_eq!(
            profile
                .fallbacks
                .if_ram_insufficient
                .as_ref()
                .map(|fallback| fallback.n_ctx),
            Some(DEFAULT_FALLBACK_N_CTX)
        );
        assert_eq!(profile.expected_performance.confidence, 0.0);
        assert!(profile.draft_model.is_none());
    }

    #[test]
    fn fills_a_missing_ram_fallback_in_an_existing_block() {
        let value = json!({
            "profile_version": 0,
            "name": "partial-fallbacks",
            "fallbacks": {}
        });
        let profile = migrate_value(value);
        assert_eq!(
            profile
                .fallbacks
                .if_ram_insufficient
                .as_ref()
                .map(|fallback| fallback.n_ctx),
            Some(DEFAULT_FALLBACK_N_CTX)
        );
    }

    #[test]
    fn keeps_a_v1_document_unchanged() {
        let profile = Profile {
            name: "already-current".to_string(),
            ..Profile::default()
        };
        let value = serde_json::to_value(&profile).expect("serialize v1 profile");

        let migrated = migrate_value(value);
        assert_eq!(migrated.profile_version, PROFILE_VERSION);
        assert_eq!(migrated.name, "already-current");
    }

    #[test]
    fn non_object_input_falls_back_to_default() {
        let migrated = migrate_value(json!("not a profile"));
        assert_eq!(migrated, Profile::default());
    }
}
