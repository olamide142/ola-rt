//! OpenAI-compatible `POST /v1/completions`.
//!
//! Only what the runtime can honour is accepted. Generation is greedy, so a
//! non-zero `temperature` or `top_p` below 1 is refused rather than silently
//! ignored, as are several choices (`n`, `best_of`), `logprobs`, `echo`,
//! `suffix`, `stop`, and `logit_bias`. Errors use OpenAI's error shape.

use std::{
    collections::BTreeMap,
    convert::Infallible,
    time::{SystemTime, UNIX_EPOCH},
};

use axum::{
    Json,
    extract::{State, rejection::JsonRejection},
    http::{HeaderValue, header},
    response::{
        IntoResponse, Response,
        sse::{Event, KeepAlive, Sse},
    },
};
use futures_util::{Stream, StreamExt, stream};
use ola_core::{GenerationConfig, Input, Priority, RequestSpec};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::ApiError;
use crate::{FinishReason, GenerationOutput, RequestEvent, RequestHandle, RuntimeHandle};

/// A completions request. Fields OpenAI defines but the runtime cannot
/// honour are parsed so they can be refused with a precise message.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct CompletionRequest {
    model: String,
    prompt: Prompt,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    temperature: Option<f64>,
    #[serde(default)]
    top_p: Option<f64>,
    #[serde(default)]
    n: Option<u32>,
    #[serde(default)]
    best_of: Option<u32>,
    #[serde(default)]
    logprobs: Option<Value>,
    #[serde(default)]
    echo: Option<bool>,
    #[serde(default)]
    suffix: Option<String>,
    #[serde(default)]
    stop: Option<Value>,
    #[serde(default)]
    logit_bias: Option<BTreeMap<String, f64>>,
    /// Accepted and ignored: greedy decoding is already deterministic.
    #[serde(default)]
    seed: Option<i64>,
    #[serde(default)]
    presence_penalty: Option<f64>,
    #[serde(default)]
    frequency_penalty: Option<f64>,
    /// Recorded as the request attribute `user`.
    #[serde(default)]
    user: Option<String>,
}

/// OpenAI's default for `max_tokens`.
const fn default_max_tokens() -> usize {
    16
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum Prompt {
    Text(String),
    Tokens(Vec<u32>),
    /// Any other array, such as several prompts.
    Batch(Vec<serde::de::IgnoredAny>),
}

impl CompletionRequest {
    fn into_spec(self) -> Result<(RequestSpec, String, bool), ApiError> {
        let refuse = |message: &str| Err(ApiError::Transport(message.to_owned()));
        if self
            .temperature
            .is_some_and(|temperature| temperature != 0.0)
        {
            return refuse("only greedy decoding is supported; omit `temperature` or set it to 0");
        }
        if self.top_p.is_some_and(|top_p| top_p != 1.0) {
            return refuse("only greedy decoding is supported; omit `top_p` or set it to 1");
        }
        if self.n.is_some_and(|n| n != 1) || self.best_of.is_some_and(|best| best != 1) {
            return refuse(
                "only one choice per request is supported (`n` and `best_of` must be 1)",
            );
        }
        if self.logprobs.as_ref().is_some_and(|value| !value.is_null()) {
            return refuse("`logprobs` is not supported");
        }
        if self.echo == Some(true) {
            return refuse("`echo` is not supported");
        }
        if self.suffix.is_some() {
            return refuse("`suffix` is not supported");
        }
        if self.stop.as_ref().is_some_and(|value| !value.is_null()) {
            return refuse("`stop` sequences are not supported");
        }
        if self
            .logit_bias
            .as_ref()
            .is_some_and(|bias| !bias.is_empty())
        {
            return refuse("`logit_bias` is not supported");
        }
        if self.presence_penalty.is_some_and(|penalty| penalty != 0.0)
            || self.frequency_penalty.is_some_and(|penalty| penalty != 0.0)
        {
            return refuse("presence and frequency penalties are not supported");
        }
        let _ = self.seed;
        let input = match self.prompt {
            Prompt::Text(text) => Input::Text(text),
            Prompt::Tokens(tokens) => Input::Tokens(tokens),
            Prompt::Batch(prompts) => {
                return Err(ApiError::Transport(format!(
                    "a batch of {} prompts is not supported; send one prompt per request",
                    prompts.len()
                )));
            }
        };
        let mut attributes = BTreeMap::new();
        if let Some(user) = self.user {
            attributes.insert("user".to_owned(), user);
        }
        let spec = RequestSpec {
            model: self.model.as_str().into(),
            inputs: vec![input],
            generation: GenerationConfig {
                max_new_tokens: self.max_tokens,
                temperature: 0.0,
                top_p: 1.0,
                seed: None,
            },
            priority: Priority::NORMAL,
            deadline: None,
            attributes,
        };
        Ok((spec, self.model, self.stream))
    }
}

#[derive(Debug, Serialize)]
struct Completion {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: [Choice; 1],
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<Usage>,
}

#[derive(Debug, Serialize)]
struct Choice {
    text: String,
    index: u32,
    logprobs: Option<()>,
    finish_reason: Option<&'static str>,
}

#[derive(Debug, Serialize)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

/// Fields shared by every chunk and the final object of one completion.
#[derive(Clone)]
struct Identity {
    id: String,
    created: u64,
    model: String,
}

impl Identity {
    fn completion(
        &self,
        text: String,
        finish: Option<FinishReason>,
        usage: Option<Usage>,
    ) -> Completion {
        Completion {
            id: self.id.clone(),
            object: "text_completion",
            created: self.created,
            model: self.model.clone(),
            choices: [Choice {
                text,
                index: 0,
                logprobs: None,
                finish_reason: finish.map(|reason| match reason {
                    FinishReason::EndOfSequence => "stop",
                    FinishReason::Length => "length",
                }),
            }],
            usage,
        }
    }
}

fn usage(output: &GenerationOutput) -> Usage {
    Usage {
        prompt_tokens: output.metrics.prompt_tokens,
        completion_tokens: output.metrics.generated_tokens,
        total_tokens: output.metrics.prompt_tokens + output.metrics.generated_tokens,
    }
}

/// An error in OpenAI's shape: `{"error": {"message", "type", "param",
/// "code"}}`, with `type` set to the native error kind.
#[derive(Debug)]
pub(super) struct OpenAiError(ApiError);

impl From<ApiError> for OpenAiError {
    fn from(error: ApiError) -> Self {
        Self(error)
    }
}

impl From<crate::ServeError> for OpenAiError {
    fn from(error: crate::ServeError) -> Self {
        Self(ApiError::Serve(error))
    }
}

impl From<JsonRejection> for OpenAiError {
    fn from(rejection: JsonRejection) -> Self {
        Self(rejection.into())
    }
}

#[derive(Serialize)]
struct ErrorBody {
    error: ErrorJson,
}

#[derive(Serialize)]
struct ErrorJson {
    message: String,
    #[serde(rename = "type")]
    kind: &'static str,
    param: Option<()>,
    code: Option<()>,
}

impl OpenAiError {
    fn body(&self) -> ErrorBody {
        let (_, kind, message) = self.0.parts();
        ErrorBody {
            error: ErrorJson {
                message,
                kind,
                param: None,
                code: None,
            },
        }
    }
}

impl IntoResponse for OpenAiError {
    fn into_response(self) -> Response {
        let (status, ..) = self.0.parts();
        let mut response = (status, Json(self.body())).into_response();
        if super::retryable(status) {
            response
                .headers_mut()
                .insert(header::RETRY_AFTER, HeaderValue::from_static("1"));
        }
        response
    }
}

pub(super) async fn completions(
    State(runtime): State<RuntimeHandle>,
    body: Result<Json<CompletionRequest>, JsonRejection>,
) -> Result<Response, OpenAiError> {
    let Json(body) = body?;
    let (spec, model, streaming) = body.into_spec()?;
    let handle = runtime.submit(spec).await?;
    let identity = Identity {
        id: format!("cmpl-{}", handle.id()),
        created: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |elapsed| elapsed.as_secs()),
        model,
    };
    if streaming {
        return Ok(Sse::new(chunks(handle, identity))
            .keep_alive(KeepAlive::default())
            .into_response());
    }
    let output = handle.result().await?;
    let usage = usage(&output);
    Ok(
        Json(identity.completion(output.text, Some(output.finish_reason), Some(usage)))
            .into_response(),
    )
}

/// Streams OpenAI chunks: one per non-empty text delta, a final chunk with
/// `finish_reason` and `usage`, then `[DONE]`. A failure sends one error
/// object, then `[DONE]`. Dropping the stream cancels the request.
fn chunks(
    handle: RequestHandle,
    identity: Identity,
) -> impl Stream<Item = Result<Event, Infallible>> {
    stream::unfold(Some(handle), move |handle| {
        let identity = identity.clone();
        async move {
            let mut handle = handle?;
            loop {
                let event = handle.next().await?;
                let data = match event {
                    RequestEvent::Token { text, .. } if text.is_empty() => continue,
                    RequestEvent::Token { text, .. } => {
                        let chunk = identity.completion(text, None, None);
                        return Some((vec![json(&chunk)], Some(handle)));
                    }
                    RequestEvent::Finished(output) => {
                        let usage = usage(&output);
                        json(&identity.completion(
                            String::new(),
                            Some(output.finish_reason),
                            Some(usage),
                        ))
                    }
                    RequestEvent::Failed(error) => {
                        json(&OpenAiError(ApiError::Serve(error)).body())
                    }
                    RequestEvent::Cancelled => {
                        json(&OpenAiError(ApiError::Serve(crate::ServeError::Cancelled)).body())
                    }
                };
                return Some((vec![data, Event::default().data("[DONE]")], None));
            }
        }
    })
    .flat_map(|events| stream::iter(events.into_iter().map(Ok)))
}

fn json(data: &impl Serialize) -> Event {
    Event::default()
        .json_data(data)
        .expect("response types always serialize")
}
