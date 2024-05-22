use std::sync::Arc;

use anyhow::anyhow;
use bytes::{BufMut, BytesMut};
use futures::stream::FuturesUnordered;
use gguf::{GGUFHeader, GGUFMetadata, GGUFMetadataValue};
use indicatif::ProgressBar;
use pyo3::{
    types::{PyAnyMethods, PyTypeMethods},
    PyResult, Python,
};
use reqwest::{
    header::{LINK, RANGE},
    IntoUrl, StatusCode,
};
use serde::Deserialize;
use tokio::{sync::Semaphore, time::Instant};
use tracing::{debug, info, info_span, Instrument};

const DANGER_LIST: [&str; 6] = [
    "__",
    "\\x5f\\x5f",
    "attr",
    "__class__",
    "__base__",
    "__subclasses__",
];
// Default to 10 Mib
const HEADER_CHUNK_LENGTH: usize = 10_485_760;
const MAX_CONCURRENT_CHECKS: usize = 128;
const WARNING_THRESHOLD: usize = 104_857_600;

#[derive(Deserialize)]
struct Sibling {
    rfilename: String,
}

#[derive(Deserialize)]
struct SiblingsList {
    _id: String,
    id: String,
    siblings: Vec<Sibling>,
}

enum StaticCheckStatus {
    Innocuous,
    Suspicious,
}

fn static_check(chat_template: &str) -> StaticCheckStatus {
    if DANGER_LIST.iter().any(|expr| chat_template.contains(expr)) {
        info!("potentially malicious chat template:\n{chat_template}");
        StaticCheckStatus::Suspicious
    } else {
        StaticCheckStatus::Innocuous
    }
}

fn get_key_value(key: &str, metadata: &[GGUFMetadata]) -> Option<(String, GGUFMetadataValue)> {
    for kv in metadata {
        if kv.key == key {
            return Some((kv.key.clone(), kv.value.clone()));
        }
    }

    None
}

struct SecurityError(bool);

async fn run_jinja_template(chat_template: String) -> anyhow::Result<SecurityError> {
    tokio::task::spawn_blocking(|| {
        let py_res: PyResult<bool> = Python::with_gil(|py| {
            let jinja2_sandbox = py.import_bound("jinja2.sandbox")?;
            let sandboxed_env = jinja2_sandbox.getattr("SandboxedEnvironment")?.call0()?;
            let template = sandboxed_env.call_method1("from_string", (chat_template,))?;
            let rendered = template.call_method1("render", ());
            match rendered {
                Ok(_) => (),
                Err(err) => {
                    let r#type = err.get_type_bound(py);
                    if r#type.name()? == "SecurityError" {
                        return Ok(true);
                    }
                }
            }

            Ok(false)
        });

        Ok(SecurityError(py_res?))
    })
    .await?
}

async fn build_repo_list(
    client: &reqwest::Client,
    url: impl IntoUrl,
) -> anyhow::Result<Vec<SiblingsList>> {
    let mut response = client.get(url).send().await?.error_for_status()?;
    let mut siblings_list: Vec<SiblingsList> = vec![];
    loop {
        let next_link = response
            .headers()
            .get(LINK)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| parse_link_header::parse(v).ok())
            .and_then(|mut links| links.remove(&Some("next".to_owned())))
            .map(|link| link.uri.clone());

        siblings_list.extend(response.json::<Vec<SiblingsList>>().await?);
        if let Some(link) = next_link {
            response = client.get(link).send().await?.error_for_status()?;
        } else {
            break;
        }
    }

    Ok(siblings_list)
}

async fn fetch_file_header(
    client: &reqwest::Client,
    url: impl IntoUrl,
) -> anyhow::Result<Option<GGUFHeader>> {
    let mut bytes = BytesMut::with_capacity(HEADER_CHUNK_LENGTH);
    let range = format!("bytes=0-{HEADER_CHUNK_LENGTH}");
    let url = url.into_url()?;
    let response = client
        .get(url.clone())
        .header(RANGE, range)
        .send()
        .await?
        .error_for_status()?;
    bytes.put(response.bytes().await?);
    let mut header = match GGUFHeader::read(&bytes).map_err(|e| anyhow!(e)) {
        Ok(header) => header,
        Err(err) => {
            debug!("failed to parse header: {err}");
            return Ok(None);
        }
    };
    let mut start = HEADER_CHUNK_LENGTH + 1;
    let mut stop = start + HEADER_CHUNK_LENGTH;
    while header.is_none() {
        let range = format!("bytes={start}-{stop}");
        let response = match client
            .get(url.clone())
            .header(RANGE, range)
            .send()
            .await?
            .error_for_status()
        {
            Ok(res) => res,
            Err(err) => {
                if err.status() == Some(StatusCode::RANGE_NOT_SATISFIABLE) {
                    debug!("failed to parse header after downloading full file");
                    return Ok(None);
                } else {
                    return Err(anyhow::Error::from(err));
                }
            }
        };
        bytes.put(response.bytes().await?);
        header = match GGUFHeader::read(&bytes).map_err(|e| anyhow!(e)) {
            Ok(header) => header,
            Err(err) => {
                debug!("failed to parse header: {err}");
                return Ok(None);
            }
        };
        if stop > WARNING_THRESHOLD {
            debug!("downloaded over {WARNING_THRESHOLD} bytes for file, skipping");
            return Ok(None);
        }
        start += stop + 1;
        stop += start + HEADER_CHUNK_LENGTH;
    }

    Ok(header)
}

struct Stats {
    no_chat_template: bool,
    parse_header_failure: bool,
    security_error: bool,
    static_check_status: StaticCheckStatus,
}

impl Stats {
    fn no_chat_template() -> Self {
        Self {
            no_chat_template: true,
            parse_header_failure: false,
            security_error: false,
            static_check_status: StaticCheckStatus::Innocuous,
        }
    }

    fn parse_header_failure() -> Self {
        Self {
            no_chat_template: false,
            parse_header_failure: true,
            security_error: false,
            static_check_status: StaticCheckStatus::Innocuous,
        }
    }

    fn verify_result(security_error: bool, static_check_status: StaticCheckStatus) -> Self {
        Self {
            no_chat_template: false,
            parse_header_failure: false,
            security_error,
            static_check_status,
        }
    }
}

async fn verify_file(
    client: reqwest::Client,
    file: Sibling,
    repo_id: String,
    semaphore: Arc<Semaphore>,
    stats_collector_tx: tokio::sync::mpsc::Sender<Stats>,
) -> anyhow::Result<()> {
    // let full_instant = Instant::now();
    let permit = semaphore.acquire_owned().await?;
    let url = format!(
        "https://huggingface.co/{}/resolve/main/{}",
        repo_id, file.rfilename
    );
    // let fetch_header_inst = Instant::now();
    let header = match fetch_file_header(&client, url).await? {
        Some(header) => header,
        None => {
            // info!(
            //     "fetch_header took: {}s",
            //     fetch_header_inst.elapsed().as_secs()
            // );
            stats_collector_tx
                .send(Stats::parse_header_failure())
                .await?;
            return Ok(());
        }
    };
    // info!(
    //     "fetch_header took: {}s",
    //     fetch_header_inst.elapsed().as_secs()
    // );
    let value = match get_key_value("tokenizer.chat_template", &header.metadata) {
        Some((_, v)) => v,
        None => {
            stats_collector_tx.send(Stats::no_chat_template()).await?;
            return Ok(());
        }
    };
    if let GGUFMetadataValue::String(value) = value {
        // let static_check_inst = Instant::now();
        let static_check_status = static_check(&value);
        // info!(
        //     "static_check took: {}s",
        //     static_check_inst.elapsed().as_secs()
        // );
        // let run_jinja_inst = Instant::now();
        let security_error = run_jinja_template(value).await?.0;
        // info!(
        //     "run_jinja_template took: {}s",
        //     run_jinja_inst.elapsed().as_secs()
        // );
        if security_error {
            info!("Security Error was caught when running chat template");
        }
        stats_collector_tx
            .send(Stats::verify_result(security_error, static_check_status))
            .await?;
        drop(permit);
        // info!("verify_file took: {}s", full_instant.elapsed().as_secs());
        Ok(())
    } else {
        drop(permit);
        // info!("verify_file took: {}s", full_instant.elapsed().as_secs());
        Err(anyhow!(
            "invalid 'tokenizer.chat_template' value, got: {:?}",
            value
        ))
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let builder = tracing_subscriber::fmt()
        .with_target(true)
        .with_line_number(true);

    builder
        // .json()
        // .flatten_event(true)
        // .with_current_span(false)
        // .with_span_list(true)
        .init();

    let client = reqwest::Client::new();
    let repo_list_url = "https://huggingface.co/api/models?filter=gguf&expand[]=siblings";

    let instant = Instant::now();
    let repos_list = build_repo_list(&client, repo_list_url).await?;
    info!("build repo list in {}s", instant.elapsed().as_secs());

    info!("repos_list len: {}", repos_list.len());

    let (stats_collector_tx, mut stats_collector_rx) = tokio::sync::mpsc::channel(512);
    let mut total_gguf_files = 0;
    let semaphore = Arc::new(Semaphore::new(MAX_CONCURRENT_CHECKS));
    let handles = FuturesUnordered::new();
    for repo in repos_list {
        for file in repo.siblings {
            if file.rfilename.ends_with(".gguf") {
                total_gguf_files += 1;

                let span = info_span!(
                    "file verification",
                    repo_id = repo.id,
                    revision = "main",
                    filename = file.rfilename
                );
                let client = client.clone();
                let repo_id = repo.id.clone();
                let semaphore = semaphore.clone();
                let stats_collector_tx = stats_collector_tx.clone();
                handles.push(tokio::spawn(
                    verify_file(client, file, repo_id, semaphore, stats_collector_tx)
                        .instrument(span),
                ));
            }
        }
    }

    drop(stats_collector_tx);
    let mut no_chat_template = 0;
    let mut parse_header_failures = 0;
    let mut sandbox_run_suspicious_files = 0;
    let mut static_scan_suspicious_files = 0;
    let bar = ProgressBar::new(total_gguf_files);
    while let Some(stats) = stats_collector_rx.recv().await {
        if stats.no_chat_template {
            no_chat_template += 1;
        }
        if stats.parse_header_failure {
            parse_header_failures += 1;
        }
        if stats.security_error {
            sandbox_run_suspicious_files += 1;
        }
        if matches!(stats.static_check_status, StaticCheckStatus::Suspicious) {
            static_scan_suspicious_files += 1;
        }
        bar.inc(1);
    }

    futures::future::join_all(handles)
        .await
        .into_iter()
        .flatten()
        .collect::<anyhow::Result<()>>()?;

    info!("{no_chat_template} ouf of {total_gguf_files} gguf files were missing a chat_template");
    info!(
        "failed to parse headers for {parse_header_failures} ouf of {total_gguf_files} gguf files"
    );
    info!("{sandbox_run_suspicious_files} ouf of {total_gguf_files} gguf files triggered a SecurityError in jinja2 sandbox environment");
    info!("{static_scan_suspicious_files} ouf of {total_gguf_files} gguf files were flagged as suspicious by static scan");

    info!("total # of processed files: {total_gguf_files}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_static_malicious() {
        let chat_template = r#"{% for x in ().__class__.__base__.__subclasses__() %}{% if "warning" in x.__name__ %}{{x()._module.__builtins__['__import__']('os').popen("touch /tmp/retr0reg")}}{%endif%}{% endfor %}"#;
        assert!(matches!(
            static_check(chat_template),
            StaticCheckStatus::Suspicious
        ));
    }

    #[tokio::test]
    async fn test_sandbox_malicious() {
        let chat_template = r#"{% for x in ().__class__.__base__.__subclasses__() %}{% if "warning" in x.__name__ %}{{x()._module.__builtins__['__import__']('os').popen("touch /tmp/retr0reg")}}{%endif%}{% endfor %}"#;
        assert!(matches!(
            run_jinja_template(chat_template.to_owned()).await,
            Ok(SecurityError(true))
        ));
    }

    // XXX: this test does not assert anything because [`verify_file`]'s return type is `()`
    // #[tokio::test]
    // async fn test_known_repo() -> anyhow::Result<()> {
    //     let client = reqwest::Client::new();
    //     let file = Sibling {
    //         rfilename: "retr0reg.gguf".to_owned(),
    //     };
    //     let repo_id = String::from("Retr0REG/Whats-up-gguf");
    //     let semaphore = Arc::new(Semaphore::new(1));
    //     assert!(matches!(
    //         verify_file(client, file, repo_id, semaphore).await?,
    //         ()
    //     ));
    //
    //     Ok(())
    // }
}
