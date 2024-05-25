# `.gguf` chat template exploit analysis

### Overview

This analysis examines the vulnerability in GGUF models related to the [recently disclosed CVE](https://github.com/abetlen/llama-cpp-python/security/advisories/GHSA-56xg-wfcc-g829) affecting jinja2's template rendering mechanism in llama-cpp-python. The goal is to identify any dangerous implementations of this template in publicly available GGUF models.

### Background

Each GGUF model can include a chat template, which utilizes jinja2 templating to format the prompt. This template resides in the file's header metadata. A potential security risk arises when this template is not rendered in a sandboxed environment, leading to possible arbitrary code execution.

### Analysis Methodology

The analysis was conducted using a *blazingly fast* Rust script that retrieves and processes a large number of GGUF files. Specifically, the script emits an HTTP request with a RANGE header to fetch only the relevant bytes of the GGUF file, containing the header & the chat template.

Two evaluation methods were employed:

- Dynamic Analysis: Executing the chat template in a jinja2.sandbox.SandboxedEnvironment to observe its behavior.
- Static Analysis: Scanning the chat templates for suspicious strings collected from various sources on the web.

### Results

Out of over 116,000 GGUF models analyzed, only one dangerous model was identified: [@retr0reg](https://x.com/retr0reg)'s exploit. Approximately 70 models were flagged as suspicious during the static analysis, but further inspection revealed no additional threats. It is worth noting that about 40% of the models (approximately 46,000) included a chat template, highlighting the potential impact of this vulnerability.

### Recommendations

- Update llama-cpp-python to the latest version to address the disclosed CVE.
- Exercise caution when loading weights or models from unknown or untrusted sources.
