# Eval corpus attribution

`articles.jsonl` is a frozen snapshot of public blog posts, taken by
`python -m eval.snapshot` at the time recorded in `manifest.json`. Each record
keeps its original URL. The text was extracted from each post's page with
trafilatura and is otherwise unmodified. It is redistributed here, under each
source's license, only so the retrieval eval runs against a fixed corpus.

| Source | Feed | License |
|---|---|---|
| Kubernetes Blog, © The Kubernetes Authors | https://kubernetes.io/feed.xml | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| The Go Blog, © The Go Authors | https://go.dev/blog/feed.atom | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (code samples: BSD-3-Clause) |
| Rust Blog and Inside Rust, © The Rust Project Developers | https://blog.rust-lang.org/feed.xml, https://blog.rust-lang.org/inside-rust/feed.xml | [Apache-2.0](https://github.com/rust-lang/blog.rust-lang.org/blob/main/LICENSE-APACHE) / MIT |

Changes from the originals: page chrome removed by text extraction; nothing
else. The bot's own feeds (`config/feeds.yaml`) are not part of this corpus.
