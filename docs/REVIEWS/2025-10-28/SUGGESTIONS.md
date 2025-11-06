# Suggestions

| Suggestion | Impact | Effort | Priority |
| --- | --- | --- | --- |
| Update implementation tasks to output `tracks.json`/`clusters.json` artifacts so runtime matches the refreshed docs. | High | Medium | P0 |
| Add a lightweight validation script in CI to diff `AGENTS/agents.yml` against generated docs snippets to avoid future drift. | Medium | Medium | P1 |
| Monitor lychee link-check runtime; if it slows CI, consider caching results or scoping to changed files. | Low | Low | P2 |
