# Environment report

Детерминированный отчёт: без timestamps и git-состояния, чтобы повторный gate
не создавал diff. Время запуска и git-состояние — в
`artifacts/runs/<run_id>/environment_stamp.json`.

| key | value |
|---|---|
| `schema_version` | `2` |
| `os` | `Darwin 25.6.0` |
| `arch` | `arm64` |
| `python_version` | `3.13.2` |
| `uv_version` | `0.12.7` |
| `uv_lock_sha256` | `d757313d9e7e308b2009300cd4f5ff438003b2e07926035e0433c03f0a6299a1` |
| `julia_pinned_version` | `1.12.7` |
| `julia_executable_version` | `1.12.7` |
| `julia_manifest_version` | `1.12.7` |
| `julia_manifest_sha256` | `26457ae28e9b299bf30a5a13a8eede55c27efe493e5ed42d743b58c6637778fb` |
| `environment_lock_hash` | `1687e0f3b95268e6096e6fb9922e0c6a49244e1828cf2e2d69bd92b67f229924` |

## Julia packages (locked)

| package | version |
|---|---|
| `JSON` | `1.8.0` |
| `Jutul` | `0.4.31` |
| `JutulDarcy` | `0.3.11` |
