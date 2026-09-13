# Environment report

Детерминированный отчёт: только то, что зафиксировано lock-файлами — без
timestamps, git-состояния и отпечатка машины, чтобы повторный gate не создавал
diff ни при повторе, ни на другой машине. Время запуска, git-состояние, `os`,
`arch`, `uv_version` и `julia_executable_version` — в
`artifacts/runs/<run_id>/environment_stamp.json`.

| key | value |
|---|---|
| `schema_version` | `3` |
| `python_version` | `3.13.2` |
| `uv_lock_sha256` | `d757313d9e7e308b2009300cd4f5ff438003b2e07926035e0433c03f0a6299a1` |
| `julia_pinned_version` | `1.12.7` |
| `julia_manifest_version` | `1.12.7` |
| `julia_manifest_sha256` | `1210c1f4b33a43db80f25bd71dd897d999cd7d59af8da9ab4d7b4bfa2d80b835` |
| `environment_lock_hash` | `376383cb30a5780b1a89be0744f6c6f438d0e824b53d2b4843f39392889a9046` |

## Julia packages (locked)

| package | version |
|---|---|
| `JSON` | `1.8.0` |
| `Jutul` | `0.4.31` |
| `JutulDarcy` | `0.3.11` |
