# csv-analyst

A conversational data analyst agent built with the Mistral Agents SDK and deployed as a
Mistral-managed (hardened) deployment.

Share a CSV file from your Google Drive, ask questions in plain language, and get answers
computed from the real data.

## How it works

- The agent uses the `google_drive_mcp` connector, acting as you, to find the file you link
  and download its content.
- It answers using four fixed tools only — `load_csv`, `describe_columns`, `filter_rows`,
  `group_by_aggregate`. There is no sandbox and no arbitrary code execution.
- Sessions are durable and stored by Mistral, so follow-up questions in the same
  conversation remember the file and everything discussed.

## Tools

| Tool | What it does |
| --- | --- |
| `load_csv` | Parses the fetched CSV text and returns a `table_id`, row count and columns |
| `describe_columns` | Reports each column's type, non-empty count, distinct values, range and samples |
| `filter_rows` | Applies AND-combined conditions and returns a new `table_id` for the matching rows |
| `group_by_aggregate` | Groups rows and computes `count`, `sum`, `mean`, `min` or `max` per group |

## Develop

```bash
uv sync
uv run --env-file .env agents chat
```

## Deploy

```bash
git add -A && git commit -m "Update agent" && git push
uv run --env-file .env agents deploy
```
