"""A conversational CSV data analyst.

The user shares a CSV file from their Google Drive and asks plain-language
questions. The agent fetches the file through the user's Google Drive
connector, then answers only from the data, using four fixed tools:
load_csv, describe_columns, filter_rows and group_by_aggregate. There is no
sandbox and no arbitrary code execution.
"""

import csv
import io
import uuid
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from mistralai import agents

MAX_ROWS = 200_000
MAX_COLUMNS = 500
PREVIEW_ROWS = 10
MAX_GROUP_RESULTS = 100

TABLES: dict[str, dict] = {}


class Filter(BaseModel):
    column: str = Field(description="Column name to filter on, exactly as in the CSV header.")
    operator: Literal["=", "!=", ">", ">=", "<", "<=", "contains", "not_contains"] = Field(
        description="Comparison operator."
    )
    value: str = Field(description="Value to compare against, as text.")


class Aggregation(BaseModel):
    column: str = Field(
        description="Column to aggregate. Use the column name itself for 'count'."
    )
    function: Literal["count", "sum", "mean", "min", "max"] = Field(
        description="Aggregate function to compute."
    )


def _to_number(text: str):
    if text is None:
        return None
    cleaned = text.strip().replace(",", "")
    if cleaned == "":
        return None
    try:
        return int(cleaned)
    except ValueError:
        pass
    try:
        return float(cleaned)
    except ValueError:
        return None


def _parse_numbers(values):
    numbers = [_to_number(v) for v in values]
    if numbers and all(n is not None for n in numbers):
        return numbers
    return None


def _get_table(table_id: str) -> dict:
    table = TABLES.get(table_id)
    if table is None:
        raise agents.ToolError(
            f"Unknown table_id '{table_id}'. Ask the user to share the CSV again, "
            "or reload it with load_csv."
        )
    return table


def _match(row: dict, column: str, operator: str, value: str) -> bool:
    cell = row.get(column)
    cell_text = "" if cell is None else str(cell)
    if operator in ("contains", "not_contains"):
        contained = value.lower() in cell_text.lower()
        return contained if operator == "contains" else not contained
    if operator in ("=", "!="):
        numbers = _parse_numbers([cell_text, value])
        if numbers is not None:
            equal = numbers[0] == numbers[1]
        else:
            equal = cell_text.strip().lower() == value.strip().lower()
        return equal if operator == "=" else not equal
    numbers = _parse_numbers([cell_text, value])
    if numbers is None:
        raise agents.ToolError(
            f"Cannot compare '{column}' with '{value}' using '{operator}': "
            "the column or the value is not numeric."
        )
    left, right = numbers
    return {
        ">": left > right,
        ">=": left >= right,
        "<": left < right,
        "<=": left <= right,
    }[operator]


def _validate_filters(columns: list[str], filters: list[Filter]) -> None:
    for condition in filters:
        if condition.column not in columns:
            raise agents.ToolError(
                f"Unknown column '{condition.column}'. Valid columns: {columns}."
            )


def _apply_filters(rows: list[dict], filters: list[Filter]) -> list[dict]:
    kept = rows
    for condition in filters:
        kept = [row for row in kept if _match(row, condition.column, condition.operator, condition.value)]
    return kept


def _column_values(rows: list[dict], column: str) -> list[str]:
    return [str(row[column]) for row in rows if row.get(column) not in (None, "")]


def _aggregate(rows: list[dict], aggregation: Aggregation):
    function = aggregation.function
    if function == "count":
        if aggregation.column:
            return len(_column_values(rows, aggregation.column))
        return len(rows)
    values = _column_values(rows, aggregation.column)
    numbers = _parse_numbers(values)
    if numbers is None:
        raise agents.ToolError(
            f"Column '{aggregation.column}' is not numeric, so '{function}' cannot be computed. "
            "Use 'count' instead, or check the column name."
        )
    if function == "sum":
        return round(sum(numbers), 6)
    if function == "mean":
        return round(sum(numbers) / len(numbers), 6)
    if function == "min":
        return min(numbers)
    return max(numbers)


@agents.tool(
    name="load_csv",
    description=(
        "Load CSV content that was already fetched from the user's Google Drive into a table "
        "and return its table_id, row count and columns. Give the raw CSV text, including the "
        "header row. Store the returned table_id: every other tool needs it."
    ),
)
async def load_csv(
    csv_content: Annotated[str, Field(description="The raw CSV text, including the header row.")],
    file_name: Annotated[str, Field(description="Name of the CSV file this data came from.")],
) -> dict:
    reader = csv.reader(io.StringIO(csv_content))
    records = [row for row in reader if any(cell.strip() != "" for cell in row)]
    if len(records) < 2:
        raise agents.ToolError(
            "The CSV content has no data rows. Check that the fetched content is the raw CSV "
            "text with a header row, not an error message or a preview."
        )
    header = [cell.strip() for cell in records[0]]
    if len(header) > MAX_COLUMNS:
        header = header[:MAX_COLUMNS]
    rows = []
    for record in records[1:MAX_ROWS]:
        row = {}
        for index, column in enumerate(header):
            row[column] = record[index].strip() if index < len(record) else ""
        rows.append(row)
    table_id = uuid.uuid4().hex[:12]
    TABLES[table_id] = {"file_name": file_name, "columns": header, "rows": rows}
    return {
        "table_id": table_id,
        "file_name": file_name,
        "row_count": len(rows),
        "columns": header,
    }


@agents.tool(
    name="describe_columns",
    description=(
        "Describe every column of a loaded table: inferred type, non-empty count, distinct "
        "values, numeric range and sample values. Call this before filtering or aggregating."
    ),
)
async def describe_columns(
    table_id: Annotated[str, Field(description="The table_id returned by load_csv.")],
) -> dict:
    table = _get_table(table_id)
    rows = table["rows"]
    columns_info = []
    for column in table["columns"]:
        values = _column_values(rows, column)
        distinct = sorted(set(values))
        info = {
            "column": column,
            "type": "empty",
            "non_empty": len(values),
            "distinct_count": len(distinct),
            "samples": distinct[:5],
        }
        numbers = _parse_numbers(values)
        if numbers is not None:
            info["type"] = "numeric"
            info["min"] = min(numbers)
            info["max"] = max(numbers)
        elif values:
            info["type"] = "text"
        columns_info.append(info)
    return {
        "table_id": table_id,
        "file_name": table["file_name"],
        "row_count": len(rows),
        "columns": columns_info,
    }


@agents.tool(
    name="filter_rows",
    description=(
        "Filter the rows of a table with one or more conditions combined with AND, and return "
        "a new table_id for the matching rows, their count and a preview. Use the exact column "
        "names and text values from describe_columns."
    ),
)
async def filter_rows(
    table_id: Annotated[str, Field(description="The table_id to filter.")],
    filters: Annotated[list[Filter], Field(description="Conditions combined with AND.")],
) -> dict:
    table = _get_table(table_id)
    _validate_filters(table["columns"], filters)
    matched = _apply_filters(table["rows"], filters)
    filtered_id = uuid.uuid4().hex[:12]
    TABLES[filtered_id] = {
        "file_name": table["file_name"],
        "columns": table["columns"],
        "rows": matched,
    }
    return {
        "table_id": filtered_id,
        "source_table_id": table_id,
        "matched_rows": len(matched),
        "total_rows": len(table["rows"]),
        "columns": table["columns"],
        "preview": matched[:PREVIEW_ROWS],
    }


@agents.tool(
    name="group_by_aggregate",
    description=(
        "Group the rows of a table by one or more columns and compute aggregates (count, sum, "
        "mean, min, max) per group. Optionally filter the rows first. Returns the real computed "
        "numbers per group."
    ),
)
async def group_by_aggregate(
    table_id: Annotated[str, Field(description="The table_id to aggregate.")],
    group_by: Annotated[list[str], Field(description="Column names to group by.")],
    aggregations: Annotated[list[Aggregation], Field(description="Aggregates to compute per group.")],
    filters: Annotated[list[Filter] | None, Field(description="Optional conditions applied first.")] = None,
) -> dict:
    table = _get_table(table_id)
    rows = table["rows"]
    if filters:
        _validate_filters(table["columns"], filters)
        rows = _apply_filters(rows, filters)
    for column in group_by:
        if column not in table["columns"]:
            raise agents.ToolError(
                f"Unknown column '{column}'. Valid columns: {table['columns']}."
            )
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = tuple(str(row.get(column, "")) for column in group_by)
        groups.setdefault(key, []).append(row)
    results = []
    for key in sorted(groups):
        members = groups[key]
        record = dict(zip(group_by, key))
        record["row_count"] = len(members)
        for aggregation in aggregations:
            label = f"{aggregation.function}_{aggregation.column}" if aggregation.column else aggregation.function
            record[label] = _aggregate(members, aggregation)
        results.append(record)
        if len(results) >= MAX_GROUP_RESULTS:
            break
    return {
        "table_id": table_id,
        "group_by": group_by,
        "groups_found": len(groups),
        "returned_groups": len(results),
        "results": results,
    }


INSTRUCTIONS = """You are a conversational data analyst for CSV files stored in the user's \
Google Drive.

How you work:
1. When the user shares a Drive link or file name, use your Google Drive connector tools to \
find the file (search_files or get_file_metadata) and fetch its content \
(download_file_content). For a plain CSV file, download the raw content; never use \
read_file_content for data, it may truncate it.
2. Pass the raw CSV text to load_csv. It returns a table_id. Remember that table_id for the \
whole conversation and reuse it for follow-up questions; do not reload the file unless the \
user asks about a different file or says the data changed.
3. Call describe_columns before answering questions about the data, so you use exact column \
names, types and values.
4. Answer every question with real numbers from the data: use filter_rows for row counts and \
subsets, and group_by_aggregate for per-group statistics. Never estimate, invent or compute \
numbers yourself: if a tool cannot answer it, say so.

Rules:
- You have only these data tools: load_csv, describe_columns, filter_rows, group_by_aggregate. \
There is no code execution. Do not try to write or run code.
- Quote exact column names in tool calls, as describe_columns reports them.
- When an answer needs several steps (filter then aggregate), chain the tools.
- Give clear answers with the real numbers, mention the row counts you computed them from, and \
say which file the numbers come from.
- If the user asks about something the CSV does not contain, say what columns are available \
instead of guessing.
- In follow-ups, remember the file, the table_id and everything discussed earlier in this \
conversation.

Beyond the CSV, you are connected to the user's Mistral knowledge bases through the \
'mistral' MCP server. When the user asks about background knowledge — AGI, ASI, knowledge, \
Verath, Chrono-Scribe, AetherLinkOS, Theia, or the content of their Markdown and PDF files — \
consult the knowledge base (list the knowledge bases, then read the relevant topic pages) and \
summarize what it says, citing which topic page it came from. Use the 'mistral' server only to \
read knowledge bases; never call its write or delete tools. If the knowledge base does not \
cover the question, say so. Keep CSV analysis and knowledge-base answers clearly separate: \
numbers about the user's CSV always come from the data tools, never from the knowledge base.
"""


agent = agents.Agent(
    model="mistral-large-4",
    name="csv-analyst",
    description=(
        "Conversational data analyst: loads a CSV from the user's Google Drive and answers "
        "plain-language questions with real numbers from the data."
    ),
    instructions=INSTRUCTIONS,
    tools=[load_csv, describe_columns, filter_rows, group_by_aggregate],
    connectors=[agents.connector("google_drive_mcp")],
    mcps={
        "mistral": agents.RemoteMCP(
            url="https://api.mistral.ai/mcp",
            auth_header_env_vars={"Authorization": "MISTRAL_MCP_AUTHORIZATION"},
            model_access="direct",
        )
    },
)
