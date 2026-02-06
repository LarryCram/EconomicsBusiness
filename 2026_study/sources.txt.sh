#!/bin/bash
# OpenAlex data pipeline
# Generated: Sat Feb  7 06:39:49 AEDT 2026

set -euo pipefail
trap 'echo "Error at $LINENO"; exit 1' ERR

INPUT="${1:-econ.txt}"
OUTPUT="$(basename "$INPUT" .txt)_works.parquet"

echo "Processing $INPUT -> $OUTPUT"

openalex works \
  --filter="publication_year:>2019,primary_location.source.id:\$(paste -sd '|' "$INPUT")" \
  --select="id,title,display_name,publication_year,authorships,concepts" \
  --rows=200 \
  --sample=50000 \
  --output=jsonl | \
duckdb -c "COPY (SELECT * FROM read_json_auto('/dev/stdin')) TO '$OUTPUT' (FORMAT PARQUET)"

echo "Done: $OUTPUT ($(duckdb -c "SELECT COUNT(*) FROM '$OUTPUT'" 2>/dev/null || echo "?"))"
