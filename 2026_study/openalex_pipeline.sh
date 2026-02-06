#!/bin/bash
# openalex_pipeline.sh - Complete pipeline in one file
# Usage: ./openalex_pipeline.sh [input_file] [sample_size]

set -euo pipefail

# Parameters
INPUT_FILE="${1:-econ.txt}"
SAMPLE_SIZE="${2:-50000}"
OUTPUT_FILE="$(basename "$INPUT_FILE" .txt)_works_$(date +%Y%m%d_%H%M%S).parquet"
LOG_FILE="logs/$(basename "$INPUT_FILE" .txt)_$(date +%Y%m%d).log"

# Create logs directory
mkdir -p logs

echo "=== OpenAlex Pipeline ===" | tee -a "$LOG_FILE"
echo "Input: $INPUT_FILE" | tee -a "$LOG_FILE"
echo "Output: $OUTPUT_FILE" | tee -a "$LOG_FILE"
echo "Sample: $SAMPLE_SIZE works" | tee -a "$LOG_FILE"

# Validate input
if [[ ! -f "$INPUT_FILE" ]]; then
    echo "ERROR: Input file $INPUT_FILE not found!" | tee -a "$LOG_FILE"
    exit 1
fi

# Build filter from source IDs
FILTER_SOURCES=$(paste -sd '|' "$INPUT_FILE")

echo "Downloading data from OpenAlex..." | tee -a "$LOG_FILE"

# Download and convert to Parquet
openalex works \
    --filter="publication_year:>2019,primary_location.source.id:$FILTER_SOURCES" \
    --select="id,title,display_name,publication_year,authorships,concepts,primary_location.source,cited_by_count" \
    --rows=200 \
    --sample="$SAMPLE_SIZE" \
    --output=jsonl | \
duckdb -c "
    INSTALL json; LOAD json;
    COPY (
        SELECT * FROM read_json_auto('/dev/stdin')
    ) TO '$OUTPUT_FILE' (FORMAT PARQUET);
"

# Verify output
if [[ -f "$OUTPUT_FILE" ]]; then
    RECORD_COUNT=$(duckdb -c "SELECT COUNT(*) FROM '$OUTPUT_FILE'" 2>/dev/null | grep -v '^---' | xargs)
    echo "Success!" | tee -a "$LOG_FILE"
    echo "Downloaded $RECORD_COUNT works to $OUTPUT_FILE" | tee -a "$LOG_FILE"
    echo "File size: $(du -h "$OUTPUT_FILE" | cut -f1)" | tee -a "$LOG_FILE"
else
    echo "ERROR: Output file was not created!" | tee -a "$LOG_FILE"
    exit 1
fi

echo "=== Pipeline Complete ===" | tee -a "$LOG_FILE"