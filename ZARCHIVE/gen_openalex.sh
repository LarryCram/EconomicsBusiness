#!/bin/bash
# process_econ_sources.sh

# Read each source ID from econ.txt and query OpenAlex
while read -r source_id; do
  echo "Processing source: $source_id"
  
  openalex download \
    --filter="publication_year:>2019,primary_location.source.id:$source_id" \
    --nested=""true" \
    --output="/home/lc/m/openalex_feb2026/data" \
    
  # Respect rate limits - add delay between requests
  sleep 1
done < sources.txt