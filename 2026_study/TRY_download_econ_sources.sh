#!/bin/bash
# process_econ_sources.sh

# Read each source ID from econ.txt and query OpenAlex
count=0
while read -r source_id; do
echo "Processing source: $source_id"
  
  openalex download \
    --filter="publication_year:>1999,primary_location.source.id:$source_id" \
    --output="/home/lc/m/openalex_feb26/json_" \
    --api-key="OchtksdohLaziRq08C4IJP" \

        # --nested \
    # --fresh \
    # --quiet \
    # --workers 8 \
    
  # Respect rate limits - add delay between requests
  sleep 1
  
  # Increment counter and exit after 16 sources
  count=$((count + 1))
  echo "Completed $count/19 sources"
done < sources.txt