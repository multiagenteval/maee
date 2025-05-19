#!/bin/bash

# Get the correct commit hash or fail
CORRECT_HASH=$(cd demo_repo && git rev-parse HEAD)
if [ -z "$CORRECT_HASH" ]; then
    echo "ERROR: Failed to determine the correct commit hash from demo_repo"
    exit 1
fi

echo "Using correct commit hash: $CORRECT_HASH"

# Find all workflow result files containing "demo" and update them
COUNT=0
for file in $(find output/workflow -name "workflow_results_*.json" -type f -exec grep -l "demo" {} \;); do
    echo "Updating $file"
    # Use sed to replace "commit_hash": "demo" with the correct commit hash
    sed -i '' 's/"commit_hash": "demo"/"commit_hash": "'"$CORRECT_HASH"'"/g' "$file"
    COUNT=$((COUNT+1))
done

# Check if the consolidated file needs to be updated
if grep -q '"commit_hash": "demo"' output/workflow/workflow_results_consolidated/workflow_results.json; then
    echo "Updating consolidated workflow results file"
    sed -i '' 's/"commit_hash": "demo"/"commit_hash": "'"$CORRECT_HASH"'"/g' output/workflow/workflow_results_consolidated/workflow_results.json
fi

echo "Commit hash updated in $COUNT workflow result files" 