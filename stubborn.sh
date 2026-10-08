#!/bin/bash

# This trap listens for Ctrl+C (SIGINT) and kills the whole script immediately
trap "echo -e '\nUser aborted. Exiting...'; pkill -9 -f 'microbleednet train'; exit 1" SIGINT

while true; do
    > temp_run.log
    
    ( tail -f temp_run.log 2>/dev/null | grep -q "device-side assert triggered" && pkill -9 -f "microbleednet train" ) &
    WATCHDOG_PID=$!

    script -q -e -c "CUDA_LAUNCH_BLOCKING=1 uv run microbleednet train --config configs/train.toml" temp_run.log
    EXIT_CODE=$?

    kill $WATCHDOG_PID 2>/dev/null

    if [ $EXIT_CODE -eq 0 ]; then
        echo "Training completed successfully!"
        break
    fi

    echo "Crash or freeze detected. Retrying in 2 seconds (Press Ctrl+C to stop)..."
    sleep 2
done
