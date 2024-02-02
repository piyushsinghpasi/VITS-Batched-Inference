#!/bin/bash

# Set default values
audio_save_dir="./VITS_TTS_samples/"
noise_scale=0.667
noise_scale_w=0.8
length_scale=1

# Check if the correct number of arguments are provided
if [ "$#" -lt 6 ]; then
    echo "Usage: $0 --csv_file <csv_file> --gpu_ids <gpu_ids> --max_process <max_process> --batch_size <batch_size> --vits_config <vits_config> --vits_checkpoint <vits_checkpoint> [--audio_save_dir <audio_save_dir> --noise_scale <noise_scale> --noise_scale_w <noise_scale_w> --length_scale <length_scale>]"
    echo "See README (or this bash file's comments) for details"
    exit 1
fi

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --csv_file)
            csv_file="$2"
            shift 2
            ;;
        --gpu_ids)
            ngpus="$2"
            IFS=',' read -ra gpu_array <<< "$ngpus"
            shift 2
            ;;
        --max_process)
            max_process="$2"
            shift 2
            ;;
        --batch_size)
            batch_size="$2"
            shift 2
            ;;
        --vits_config)
            vits_config="$2"
            shift 2
            ;;
        --vits_checkpoint)
            vits_checkpoint="$2"
            shift 2
            ;;
        --audio_save_dir)
            audio_save_dir="$2"
            shift 2
            ;;
        --noise_scale)
            noise_scale="$2"
            shift 2
            ;;
        --noise_scale_w)
            noise_scale_w="$2"
            shift 2
            ;;
        --length_scale)
            length_scale="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "CSV File: $csv_file"
echo "GPU IDs: ${gpu_array[@]}"
echo "Max Parallel: $max_process"
echo "Batch Size: $batch_size"
echo "VITS Config: $vits_config"
echo "VITS Checkpoint: $vits_checkpoint"
echo "Audio Save Directory: $audio_save_dir"
echo "Noise Scale: $noise_scale"
echo "Noise Scale W: $noise_scale_w"
echo "Length Scale: $length_scale"

mkdir -p ./logs

# Count the number of lines in the CSV file and subtract 1 for the header
total_lines=$(wc -l < "$csv_file")
header_lines=1
data_lines=$((total_lines - header_lines))
# data_lines=20

# Calculate the appropriate batch size to limit parallel executions
num_rows=$((data_lines / max_process))
echo num rows "$num_rows"

gpu_index=0

# Loop through the iterations and run the Python script in parallel
for ((i=0; i<max_process; i++)); do
    
    start=$((i * num_rows))
    end=$((start + num_rows))

    # Adjust the end index for the last iteration
    if [ $i -eq $((max_process - 1)) ]; then
        end=$data_lines
    fi

    echo Processing "$start" to "$end"

    # Set CUDA_VISIBLE_DEVICES for each iteration 
    export CUDA_VISIBLE_DEVICES="${gpu_array[gpu_index]}"

    # move to next gpu_index in round robin fashion for next process
    gpu_index=$(( (gpu_index + 1) % ${#gpu_array[@]} ))

    # Define the log file for each iteration
    log_file="./logs/log_$((i+1)).txt"

    # Run the Python script in parallel for the specified batch range and log to the appropriate file
    python batched_vits_inference.py \
        --data_file "$csv_file" \
        --vits_config "$vits_config" \
        --vits_checkpoint "$vits_checkpoint" \
        --audio_saving_dir "$audio_save_dir" \
        --noise_scale "$noise_scale" \
        --noise_scale_w "$noise_scale_w" \
        --length_scale "$length_scale" \
        --batch_size "$batch_size" \
        --start_idx "$start" \
        --end_idx "$end" > "$log_file" 2>&1 &
done

# Wait for all background processes to finish
wait

echo "Processing completed. Log files created: log_1.txt, log_2.txt, ..., log_$max_process.txt"
echo Audio saved in "$audio_save_dir"
echo CAUTION audio will not be generated if "$audio_save_dir"/filename.wav already exists