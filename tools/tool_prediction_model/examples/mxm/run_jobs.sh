export OMP_NUM_THREADS=24
export CHAMELEON_TOOL=1
export CHAMELEON_TOOL_LIBRARIES=1

num_tasks=(200 400 600)
num_samples=(100, 200, 300, 400, 500)
num_epochs=(1000 2000 3000 4000 5000)

for i in "${num_tasks[@]}"; do
    export NUM_TASK=$i
    num_samp=$((i / 2))
    export NUM_SAMPLE=$num_samp
    export NUM_EPOCH=1000
    echo "Submit job with num_tasks = ${i}, num_samples = ${num_samp}, num_epochs=${NUM_EPOCH}"
    sbatch submit_script.cmd
done

# for test
# export NUM_TASK=400
# export NUM_SAMPLE=200
# export NUM_EPOCH=1000
# sbatch submit_script.cmd