#!/usr/bin/env nextflow
nextflow.enable.dsl=2

params.two_moons_script = "/n/fs/ragr-research/projects/convex_lrot/scripts/experiments/moons_and_gaussians.py"

params.seeds      = [1, 2, 3]
params.ns         = [50, 100, 250, 500, 2500, 5000]
params.output     = "results/twomoons"
params.algorithms = ['clrot']
params.rs         = [1] + (5..100).step(5)

process run_experiment {
    cpus 8
    memory '4 GB'
    time '59m'

    clusterOptions '--gres=gpu:1'

    publishDir "nextflow_results/${algo}_n${n}_s${s}_r${r}/"

    input:
        tuple val(algo), val(r), val(n), val(s)

    output:
        tuple path("results.csv"), path("timing.txt")

    script:
    """
    module load gurobi
    MOSEKLM_LICENSE_FILE=/n/fs/grad/hs2435
    /usr/bin/time -v python ${params.two_moons_script} \
        --seed ${s} \
        -n ${n} \
        -r ${r} \
        --algorithm ${algo} \
        --output "results.csv" 2> timing.txt
    """
}

workflow {
    sims = Channel
        .fromList(params.algorithms)
        .combine(params.rs)      
        .combine(params.ns)      
        .combine(params.seeds)      

    sims | run_experiment
}

