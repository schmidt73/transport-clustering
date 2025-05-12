#!/usr/bin/env nextflow
nextflow.enable.dsl=2

params.two_moons_script = "/n/fs/ragr-research/projects/convex_lrot/scripts/experiments/moons_and_gaussians.py"

params.seed       = 1
params.n          = 500
params.output     = "results/twomoons"
params.algorithms = ['clrot'] // ['clrot', 'frlc']
params.rs         = [1] + (5..100).step(5)

process run_experiment {
    cpus 8
    memory '4 GB'
    time '59m'

    publishDir "nextflow_results/${algo}_n${params.n}_s${params.seed}_r${r}/"

    input:
        tuple val(algo), val(r)

    output:
        tuple path("results.csv"), path("timing.txt")

    script:
    """
    module load gurobi
    MOSEKLM_LICENSE_FILE=/n/fs/grad/hs2435
    /usr/bin/time -v python ${params.two_moons_script} \
        --seed ${params.seed} \
        -n ${params.n} \
        -r ${r} \
        --algorithm ${algo} \
        --output "results.csv" 2> timing.txt
    """
}

workflow {
    sims = Channel
        .fromList(params.algorithms)
        .combine(params.rs)      

    sims | run_experiment
}

