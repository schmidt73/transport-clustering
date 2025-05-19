#!/usr/bin/env nextflow
nextflow.enable.dsl=2

params.proj_dir       = "/n/fs/ragr-research/projects/convex_lrot"
params.methods_script = "${params.proj_dir}/scripts/run_methods.py"
params.sims_dir       = "${params.proj_dir}/nextflow_results/simulations/"

params.algorithms = ['clrot', 'frlc', 'lot', 'fullrankround']
params.rs         = [1] + (5..100).step(5)

params.simulated_instances = [
    'twomoons_n100_s1',
    'twomoons_n250_s1',
    'twomoons_n500_s1',
    'twomoons_n1000_s1',
    'twomoons_n2500_s1',
    'twomoons_n5000_s1'
]

process run_methods {
    cpus 8
    memory '4 GB'
    time '59m'

    clusterOptions '--gres=gpu:1'

    publishDir "nextflow_results/algorithms/${algo}_r${r}_${id}/"

    input:
        tuple val(algo), val(r), val(id), path(cost_matrix)

    output:
        tuple path("results.csv"), path("timing.txt")

    script:
    """
    MOSEKLM_LICENSE_FILE=/n/fs/grad/hs2435
    /usr/bin/time -v python ${params.methods_script} \
        ${cost_matrix} \
        --seed 0 \
        --rank ${r} \
        --algorithm ${algo} \
        --output "results.csv" 2> timing.txt
    """
}

workflow {
    sims = Channel
        .fromList(params.algorithms)
        .combine(params.rs)
        .combine(params.simulated_instances)   

    instances = sims | map { algo, rank, id ->
        cost_matrix = "${params.sims_dir}/${id}/cost_matrix.txt"
        [algo, rank, id, cost_matrix]
    }

    instances | run_methods
}

