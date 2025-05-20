nextflow.enable.dsl=2

params.proj_dir         = "/n/fs/ragr-research/projects/convex_lrot"
params.two_moons_script = "${params.proj_dir}/scripts/experiments/moons_and_gaussians.py"

params.seeds      = [1, 2, 3]
params.ns         = [50, 100, 250, 500, 1000, 2500, 5000]
params.output     = "results/simulations/"

process construct_simulations {
    cpus 8
    memory '4 GB'
    time '59m'

    clusterOptions '--gres=gpu:1'

    publishDir "nextflow_results/simulations/twomoons_n${n}_s${s}"

    input:
        tuple val(n), val(s)

    output:
        path("cost_matrix.txt")

    script:
    """
    python ${params.two_moons_script} --seed ${s} -n ${n} --output cost_matrix.txt
    """
}

workflow {
    sims = Channel
        .fromList(params.ns)      
        .combine(params.seeds)      

    sims | construct_simulations
}

