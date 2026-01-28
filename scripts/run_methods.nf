#!/usr/bin/env nextflow
nextflow.enable.dsl=2

params.proj_dir       = "/n/fs/ragr-research/projects/convex_lrot"
params.methods_script = "${params.proj_dir}/scripts/run_methods.py"
params.sims_dir       = "${params.proj_dir}/simulated_data/"

params.algorithms = ['mr']//, 'frlc', 'lot', 'lin']
params.rs         = [50, 75, 100, 125, 150, 175, 200, 225, 250]
params.seeds      = [1, 2, 3, 4, 5]

params.simulated_instances = [
    'planted_gaussians/n5000_k250_sigma0.1_perturb0.1',
    'planted_gaussians/n5000_k250_sigma0.2_perturb0.1',
    'planted_gaussians/n5000_k250_sigma0.3_perturb0.1',
    'moons_and_gaussians/n5000_s1_noise0.1',
    'moons_and_gaussians/n5000_s1_noise0.25',
    'moons_and_gaussians/n5000_s1_noise0.5'
]

process run_methods {
    cpus 8
    memory '10 GB'
    time '59m'

    clusterOptions '--gres=gpu:1'

    publishDir "nextflow_results/algorithms/${algo}_r${r}_s${s}_${id}/"

    input:
        tuple val(algo), val(r), val(s), val(id), path(X_points), path(Y_points)

    output:
        tuple path("result_summary.json"), path("result_Q.txt"), path("result_R.txt"), path("timing.txt")

    script:
    """
    export TMPDIR="\${SLURM_TMPDIR:-$PWD/tmp}"
    mkdir -p "\$TMPDIR"
    MOSEKLM_LICENSE_FILE=/n/fs/grad/hs2435
    PYTHONPATH="/n/fs/ragr-research/projects/convex_lrot/src:$PYTHONPATH"
    /usr/bin/time -v python ${params.methods_script} \
        --points ${X_points} ${Y_points} \
        --seed ${s} \
        --rank ${r} \
        --algorithm ${algo} \
        --output result 2> timing.txt
    """
}

workflow {
    sims = Channel
        .fromList(params.algorithms)
        .combine(params.rs)
        .combine(params.seeds)
        .combine(params.simulated_instances)   

    instances = sims | map { algo, rank, seed, id ->
        cost_matrix = "${params.sims_dir}/${id}_cost_matrix.txt"
        X_points = "${params.sims_dir}/${id}_X.txt"
        Y_points = "${params.sims_dir}/${id}_Y.txt"
        [algo, rank, seed, id, X_points, Y_points]
    }

    instances | run_methods
}

