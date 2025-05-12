#!/usr/bin/env nextflow
nextflow.enable.dsl=2

/*  -------------------------------------------------------------
    Moons & Gaussians experiment sweep
    -------------------------------------------------------------
    Launches the same commands you listed, parallelised by Nextflow.
    Override any param on the CLI, e.g.  --seed 42  --n 500  --output results/new
*/

params.seed       = 1
params.n          = 250
params.output     = "results/twomoons"
params.algorithms = ['clrot', 'frlc']
params.rs         = [1] + (5..100).step(5)

///////////////////////////////////////////////////////////////
//  Workflow definition                                      //
///////////////////////////////////////////////////////////////

workflow {
    Channel
        .fromList(params.algorithms)
        .cross(params.rs)      // all combinations (algo, r)
        .set { jobs }

    run_experiment(jobs)
}

///////////////////////////////////////////////////////////////
//  Processes                                                //
///////////////////////////////////////////////////////////////

process run_experiment {
    tag "${algo}_${r}"

    input:
        tuple val(algo), val(r)

    /* Adjust resources to your environment */
    cpus 1
    memory '2 GB'

    /* Collect each run’s outputs in the same directory */
    publishDir params.output, mode: 'copy'

    script:
    """
    python scripts/experiments/moons_and_gaussians.py \
        --seed ${params.seed} \
        -n ${params.n} \
        -r ${r} \
        --algorithm ${algo} \
        --output ${params.output}
    """
}
