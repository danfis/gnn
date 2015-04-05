/***
 * GNN
 * ----
 * Copyright (c)2015 Daniel Fiser <danfis@danfis.cz>
 *
 *  This file is part of GNN.
 *
 *  Distributed under the OSI-approved BSD License (the "License");
 *  see accompanying file BDS-LICENSE for details or see
 *  <http://www.opensource.org/licenses/bsd-license.php>.
 *
 *  This software is distributed WITHOUT ANY WARRANTY; without even the
 *  implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *  See the License for more information.
 */

#ifndef __GNN_GSRM_H__
#define __GNN_GSRM_H__

#include <boruvka/timer.h>
#include <boruvka/pc.h>
#include <boruvka/mesh3.h>
#include <boruvka/nn.h>
#include <boruvka/pairheap.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

struct _bor_gsrm_cache_t;

/** How often progress should be printed - it will be _approximately_
 *  every BOR_GSRM_PROGRESS_REFRESH'th new node. */
#define BOR_GSRM_PROGRESS_REFRESH 1000

/**
 * GSRM
 * =====
 *
 * TODO
 */

/** vvvv */
struct _bor_gsrm_params_t {
    size_t lambda;    /*!< Number of steps between adding nodes */
    bor_real_t eb;    /*!< Winner node learning rate */
    bor_real_t en;    /*!< Winners' neighbor learning rate */
    bor_real_t alpha; /*!< Decrease error counter rate */
    bor_real_t beta;  /*!< Decrease error counter rate for all nodes */
    int age_max;      /*!< Maximal age of edge */

    size_t max_nodes; /*!< Termination condition - a goal number of nodes */

    bor_real_t min_dangle;        /*! minimal dihedral angle between faces */
    bor_real_t max_angle;         /*! max angle between nodes to form face */
    bor_real_t angle_merge_edges; /*!< minimal angle between two edges to
                                       be merged */

    int verbosity; /*!< Verbosity level */

    bor_nn_params_t nn; /*!< Params for nearest neighbor search. Default is
                             used Growing Uniform Grid with default values */

    int unoptimized_err; /*!< True if unoptimized error handling should be
                              used. Default: false */
};
typedef struct _bor_gsrm_params_t bor_gsrm_params_t;

/**
 * Initializes parameters of GSRM to default values.
 */
void borGSRMParamsInit(bor_gsrm_params_t *params);

struct _bor_gsrm_t {
    bor_gsrm_params_t params; /*!< Parameters of algorithm */

    bor_pc_t *is;      /*!< Input signals */
    bor_pc_it_t isit;  /*!< Iterator over is */
    bor_mesh3_t *mesh; /*!< Reconstructed mesh */
    bor_nn_t *nn;      /*!< Search structure for nearest neighbor */

    bor_real_t *beta_n;        /*!< Precomputed beta^n for n = 1, ..., lambda */
    bor_real_t *beta_lambda_n; /*!< Precomputed beta^(n*lambda) for
                                    n = 1, ..., .beta_lambda_n_len */
    size_t beta_lambda_n_len;
    bor_pairheap_t *err_heap;

    size_t step;
    unsigned long cycle;

    bor_timer_t timer;

    struct _bor_gsrm_cache_t *c; /*!< Internal cache, don't touch it! */
};
typedef struct _bor_gsrm_t bor_gsrm_t;
/** ^^^^ */


/**
 * Allocates core struct and initializes to default values.
 */
bor_gsrm_t *borGSRMNew(const bor_gsrm_params_t *params);

/**
 * Deallocates struct.
 */
void borGSRMDel(bor_gsrm_t *g);

/**
 * Adds input signals from given file.
 * Returns number of read points.
 */
size_t borGSRMAddInputSignals(bor_gsrm_t *g, const char *fn);

/**
 * Runs GSRM algorithm.
 * Returns 0 on success.
 * Returns -1 if no there are no input signals.
 */
int borGSRMRun(bor_gsrm_t *g);

/**
 * Performs postprocessing of mesh.
 * This function should be called _after_ borGSRMRun().
 */
int borGSRMPostprocess(bor_gsrm_t *g);

/**
 * Return struct with GSRM parameters.
 */
_bor_inline const bor_gsrm_params_t *borGSRMParams(bor_gsrm_t *g);

/**
 * Returns pointer to mesh.
 */
_bor_inline bor_mesh3_t *borGSRMMesh(bor_gsrm_t *g);


/**** INLINES ****/
_bor_inline const bor_gsrm_params_t *borGSRMParams(bor_gsrm_t *g)
{
    return &g->params;
}

_bor_inline bor_mesh3_t *borGSRMMesh(bor_gsrm_t *g)
{
    return g->mesh;
}

#ifdef __cplusplus
} /* extern "C" */
#endif /* __cplusplus */

#endif /* __GNN_GSRM_H__ */
