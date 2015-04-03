/***
 * fermat
 * -------
 * Copyright (c)2011 Daniel Fiser <danfis@danfis.cz>
 *
 *  This file is part of fermat.
 *
 *  Distributed under the OSI-approved BSD License (the "License");
 *  see accompanying file BDS-LICENSE for details or see
 *  <http://www.opensource.org/licenses/bsd-license.php>.
 *
 *  This software is distributed WITHOUT ANY WARRANTY; without even the
 *  implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *  See the License for more information.
 */

#ifndef __FER_GSRM_H__
#define __FER_GSRM_H__

#include <fermat/core.h>
#include <fermat/timer.h>
#include <fermat/pc.h>
#include <fermat/mesh3.h>
#include <fermat/nn.h>
#include <fermat/pairheap.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

struct _fer_gsrm_cache_t;

/** How often progress should be printed - it will be _approximately_
 *  every FER_GSRM_PROGRESS_REFRESH'th new node. */
#define FER_GSRM_PROGRESS_REFRESH 1000

/**
 * GSRM
 * =====
 *
 * TODO
 */

/** vvvv */
struct _fer_gsrm_params_t {
    size_t lambda;    /*!< Number of steps between adding nodes */
    fer_real_t eb;    /*!< Winner node learning rate */
    fer_real_t en;    /*!< Winners' neighbor learning rate */
    fer_real_t alpha; /*!< Decrease error counter rate */
    fer_real_t beta;  /*!< Decrease error counter rate for all nodes */
    int age_max;      /*!< Maximal age of edge */

    size_t max_nodes; /*!< Termination condition - a goal number of nodes */

    fer_real_t min_dangle;        /*! minimal dihedral angle between faces */
    fer_real_t max_angle;         /*! max angle between nodes to form face */
    fer_real_t angle_merge_edges; /*!< minimal angle between two edges to
                                       be merged */

    int verbosity; /*!< Verbosity level */

    fer_nn_params_t nn; /*!< Params for nearest neighbor search. Default is
                             used Growing Uniform Grid with default values */

    int unoptimized_err; /*!< True if unoptimized error handling should be
                              used. Default: false */
};
typedef struct _fer_gsrm_params_t fer_gsrm_params_t;

/**
 * Initializes parameters of GSRM to default values.
 */
void ferGSRMParamsInit(fer_gsrm_params_t *params);

struct _fer_gsrm_t {
    fer_gsrm_params_t params; /*!< Parameters of algorithm */

    fer_pc_t *is;      /*!< Input signals */
    fer_pc_it_t isit;  /*!< Iterator over is */
    fer_mesh3_t *mesh; /*!< Reconstructed mesh */
    fer_nn_t *nn;      /*!< Search structure for nearest neighbor */

    fer_real_t *beta_n;        /*!< Precomputed beta^n for n = 1, ..., lambda */
    fer_real_t *beta_lambda_n; /*!< Precomputed beta^(n*lambda) for
                                    n = 1, ..., .beta_lambda_n_len */
    size_t beta_lambda_n_len;
    fer_pairheap_t *err_heap;

    size_t step;
    unsigned long cycle;

    fer_timer_t timer;

    struct _fer_gsrm_cache_t *c; /*!< Internal cache, don't touch it! */
};
typedef struct _fer_gsrm_t fer_gsrm_t;
/** ^^^^ */


/**
 * Allocates core struct and initializes to default values.
 */
fer_gsrm_t *ferGSRMNew(const fer_gsrm_params_t *params);

/**
 * Deallocates struct.
 */
void ferGSRMDel(fer_gsrm_t *g);

/**
 * Adds input signals from given file.
 * Returns number of read points.
 */
size_t ferGSRMAddInputSignals(fer_gsrm_t *g, const char *fn);

/**
 * Runs GSRM algorithm.
 * Returns 0 on success.
 * Returns -1 if no there are no input signals.
 */
int ferGSRMRun(fer_gsrm_t *g);

/**
 * Performs postprocessing of mesh.
 * This function should be called _after_ ferGSRMRun().
 */
int ferGSRMPostprocess(fer_gsrm_t *g);

/**
 * Return struct with GSRM parameters.
 */
_fer_inline const fer_gsrm_params_t *ferGSRMParams(fer_gsrm_t *g);

/**
 * Returns pointer to mesh.
 */
_fer_inline fer_mesh3_t *ferGSRMMesh(fer_gsrm_t *g);


/**** INLINES ****/
_fer_inline const fer_gsrm_params_t *ferGSRMParams(fer_gsrm_t *g)
{
    return &g->params;
}

_fer_inline fer_mesh3_t *ferGSRMMesh(fer_gsrm_t *g)
{
    return g->mesh;
}

#ifdef __cplusplus
} /* extern "C" */
#endif /* __cplusplus */

#endif /* __FER_GSRM_H__ */

