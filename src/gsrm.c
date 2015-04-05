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


#include <boruvka/alloc.h>
#include <boruvka/dbg.h>
#include <gnn/gsrm.h>

/** Print progress */
#define PR_PROGRESS(g) \
    borTimerStop(&g->timer); \
    borTimerPrintElapsed(&(g)->timer, stderr, " n: %d/%d, e: %d, f: %d\r", \
                         borMesh3VerticesLen((g)->mesh), \
                         (g)->params.max_nodes, \
                         borMesh3EdgesLen((g)->mesh), \
                         borMesh3FacesLen((g)->mesh)); \
    fflush(stderr)

#define PR_PROGRESS_PREFIX(g, prefix) \
    borTimerStop(&(g)->timer); \
    borTimerPrintElapsed(&(g)->timer, stderr, prefix " n: %d/%d, e: %d, f: %d\n", \
                         borMesh3VerticesLen((g)->mesh), \
                         (g)->params.max_nodes, \
                         borMesh3EdgesLen((g)->mesh), \
                         borMesh3FacesLen((g)->mesh)); \
    fflush(stderr)


struct _node_t {
    bor_vec3_t *v; /*!< Position of node (weight vector) */

    bor_real_t err;               /*!< Error counter */
    unsigned long err_cycle;      /*!< Last cycle in which were .err changed */
    bor_pairheap_node_t err_heap; /*!< Connection into error heap */

    bor_mesh3_vertex_t vert; /*!< Vertex in mesh */
    bor_nn_el_t nn;          /*!< Struct for NN search */
};
typedef struct _node_t node_t;

struct _edge_t {
    int age;      /*!< Age of edge */

    bor_mesh3_edge_t edge; /*!< Edge in mesh */
};
typedef struct _edge_t edge_t;

struct _face_t {
    bor_mesh3_face_t face; /*!< Face in mesh */
};
typedef struct _face_t face_t;

struct _bor_gsrm_cache_t {
    bor_vec3_t *is;     /*!< Input signal */
    node_t *nearest[2]; /*!< Two nearest nodes */

    node_t **common_neighb;    /*!< Array of common neighbors */
    size_t common_neighb_size; /*!< Size of .common_neighb - num allocated
                                    bytes */
    size_t common_neighb_len;  /*!< Number of nodes in .common_neighb */

    size_t err_counter_mark;      /*!< Contains mark used for accumalet
                                       error counter. It holds how many
                                       times were applied parameter alpha */
    bor_real_t err_counter_scale; /*!< Accumulated error counter - alpha^mark */

    bor_real_t pp_min, pp_max; /*!< Min and max area2 of face - used in
                                    postprocessing */
};
typedef struct _bor_gsrm_cache_t bor_gsrm_cache_t;


/** Allocates and deallocates cache */
static bor_gsrm_cache_t *cacheNew(void);
static void cacheDel(bor_gsrm_cache_t *c);


/** --- Node functions --- */
/** Creates new node and sets its weight to given vector */
static node_t *nodeNew(bor_gsrm_t *g, const bor_vec3_t *v);
/** Deletes node */
static void nodeDel(bor_gsrm_t *g, node_t *n);
/** Deletes node - proposed for borMesh3Del2() function */
static void nodeDel2(bor_mesh3_vertex_t *v, void *data);
/** Fixes node's error counter, i.e. applies correct beta^(n * lambda) */
_bor_inline void nodeFixError(bor_gsrm_t *gng, node_t *n);
/** Increment error counter */
_bor_inline void nodeIncError(bor_gsrm_t *gng, node_t *n, bor_real_t inc);
/** Scales error counter */
_bor_inline void nodeScaleError(bor_gsrm_t *gng, node_t *n, bor_real_t scale);


/** --- Edge functions --- */
/** Creates new edge as connection between two given nodes */
static edge_t *edgeNew(bor_gsrm_t *g, node_t *n1, node_t *n2);
/** Deletes edge */
static void edgeDel(bor_gsrm_t *g, edge_t *e);
/** Deteles edge - proposed for borMesh3Del2() function */
static void edgeDel2(bor_mesh3_edge_t *v, void *data);

/** --- Face functions --- */
static face_t *faceNew(bor_gsrm_t *g, edge_t *e, node_t *n);
static void faceDel(bor_gsrm_t *g, face_t *e);
static void faceDel2(bor_mesh3_face_t *v, void *data);


static int init(bor_gsrm_t *g);
static void adapt(bor_gsrm_t *g);
static void newNode(bor_gsrm_t *g);

/* Initializes mesh with three random nodes from input */
static void meshInit(bor_gsrm_t *g);

static void drawInputPoint(bor_gsrm_t *g);
/** Performes Extended Competitive Hebbian Learning */
static void echl(bor_gsrm_t *g);
static void echlConnectNodes(bor_gsrm_t *g);
static void echlMove(bor_gsrm_t *g);
static void echlUpdate(bor_gsrm_t *g);
/** Creates new node */
static void createNewNode(bor_gsrm_t *g);

/** Initializes mesh with three random nodes from input */
static void meshInit(bor_gsrm_t *g);
/** Choose random input signal and stores it in cache */
static void drawInputPoint(bor_gsrm_t *g);


/** --- ECHL functions --- */
/** Performs ECHL algorithm */
static void echl(bor_gsrm_t *g);
/** Gathers common neighbors of n1 and n2 and stores them in cache. */
static void echlCommonNeighbors(bor_gsrm_t *g, node_t *n1, node_t *n2);
/** Remove edge if it is inside thales sphere of n1, n2 and theirs common
 *  neighbors */
static void echlRemoveThales(bor_gsrm_t *g, edge_t *e, node_t *n1, node_t *n2);
/** Removes all edges that connect common neighbors between each other */
static void echlRemoveNeighborsEdges(bor_gsrm_t *g);
/** Create faces between given edge and common neighbors stored in cache's
 *  .common_neighb array */
static void echlCreateFaces(bor_gsrm_t *g, edge_t *e);
/** Connect winner nodes and if they are already connected update that
 *  connection */
static void echlConnectNodes(bor_gsrm_t *g);
/** Moves node towards input signal by given factor */
_bor_inline void echlMoveNode(bor_gsrm_t *g, node_t *n, bor_real_t k);
/** Move winner nodes towards input signal */
static void echlMove(bor_gsrm_t *g);
/** Updates all edges emitating from winning node */
static void echlUpdate(bor_gsrm_t *g);


/** -- Create New Node functions --- */
/** Performs "Create New Node" operation */
static void createNewNode(bor_gsrm_t *g);
/** Returns node with highest error counter */
static node_t *nodeWithHighestErrCounter(bor_gsrm_t *g);
/** Returns node with highests error counter that is neighbor of sq */
static node_t *nodesNeighborWithHighestErrCounter(bor_gsrm_t *g, node_t *sq);
/** Actually creates new node between sq and sf */
static node_t *createNewNode2(bor_gsrm_t *g, node_t *sq, node_t *sf);


/** --- Topology learning --- */
static void learnTopology(bor_gsrm_t *g);


/** --- Postprocessing functions --- */
/** Returns (via min, max, avg arguments) minimum, maximum and average area
 *  of faces in a mesh. */
static void faceAreaStat(bor_gsrm_t *g, bor_real_t *min, bor_real_t *max,
                         bor_real_t *avg);
/** Deletes incorrect faces from mesh */
static void delIncorrectFaces(bor_gsrm_t *g);
/** Deletes incorrect edges from mesh */
static void delIncorrectEdges(bor_gsrm_t *g);
/** Merges all edges that can be merged */
static void mergeEdges(bor_gsrm_t *g);
/** Tries to finish (triangulate) surface */
static void finishSurface(bor_gsrm_t *g);
/** Deletes lonely nodes, edges and faces */
static void delLonelyNodesEdgesFaces(bor_gsrm_t *g);
/** Embed triangles everywhere it can */
static void finishSurfaceEmbedTriangles(bor_gsrm_t *g);
/** Returns true if all internal angles of face is smaller than
 *  g->params.max_angle */
static int faceCheckAngle(bor_gsrm_t *g, bor_mesh3_vertex_t *v1,
                          bor_mesh3_vertex_t *v2, bor_mesh3_vertex_t *v3);
/** Deletes one of triangles (the triangles are considered to have dihedral
 *  angle smaller than g->params.min_dangle.
 *  First is deleted face that have less incidenting faces. If both have
 *  same, faces with smaller area is deleted. */
static void delFacesDangle(bor_gsrm_t *g, face_t *f1, face_t *f2);
/** Returns true if given edge can't be used for face creation */
static int edgeNotUsable(bor_mesh3_edge_t *e);
/** Tries to finish triangle incidenting with e.
 *  It's assumend that e has already one incidenting face. */
static int finishSurfaceTriangle(bor_gsrm_t *g, edge_t *e);
/** Tries to create completely new face */
static int finishSurfaceNewFace(bor_gsrm_t *g, edge_t *e);

static int errHeapLT(const bor_pairheap_node_t *_n1,
                     const bor_pairheap_node_t *_n2,
                     void *data);

void borGSRMParamsInit(bor_gsrm_params_t *params)
{
    params->lambda = 200;
    params->eb = 0.05;
    params->en = 0.0006;
    params->alpha = 0.95;
    params->beta = 0.9995;
    params->age_max = 200;
    params->max_nodes = 5000;

    params->min_dangle = M_PI_4;
    params->max_angle = M_PI_2 * 1.5;
    params->angle_merge_edges = M_PI * 0.9;

    params->verbosity = 1;

    borNNParamsInit(&params->nn);
    params->nn.type = BOR_NN_GUG;
    params->nn.gug.dim = 3;
    params->nn.vptree.dim = 3;
    params->nn.linear.dim = 3;

    params->unoptimized_err = 0;
}

bor_gsrm_t *borGSRMNew(const bor_gsrm_params_t *params)
{
    bor_gsrm_t *g;

    g = BOR_ALLOC(bor_gsrm_t);
    g->params = *params;

    // initialize point cloude (input signals)
    g->is = borPCNew(3);

    // init 3D mesh
    g->mesh = borMesh3New();

    // init nn for NN search to NULL, actual allocation will be made
    // after we know what area do we need to cover
    g->nn = NULL;

    g->c = NULL;

    g->beta_n = NULL;
    g->beta_lambda_n = NULL;
    g->beta_lambda_n_len = 0;

    g->err_heap = NULL;

    return g;
}

void borGSRMDel(bor_gsrm_t *g)
{
    if (g->c)
        cacheDel(g->c);

    if (g->is)
        borPCDel(g->is);

    if (g->mesh)
        borMesh3Del2(g->mesh, nodeDel2, (void *)g,
                              edgeDel2, (void *)g,
                              faceDel2, (void *)g);

    if (g->nn)
        borNNDel(g->nn);

    if (g->beta_n)
        BOR_FREE(g->beta_n);
    if (g->beta_lambda_n)
        BOR_FREE(g->beta_lambda_n);
    if (g->err_heap)
        borPairHeapDel(g->err_heap);

    BOR_FREE(g);
}

size_t borGSRMAddInputSignals(bor_gsrm_t *g, const char *fn)
{
    return borPCAddFromFile(g->is, fn);
}

int borGSRMRun(bor_gsrm_t *g)
{
    size_t cycle;

    cycle = 0;
    init(g);

    if (g->params.verbosity >= 1){
        PR_PROGRESS(g);
    }

    do {
        for (g->step = 1; g->step <= g->params.lambda; g->step++){
            adapt(g);
        }
        newNode(g);

        cycle++;
        if (g->params.verbosity >= 2
                && bor_unlikely(cycle == BOR_GSRM_PROGRESS_REFRESH)){
            PR_PROGRESS(g);
            cycle = 0;
        }

        g->cycle++;
    } while (borMesh3VerticesLen(g->mesh) < g->params.max_nodes);

    if (g->params.verbosity >= 1){
        PR_PROGRESS(g);
        fprintf(stderr, "\n");
        fflush(stderr);
    }

    learnTopology(g);

    return 0;
}


int borGSRMPostprocess(bor_gsrm_t *g)
{
    bor_real_t min, max, avg;

    // set up limits
    faceAreaStat(g, &min, &max, &avg);
    g->c->pp_min = min;
    g->c->pp_max = max;

    // Phase 1:
    // Remove incorrect faces from whole mesh.
    // Incorrect faces are those that have any of internal angle bigger
    // than treshold (.param.max_angle) or if dihedral angle with any other
    // face is smaller than treshold (.param.min_dangle)
    delIncorrectFaces(g);
    if (g->params.verbosity >= 3){
        PR_PROGRESS_PREFIX(g, " -1- DIF:");
    }

    // Remove incorrect edges.
    // Incorrect edges are those that have (on one end) no incidenting
    // edge or if edge can't be used for face creation.
    delIncorrectEdges(g);
    if (g->params.verbosity >= 3){
        PR_PROGRESS_PREFIX(g, " -1- DIE:");
    }

    // Merge edges.
    // Merge all pairs of edges that have common node that have no other
    // incidenting edges than the two.
    mergeEdges(g);
    if (g->params.verbosity >= 3){
        PR_PROGRESS_PREFIX(g, " -1- ME: ");
    }

    // Finish surface
    finishSurface(g);
    if (g->params.verbosity >= 3){
        PR_PROGRESS_PREFIX(g, " -1- FS: ");
    }else if (g->params.verbosity >= 2){
        PR_PROGRESS_PREFIX(g, " -1-");
    }


    // Phase 2:
    // Del incorrect edges again
    delIncorrectEdges(g);
    if (g->params.verbosity >= 3){
        PR_PROGRESS_PREFIX(g, " -2- DIE:");
    }

    // finish surface
    finishSurface(g);
    if (g->params.verbosity >= 3){
        PR_PROGRESS_PREFIX(g, " -2- FS: ");
    }else if (g->params.verbosity >= 2){
        PR_PROGRESS_PREFIX(g, " -2-");
    }


    // Phase 3:
    // delete lonely faces, edges and nodes
    delLonelyNodesEdgesFaces(g);
    if (g->params.verbosity >= 3){
        PR_PROGRESS_PREFIX(g, " -3- DL: ");
    }

    // try finish surface again
    finishSurface(g);
    if (g->params.verbosity >= 3){
        PR_PROGRESS_PREFIX(g, " -3- FS: ");
    }else if (g->params.verbosity >= 2){
        PR_PROGRESS_PREFIX(g, " -3-");
    }

    // Phase 4:
    // delete lonely faces, edges and nodes
    delLonelyNodesEdgesFaces(g);
    if (g->params.verbosity >= 3){
        PR_PROGRESS_PREFIX(g, " -4- DL: ");
    }

    finishSurfaceEmbedTriangles(g);
    if (g->params.verbosity >= 3){
        PR_PROGRESS_PREFIX(g, " -4- FET:");
    }else if (g->params.verbosity >= 2){
        PR_PROGRESS_PREFIX(g, " -4-");
    }else if (g->params.verbosity >= 1){
        PR_PROGRESS(g);
        fprintf(stderr, "\n");
    }

    return 0;
}


static int init(bor_gsrm_t *g)
{
    size_t i;
    bor_real_t maxbeta;
    bor_real_t aabb[6];

    // check if there are some input signals
    if (borPCLen(g->is) <= 3){
        DBG2("No input signals!");
        return -1;
    }

    g->cycle = 1L;

    // initialize error heap
    if (!g->params.unoptimized_err){
        if (g->err_heap)
            borPairHeapDel(g->err_heap);
        g->err_heap = borPairHeapNew(errHeapLT, (void *)g);
    }

    // precompute beta^n
    if (g->beta_n)
        BOR_FREE(g->beta_n);

    g->beta_n = BOR_ALLOC_ARR(bor_real_t, g->params.lambda);
    g->beta_n[0] = g->params.beta;
    for (i = 1; i < g->params.lambda; i++){
        g->beta_n[i] = g->beta_n[i - 1] * g->params.beta;
    }

    // precompute beta^(n * lambda)
    if (g->beta_lambda_n)
        BOR_FREE(g->beta_lambda_n);

    maxbeta = g->beta_n[g->params.lambda - 1];

    g->beta_lambda_n_len = 1000;
    g->beta_lambda_n = BOR_ALLOC_ARR(bor_real_t, g->beta_lambda_n_len);
    g->beta_lambda_n[0] = maxbeta;
    for (i = 1; i < g->beta_lambda_n_len; i++){
        g->beta_lambda_n[i] = g->beta_lambda_n[i - 1] * maxbeta;
    }

    // initialize cache
    if (!g->c)
        g->c = cacheNew();

    // initialize NN search structure
    if (g->nn)
        borNNDel(g->nn);
    borPCAABB(g->is, aabb);
    g->params.nn.linear.dim = 3;
    g->params.nn.vptree.dim = 3;
    g->params.nn.gug.dim    = 3;
    g->params.nn.gug.aabb   = aabb;
    g->nn = borNNNew(&g->params.nn);

    // first shuffle of all input signals
    borPCPermutate(g->is);
    // and initialize its iterator
    borPCItInit(&g->isit, g->is);


    // start timer
    borTimerStart(&g->timer);

    // initialize mesh with three random nodes
    meshInit(g);

    return 0;

}

static void adapt(bor_gsrm_t *g)
{
    drawInputPoint(g);
    echl(g);
}

static void newNode(bor_gsrm_t *g)
{
    createNewNode(g);
}

static bor_gsrm_cache_t *cacheNew(void)
{
    bor_gsrm_cache_t *c;

    c = BOR_ALLOC(bor_gsrm_cache_t);
    c->nearest[0] = c->nearest[1] = NULL;

    c->common_neighb_size = 3;
    c->common_neighb = BOR_ALLOC_ARR(node_t *, c->common_neighb_size);
    c->common_neighb_len = 0;

    c->err_counter_mark = 0;
    c->err_counter_scale = BOR_ONE;

    return c;
}

static void cacheDel(bor_gsrm_cache_t *c)
{
    BOR_FREE(c->common_neighb);
    BOR_FREE(c);
}



/** --- Node functions --- **/
static node_t *nodeNew(bor_gsrm_t *g, const bor_vec3_t *v)
{
    node_t *n;

    n = BOR_ALLOC(node_t);
    n->v = borVec3Clone(v);

    // initialize mesh's vertex struct with weight vector
    borMesh3VertexSetCoords(&n->vert, n->v);

    // initialize cells struct with its own weight vector
    borNNElInit(g->nn, &n->nn, (bor_vec_t *)n->v);

    // add node into mesh
    borMesh3AddVertex(g->mesh, &n->vert);
    // and add node into cells
    borNNAdd(g->nn, &n->nn);

    // set error counter
    n->err = BOR_ZERO;
    if (!g->params.unoptimized_err){
        n->err_cycle = g->cycle;
        borPairHeapAdd(g->err_heap, &n->err_heap);
    }

    //DBG("n: %lx, vert: %lx (%g %g %g)", (long)n, (long)&n->vert,
    //    borVec3X(&n->v), borVec3Y(&n->v), borVec3Z(&n->v));

    return n;
}

_bor_inline void nodeFixError(bor_gsrm_t *g, node_t *n)
{
    unsigned long diff;

    diff = g->cycle - n->err_cycle;
    if (diff > 0 && diff <= g->beta_lambda_n_len){
        n->err *= g->beta_lambda_n[diff - 1];
    }else if (diff > 0){
        n->err *= g->beta_lambda_n[g->params.lambda - 1];

        diff = diff - g->beta_lambda_n_len;
        n->err *= pow(g->beta_n[g->params.lambda - 1], diff);
    }
    n->err_cycle = g->cycle;
}

_bor_inline void nodeIncError(bor_gsrm_t *g, node_t *n, bor_real_t inc)
{
    nodeFixError(g, n);
    n->err += inc;
    borPairHeapUpdate(g->err_heap, &n->err_heap);
}

_bor_inline void nodeScaleError(bor_gsrm_t *g, node_t *n, bor_real_t scale)
{
    nodeFixError(g, n);
    n->err *= scale;
    borPairHeapUpdate(g->err_heap, &n->err_heap);
}

static void nodeDel(bor_gsrm_t *g, node_t *n)
{
    bor_list_t *list, *item, *item_tmp;
    bor_mesh3_edge_t *edge;
    edge_t *e;
    int res;

    // remove node from mesh
    if (bor_unlikely(borMesh3VertexEdgesLen(&n->vert) > 0)){
        // remove edges first
        list = borMesh3VertexEdges(&n->vert);
        BOR_LIST_FOR_EACH_SAFE(list, item, item_tmp){
            edge = borMesh3EdgeFromVertexList(item);
            e = bor_container_of(edge, edge_t, edge);

            edgeDel(g, e);
        }
    }

    // then vertex
    res = borMesh3RemoveVertex(g->mesh, &n->vert);
    if (bor_unlikely(res != 0)){
        DBG2("Node couldn't be removed from mesh - this shouldn't happen!");
        exit(-1);
    }

    borVec3Del(n->v);

    // remove node from cells
    borNNRemove(g->nn, &n->nn);

    // remove from error heap
    if (!g->params.unoptimized_err){
        borPairHeapRemove(g->err_heap, &n->err_heap);
    }

    // Note: no need of deallocation of .vert and .cells
    BOR_FREE(n);
}

static void nodeDel2(bor_mesh3_vertex_t *v, void *data)
{
    bor_gsrm_t *g = (bor_gsrm_t *)data;
    node_t *n;
    n = bor_container_of(v, node_t, vert);

    borVec3Del(n->v);

    // remove node from cells
    borNNRemove(g->nn, &n->nn);

    BOR_FREE(n);
}



/** --- Edge functions --- **/
static edge_t *edgeNew(bor_gsrm_t *g, node_t *n1, node_t *n2)
{
    edge_t *e;

    e = BOR_ALLOC(edge_t);
    e->age = 0;

    borMesh3AddEdge(g->mesh, &e->edge, &n1->vert, &n2->vert);

    //DBG("e: %lx, edge: %lx", (long)e, (long)&e->edge);

    return e;
}

static void edgeDel(bor_gsrm_t *g, edge_t *e)
{
    bor_mesh3_face_t *face;
    int res;

    // first remove incidenting faces
    while ((face = borMesh3EdgeFace(&e->edge, 0)) != NULL){
        faceDel(g, bor_container_of(face, face_t, face));
    }

    // then remove edge itself
    res = borMesh3RemoveEdge(g->mesh, &e->edge);
    if (bor_unlikely(res != 0)){
        DBG2("Can't remove edge - this shouldn'h happen!");
        exit(-1);
    }

    BOR_FREE(e);
}

static void edgeDel2(bor_mesh3_edge_t *edge, void *data)
{
    edge_t *e;
    e = bor_container_of(edge, edge_t, edge);
    BOR_FREE(e);
}




/** --- Face functions --- **/
static face_t *faceNew(bor_gsrm_t *g, edge_t *e, node_t *n)
{
    face_t *f;
    bor_mesh3_edge_t *e2, *e3;
    int res;

    e2 = borMesh3VertexCommonEdge(borMesh3EdgeVertex(&e->edge, 0), &n->vert);
    e3 = borMesh3VertexCommonEdge(borMesh3EdgeVertex(&e->edge, 1), &n->vert);
    if (bor_unlikely(!e2 || !e3)){
        DBG2("Can't create face because *the* three nodes are not connected "
             " - this shouldn't happen!");
        return NULL;
    }

    f = BOR_ALLOC(face_t);

    res = borMesh3AddFace(g->mesh, &f->face, &e->edge, e2, e3);
    if (bor_unlikely(res != 0)){
        BOR_FREE(f);
        return NULL;
    }

    // TODO: check if face already exists

    return f;
}

static void faceDel(bor_gsrm_t *g, face_t *f)
{
    borMesh3RemoveFace(g->mesh, &f->face);
    BOR_FREE(f);
}

static void faceDel2(bor_mesh3_face_t *face, void *data)
{
    face_t *f;
    f = bor_container_of(face, face_t, face);
    BOR_FREE(f);
}




static void meshInit(bor_gsrm_t *g)
{
    bor_vec3_t *v;
    size_t i;

    for (i = 0; i < 3; i++){
        // obtain input signal
        v = (bor_vec3_t *)borPCItGet(&g->isit);

        // create new node
        nodeNew(g, v);

        // move to next point
        borPCItNext(&g->isit);
    }
}

static void drawInputPoint(bor_gsrm_t *g)
{
    if (borPCItEnd(&g->isit)){
        // if iterator is at the end permutate point cloud again
        borPCPermutate(g->is);
        // and re-initialize iterator
        borPCItInit(&g->isit, g->is);
    }
    g->c->is = (bor_vec3_t *)borPCItGet(&g->isit);
    borPCItNext(&g->isit);
}



/** --- ECHL functions --- **/
static void decreaseAllErrors(bor_gsrm_t *g)
{
    bor_list_t *list, *item;
    bor_mesh3_vertex_t *v;
    node_t *n;

    list = borMesh3Vertices(g->mesh);
    BOR_LIST_FOR_EACH(list, item){
        v = BOR_LIST_ENTRY(item, bor_mesh3_vertex_t, list);
        n = bor_container_of(v, node_t, vert);
        n->err = n->err * g->params.beta;
    }
}


static void echl(bor_gsrm_t *g)
{
    bor_nn_el_t *el[2];

    // 1. Find two nearest nodes
    borNNNearest(g->nn, (const bor_vec_t *)g->c->is, 2, el);
    g->c->nearest[0] = bor_container_of(el[0], node_t, nn);
    g->c->nearest[1] = bor_container_of(el[1], node_t, nn);

    // 2. Connect winning nodes
    echlConnectNodes(g);

    // 3. Move winning node and its neighbors towards input signal
    echlMove(g);

    // 4. Update all edges emitating from winning node
    echlUpdate(g);

    if (g->params.unoptimized_err){
        decreaseAllErrors(g);
    }
}

static void echlCommonNeighbors(bor_gsrm_t *g, node_t *n1, node_t *n2)
{
    bor_list_t *list1, *list2;
    bor_list_t *item1, *item2;
    bor_mesh3_edge_t *edge1, *edge2;
    bor_mesh3_vertex_t *o1, *o2;
    node_t *n;
    size_t len;

    // allocate enough memory for common neighbors
    if (g->c->common_neighb_size < borMesh3VertexEdgesLen(&n1->vert)
            && g->c->common_neighb_size < borMesh3VertexEdgesLen(&n2->vert)){
        len = borMesh3VertexEdgesLen(&n1->vert);
        len = BOR_MIN(len, borMesh3VertexEdgesLen(&n2->vert));

        g->c->common_neighb = BOR_REALLOC_ARR(g->c->common_neighb, node_t *, len);
        g->c->common_neighb_size = len;
    }

    list1 = borMesh3VertexEdges(&n1->vert);
    list2 = borMesh3VertexEdges(&n2->vert);
    len = 0;
    BOR_LIST_FOR_EACH(list1, item1){
        edge1 = borMesh3EdgeFromVertexList(item1);
        o1 = borMesh3EdgeVertex(edge1, 0);
        if (o1 == &n1->vert)
            o1 = borMesh3EdgeVertex(edge1, 1);

        BOR_LIST_FOR_EACH(list2, item2){
            edge2 = borMesh3EdgeFromVertexList(item2);
            o2 = borMesh3EdgeVertex(edge2, 0);
            if (o2 == &n2->vert)
                o2 = borMesh3EdgeVertex(edge2, 1);

            if (o1 == o2){
                n = bor_container_of(o1, node_t, vert);
                g->c->common_neighb[len] = n;
                len++;
            }
        }
    }

    g->c->common_neighb_len = len;
}

static void echlRemoveThales(bor_gsrm_t *g, edge_t *e, node_t *n1, node_t *n2)
{
    node_t *nb;
    size_t i, len;

    len = g->c->common_neighb_len;
    for (i=0; i < len; i++){
        nb = g->c->common_neighb[i];

        if (borVec3Angle(n1->v, nb->v, n2->v) > M_PI_2){
            // remove edge
            edgeDel(g, e);
            return;
        }
    }
}

static void echlRemoveNeighborsEdges(bor_gsrm_t *g)
{
    size_t i, j, len;
    bor_mesh3_edge_t *edge;
    node_t **ns;
    edge_t *e;

    ns = g->c->common_neighb;
    len = g->c->common_neighb_len;
    if (len == 0)
        return;

    for (i = 0; i < len; i++){
        for (j = i + 1; j < len; j++){
            edge = borMesh3VertexCommonEdge(&ns[i]->vert, &ns[j]->vert);
            if (edge != NULL){
                e = bor_container_of(edge, edge_t, edge);
                edgeDel(g, e);
            }
        }
    }
}

static void echlCreateFaces(bor_gsrm_t *g, edge_t *e)
{
    size_t i, len;
    node_t **ns;

    len = BOR_MIN(g->c->common_neighb_len, 2);
    ns = g->c->common_neighb;
    for (i = 0; i < len; i++){
        faceNew(g, e, ns[i]);
    }
}

static void echlConnectNodes(bor_gsrm_t *g)
{
    bor_mesh3_edge_t *edge;
    edge_t *e;
    node_t *n1, *n2;

    n1 = g->c->nearest[0];
    n2 = g->c->nearest[1];

    // get edge connecting n1 and n2
    e = NULL;
    edge = borMesh3VertexCommonEdge(&n1->vert, &n2->vert);
    if (edge){
        e = bor_container_of(edge, edge_t, edge);
    }

    // get common neighbors
    echlCommonNeighbors(g, n1, n2);

    if (e != NULL){
        //DBG2("Nodes are connected");

        // set age of edge to zero
        e->age = 0;

        // Remove edge if opposite node lies inside thales sphere
        echlRemoveThales(g, e, n1, n2);
    }else{
        //DBG2("Nodes are NOT connected");

        // remove all edges that connect common neighbors
        echlRemoveNeighborsEdges(g);

        // create new edge between n1 and n2
        e = edgeNew(g, n1, n2);

        // create faces with common neighbors
        echlCreateFaces(g, e);
    }
}

_bor_inline void echlMoveNode(bor_gsrm_t *g, node_t *n, bor_real_t k)
{
    bor_vec3_t v;

    // compute shifting
    borVec3Sub2(&v, g->c->is, n->v);
    borVec3Scale(&v, k);

    // move node
    borVec3Add(n->v, &v);

    // update node in search structure
    borNNUpdate(g->nn, &n->nn);
}

static void echlMove(bor_gsrm_t *g)
{
    bor_list_t *list, *item;
    bor_mesh3_edge_t *edge;
    bor_mesh3_vertex_t *wvert, *vert;
    node_t *wn;
    bor_real_t err;

    wn = g->c->nearest[0];
    wvert = &wn->vert;

    // move winning node
    echlMoveNode(g, wn, g->params.eb);

    // increase error counter
    if (!g->params.unoptimized_err){
        err  = borVec3Dist2(wn->v, g->c->is);
        err *= g->beta_n[g->params.lambda - g->step];
        nodeIncError(g, wn, err);
    }else{
        wn->err += borVec3Dist2(wn->v, g->c->is);
    }

    // move nodes connected with the winner
    list = borMesh3VertexEdges(wvert);
    BOR_LIST_FOR_EACH(list, item){
        edge = borMesh3EdgeFromVertexList(item);
        vert = borMesh3EdgeOtherVertex(edge, wvert);

        echlMoveNode(g, bor_container_of(vert, node_t, vert), g->params.en);
    }
}

static void echlUpdate(bor_gsrm_t *g)
{
    node_t *wn;
    edge_t *e;
    bor_list_t *list, *item, *tmp_item;
    bor_mesh3_edge_t *edge;
    bor_mesh3_vertex_t *vert;

    wn = g->c->nearest[0];

    list = borMesh3VertexEdges(&wn->vert);
    BOR_LIST_FOR_EACH_SAFE(list, item, tmp_item){
        edge = borMesh3EdgeFromVertexList(item);
        e = bor_container_of(edge, edge_t, edge);

        // increment age of edge
        e->age++;

        // if age of edge is above treshold remove edge and nodes which
        // remain unconnected
        if (e->age > g->params.age_max){
            // get other node than winning one
            vert = borMesh3EdgeOtherVertex(edge, &wn->vert);

            // delete edge
            edgeDel(g, e);

            // check if n is connected in mesh, if not delete it
            if (borMesh3VertexEdgesLen(vert) == 0){
                nodeDel(g, bor_container_of(vert, node_t, vert));
            }
        }
    }

    // check if winning node remains connected
    if (borMesh3VertexEdgesLen(&wn->vert) == 0){
        nodeDel(g, wn);
    }
}




/** --- Create New Node functions --- **/
static node_t *nodeWithHighestErrCounterLinear(bor_gsrm_t *g)
{
    bor_list_t *list, *item;
    bor_mesh3_vertex_t *v;
    node_t *n;
    bor_real_t err;
    node_t *max;

    max = NULL;
    err = -BOR_REAL_MAX;

    list = borMesh3Vertices(g->mesh);
    BOR_LIST_FOR_EACH(list, item){
        v = BOR_LIST_ENTRY(item, bor_mesh3_vertex_t, list);
        n = bor_container_of(v, node_t, vert);

        if (n->err > err){
            max = n;
            err = n->err;
        }
    }

    return max;
}

static node_t *nodeWithHighestErrCounter(bor_gsrm_t *g)
{
    bor_pairheap_node_t *max;
    node_t *maxn;

    if (g->params.unoptimized_err){
        return nodeWithHighestErrCounterLinear(g);
    }

    max  = borPairHeapMin(g->err_heap);
    maxn = bor_container_of(max, node_t, err_heap);

    return maxn;
}

static node_t *nodesNeighborWithHighestErrCounter(bor_gsrm_t *g, node_t *sq)
{
    bor_list_t *list, *item;
    bor_mesh3_edge_t *edge;
    bor_mesh3_vertex_t *other_vert;
    bor_real_t max_err;
    node_t *n, *max_n;


    max_err = -BOR_REAL_MAX;
    max_n   = NULL;
    list = borMesh3VertexEdges(&sq->vert);
    BOR_LIST_FOR_EACH(list, item){
        edge = borMesh3EdgeFromVertexList(item);
        other_vert = borMesh3EdgeOtherVertex(edge, &sq->vert);
        n = bor_container_of(other_vert, node_t, vert);

        if (!g->params.unoptimized_err){
            nodeFixError(g, n);
        }

        if (n->err > max_err){
            max_err = n->err;
            max_n   = n;
        }
    }

    return max_n;
}

static node_t *createNewNode2(bor_gsrm_t *g, node_t *sq, node_t *sf)
{
    node_t *sr;
    bor_vec3_t v;

    borVec3Add2(&v, sq->v, sf->v);
    borVec3Scale(&v, BOR_REAL(0.5));

    sr = nodeNew(g, &v);

    return sr;
}

static void createNewNode(bor_gsrm_t *g)
{
    node_t *sq, *sf, *sr;
    bor_mesh3_edge_t *edge;


    // get node with highest error counter and its neighbor with highest
    // error counter
    sq = nodeWithHighestErrCounter(g);
    sf = nodesNeighborWithHighestErrCounter(g, sq);
    if (!sq || !sf){
        DBG("%lx %lx", (long)sq, (long)sf);
        DBG2("Can't create new node, because sq has no neighbors");
        return;
    }

    //DBG("sq: %lx, sf: %lx", (long)sq, (long)sf);

    // delete common edge of sq and sf
    edge = borMesh3VertexCommonEdge(&sq->vert, &sf->vert);
    if (edge){
        edgeDel(g, bor_container_of(edge, edge_t, edge));
    }

    // create new node
    sr = createNewNode2(g, sq, sf);

    // set up error counters of sq, sf and sr
    if (!g->params.unoptimized_err){
        nodeScaleError(g, sq, g->params.alpha);
        nodeScaleError(g, sf, g->params.alpha);
        sr->err  = sq->err + sf->err;
        sr->err /= BOR_REAL(2.);
        sr->err_cycle = g->cycle;
        borPairHeapUpdate(g->err_heap, &sr->err_heap);
    }else{
        sq->err *= g->params.alpha;
        sf->err *= g->params.alpha;
        sr->err  = sq->err + sf->err;
        sr->err /= BOR_REAL(2.);
    }

    // create edges sq-sr and sf-sr
    edgeNew(g, sq, sr);
    edgeNew(g, sf, sr);
}




/** --- Topology learning --- */
static void learnTopology(bor_gsrm_t *g)
{
    bor_vec_t *is;
    bor_pc_it_t pcit;
    bor_nn_el_t *el[2];

    // for each input point
    borPCItInit(&pcit, g->is);
    while (!borPCItEnd(&pcit)){
        is = borPCItGet(&pcit);

        // 1. Find two nearest nodes
        borNNNearest(g->nn, is, 2, el);
        g->c->nearest[0] = bor_container_of(el[0], node_t, nn);
        g->c->nearest[1] = bor_container_of(el[1], node_t, nn);

        // 2. Connect winning nodes
        echlConnectNodes(g);

        borPCItNext(&pcit);
    }
}

/** --- Postprocessing functions --- */
static void faceAreaStat(bor_gsrm_t *g, bor_real_t *_min, bor_real_t *_max,
                         bor_real_t *_avg)
{
    bor_real_t area, min, max, avg;
    bor_list_t *list, *item;
    bor_mesh3_face_t *face;

    max = avg = BOR_ZERO;
    min = BOR_REAL_MAX;
    list = borMesh3Faces(g->mesh);
    BOR_LIST_FOR_EACH(list, item){
        face = BOR_LIST_ENTRY(item, bor_mesh3_face_t, list);
        area = borMesh3FaceArea2(face);

        if (area < min){
            min = area;
        }

        if (area > max){
            max = area;
        }

        avg += area;
    }

    avg /= (bor_real_t)borMesh3FacesLen(g->mesh);

    *_min = min;
    *_max = max;
    *_avg = avg;
}

static void delIncorrectFaces(bor_gsrm_t *g)
{
    bor_mesh3_vertex_t *vs[4];
    bor_list_t *list, *item, *item_tmp;
    bor_mesh3_face_t *face, *faces[2];
    bor_mesh3_edge_t *edge;
    face_t *f, *fs[2];
    bor_real_t dangle;

    // iterate over all faces
    list = borMesh3Faces(g->mesh);
    BOR_LIST_FOR_EACH_SAFE(list, item, item_tmp){
        face = BOR_LIST_ENTRY(item, bor_mesh3_face_t, list);
        borMesh3FaceVertices(face, vs);

        // check internal angle of face
        if (!faceCheckAngle(g, vs[0], vs[1], vs[2])){
            f = bor_container_of(face, face_t, face);
            faceDel(g, f);
        }
    }

    // iterate over all edges
    list = borMesh3Edges(g->mesh);
    BOR_LIST_FOR_EACH_SAFE(list, item, item_tmp){
        edge = BOR_LIST_ENTRY(item, bor_mesh3_edge_t, list);

        if (borMesh3EdgeFacesLen(edge) == 2){
            // get incidenting faces
            faces[0] = borMesh3EdgeFace(edge, 0);
            faces[1] = borMesh3EdgeFace(edge, 1);

            // get end points of edge
            vs[0] = borMesh3EdgeVertex(edge, 0);
            vs[1] = borMesh3EdgeVertex(edge, 1);

            // get remaining two points defining the two faces
            vs[2] = borMesh3FaceOtherVertex(faces[0], vs[0], vs[1]);
            vs[3] = borMesh3FaceOtherVertex(faces[1], vs[0], vs[1]);

            // check dihedral angle between faces and if it is smaller than
            // treshold delete one of them
            dangle = borVec3DihedralAngle(vs[2]->v, vs[0]->v, vs[1]->v, vs[3]->v);
            if (dangle < g->params.min_dangle){
                fs[0] = bor_container_of(faces[0], face_t, face);
                fs[1] = bor_container_of(faces[1], face_t, face);
                delFacesDangle(g, fs[0], fs[1]);
            }
        }
    }
}

static void delIncorrectEdges(bor_gsrm_t *g)
{
    int madechange;
    bor_list_t *list, *item, *item_tmp;
    bor_mesh3_edge_t *edge;
    bor_mesh3_vertex_t *vs[2];
    edge_t *e;

    madechange = 1;
    while (madechange){
        madechange = 0;

        list = borMesh3Edges(g->mesh);
        BOR_LIST_FOR_EACH_SAFE(list, item, item_tmp){
            edge = BOR_LIST_ENTRY(item, bor_mesh3_edge_t, list);
            vs[0] = borMesh3EdgeVertex(edge, 0);
            vs[1] = borMesh3EdgeVertex(edge, 1);

            if (borMesh3VertexEdgesLen(vs[0]) == 1
                    || borMesh3VertexEdgesLen(vs[1]) == 1
                    || edgeNotUsable(edge)){
                e = bor_container_of(edge, edge_t, edge);
                edgeDel(g, e);
                madechange = 1;
            }
        }
    }
}

static void mergeEdges(bor_gsrm_t *g)
{
    int madechange;
    bor_list_t *list, *item, *item_tmp, *list2;
    bor_mesh3_vertex_t *vert;
    bor_mesh3_vertex_t *vs[2];
    bor_mesh3_edge_t *edge[2];
    bor_real_t angle;
    edge_t *e[2];
    node_t *n[2];

    madechange = 1;
    while (madechange){
        madechange = 0;

        list = borMesh3Vertices(g->mesh);
        BOR_LIST_FOR_EACH_SAFE(list, item, item_tmp){
            vert = BOR_LIST_ENTRY(item, bor_mesh3_vertex_t, list);
            if (borMesh3VertexEdgesLen(vert) == 2){
                // get incidenting edges
                list2 = borMesh3VertexEdges(vert);
                edge[0] = borMesh3EdgeFromVertexList(borListNext(list2));
                edge[1] = borMesh3EdgeFromVertexList(borListPrev(list2));

                // only edges that don't incident with any face can be
                // merged
                if (borMesh3EdgeFacesLen(edge[0]) == 0
                        && borMesh3EdgeFacesLen(edge[1]) == 0){
                    // get and points of edges
                    vs[0] = borMesh3EdgeOtherVertex(edge[0], vert);
                    vs[1] = borMesh3EdgeOtherVertex(edge[1], vert);

                    // compute angle between edges and check if it is big
                    // enough to perform merging
                    angle = borVec3Angle(vs[0]->v, vert->v, vs[1]->v);
                    if (angle > g->params.angle_merge_edges){
                        // finally, we can merge edges
                        e[0] = bor_container_of(edge[0], edge_t, edge);
                        e[1] = bor_container_of(edge[1], edge_t, edge);
                        n[0] = bor_container_of(vert, node_t, vert);

                        // first, remove edges
                        edgeDel(g, e[0]);
                        edgeDel(g, e[1]);

                        // then remove node
                        nodeDel(g, n[0]);

                        // and finally create new node
                        n[0] = bor_container_of(vs[0], node_t, vert);
                        n[1] = bor_container_of(vs[1], node_t, vert);
                        edgeNew(g, n[0], n[1]);
                        madechange = 1;
                    }
                }
            }
        }
    }
}

static void finishSurface(bor_gsrm_t *g)
{
    int madechange;
    bor_list_t *list, *item;
    bor_mesh3_edge_t *edge;
    edge_t *e;

    madechange = 1;
    while (madechange){
        madechange = 0;

        list = borMesh3Edges(g->mesh);
        BOR_LIST_FOR_EACH(list, item){
            edge = BOR_LIST_ENTRY(item, bor_mesh3_edge_t, list);

            // if it is border edge
            if (borMesh3EdgeFacesLen(edge) == 1){
                e = bor_container_of(edge, edge_t, edge);

                // try to finish triangle face
                if (finishSurfaceTriangle(g, e) == 0){
                    madechange = 1;

                    // try to create face incidenting with edge
                }else if (finishSurfaceNewFace(g, e) == 0){
                    madechange = 1;
                }
            }
        }
    }
}

static void delLonelyNodesEdgesFaces(bor_gsrm_t *g)
{
    bor_list_t *list, *item, *item_tmp;
    bor_mesh3_face_t *face;
    bor_mesh3_edge_t *edge;
    bor_mesh3_vertex_t *vert;
    bor_mesh3_edge_t *es[3];
    face_t *f;
    edge_t *e;
    node_t *n;

    list = borMesh3Faces(g->mesh);
    BOR_LIST_FOR_EACH_SAFE(list, item, item_tmp){
        face = BOR_LIST_ENTRY(item, bor_mesh3_face_t, list);
        es[0] = borMesh3FaceEdge(face, 0);
        es[1] = borMesh3FaceEdge(face, 1);
        es[2] = borMesh3FaceEdge(face, 2);

        if (borMesh3EdgeFacesLen(es[0]) == 1
                && borMesh3EdgeFacesLen(es[1]) == 1
                && borMesh3EdgeFacesLen(es[2]) == 1){
            f = bor_container_of(face, face_t, face);
            faceDel(g, f);
        }

    }


    list = borMesh3Edges(g->mesh);
    BOR_LIST_FOR_EACH_SAFE(list, item, item_tmp){
        edge = BOR_LIST_ENTRY(item, bor_mesh3_edge_t, list);
        if (borMesh3EdgeFacesLen(edge) == 0){
            e = bor_container_of(edge, edge_t, edge);
            edgeDel(g, e);
        }
    }


    list = borMesh3Vertices(g->mesh);
    BOR_LIST_FOR_EACH_SAFE(list, item, item_tmp){
        vert = BOR_LIST_ENTRY(item, bor_mesh3_vertex_t, list);
        if (borMesh3VertexEdgesLen(vert) == 0){
            n = bor_container_of(vert, node_t, vert);
            nodeDel(g, n);
        }
    }
}

static void finishSurfaceEmbedTriangles(bor_gsrm_t *g)
{
    bor_list_t *list, *item;
    bor_mesh3_edge_t *es[3];
    bor_mesh3_vertex_t *vs[3];
    edge_t *e;
    node_t *n[3];
    size_t i;

    list = borMesh3Edges(g->mesh);
    BOR_LIST_FOR_EACH(list, item){
        es[0] = BOR_LIST_ENTRY(item, bor_mesh3_edge_t, list);

        vs[0] = borMesh3EdgeVertex(es[0], 0);
        vs[1] = borMesh3EdgeVertex(es[0], 1);
        n[0]  = bor_container_of(vs[0], node_t, vert);
        n[1]  = bor_container_of(vs[1], node_t, vert);

        echlCommonNeighbors(g, n[0], n[1]);

        for (i = 0; i < g->c->common_neighb_len; i++){
            n[2] = g->c->common_neighb[i];

            es[1] = borMesh3VertexCommonEdge(vs[0], &n[2]->vert);
            if (!es[1] || borMesh3EdgeFacesLen(es[1]) != 1)
                continue;

            es[2] = borMesh3VertexCommonEdge(vs[1], &n[2]->vert);
            if (!es[2] || borMesh3EdgeFacesLen(es[2]) != 1)
                continue;

            e = bor_container_of(es[0], edge_t, edge);
            faceNew(g, e, n[2]);
            break;
        }
    }
}

static int faceCheckAngle(bor_gsrm_t *g, bor_mesh3_vertex_t *v1,
                          bor_mesh3_vertex_t *v2, bor_mesh3_vertex_t *v3)
{
    if (borVec3Angle(v1->v, v2->v, v3->v) > g->params.max_angle
            || borVec3Angle(v2->v, v3->v, v1->v) > g->params.max_angle
            || borVec3Angle(v3->v, v1->v, v2->v) > g->params.max_angle)
        return 0;
    return 1;
}

static void delFacesDangle(bor_gsrm_t *g, face_t *f1, face_t *f2)
{
    int f1_neighbors, f2_neighbors;
    size_t i;

    f1_neighbors = f2_neighbors = 0;
    for (i = 0; i < 3; i++){
        if (borMesh3EdgeFacesLen(borMesh3FaceEdge(&f1->face, i)) == 2)
            f1_neighbors++;
        if (borMesh3EdgeFacesLen(borMesh3FaceEdge(&f2->face, i)) == 2)
            f2_neighbors++;
    }

    if (f1_neighbors < f2_neighbors){
        faceDel(g, f1);
    }else if (f1_neighbors > f2_neighbors){
        faceDel(g, f2);
    }else{
        if (borMesh3FaceArea2(&f1->face) < borMesh3FaceArea2(&f2->face)){
            faceDel(g, f1);
        }else{
            faceDel(g, f2);
        }
    }
}

static int edgeNotUsable(bor_mesh3_edge_t *e)
{
    bor_list_t *list, *item;
    bor_mesh3_vertex_t *vs[2];
    bor_mesh3_edge_t *edge;
    size_t i, usable_edges;

    vs[0] = borMesh3EdgeVertex(e, 0);
    vs[1] = borMesh3EdgeVertex(e, 1);
    for (i = 0; i < 2; i++){
        usable_edges = 0;

        list = borMesh3VertexEdges(vs[0]);
        BOR_LIST_FOR_EACH(list, item){
            edge = borMesh3EdgeFromVertexList(item);
            if (borMesh3EdgeFacesLen(edge) < 2)
                usable_edges++;
        }

        if (usable_edges == 1)
            return 1;
    }

    return 0;
}

static int finishSurfaceTriangle(bor_gsrm_t *g, edge_t *e)
{
    size_t i;
    int ret;
    bor_mesh3_face_t *face;
    bor_mesh3_edge_t *es[2];
    bor_mesh3_vertex_t *vs[3];
    node_t *n[3];
    node_t *s;
    bor_real_t dangle;

    ret = -1;

    // get nodes of already existing face
    face = borMesh3EdgeFace(&e->edge, 0);
    vs[0] = borMesh3EdgeVertex(&e->edge, 0);
    vs[1] = borMesh3EdgeVertex(&e->edge, 1);
    vs[2] = borMesh3FaceOtherVertex(face, vs[0], vs[1]);

    n[0] = bor_container_of(vs[0], node_t, vert);
    n[1] = bor_container_of(vs[1], node_t, vert);
    n[2] = bor_container_of(vs[2], node_t, vert);
    echlCommonNeighbors(g, n[0], n[1]);

    // all common neighbors of n[0,1] form triangles
    for (i = 0; i < g->c->common_neighb_len; i++){
        if (borMesh3EdgeFacesLen(&e->edge) == 2)
            break;

        s = g->c->common_neighb[i];
        if (s != n[2]){
            // check angle
            if (!faceCheckAngle(g, vs[0], vs[1], &s->vert))
                continue;

            // check dihedral angle
            dangle = borVec3DihedralAngle(vs[2]->v, vs[0]->v, vs[1]->v, s->v);
            if (dangle < g->params.min_dangle)
                continue;

            // check if face can be created inside triplet of edges
            es[0] = borMesh3VertexCommonEdge(vs[0], &s->vert);
            es[1] = borMesh3VertexCommonEdge(vs[1], &s->vert);
            if ((es[0] && borMesh3EdgeFacesLen(es[0]) == 2)
                    || (es[1] && borMesh3EdgeFacesLen(es[1]) == 2))
                continue;

            // create edges if necessary
            if (!es[0]){
                edgeNew(g, n[0], s);
            }

            if (!es[1]){
                edgeNew(g, n[1], s);
            }

            faceNew(g, e, s);
            ret = 0;
        }
    }

    return ret;
}

static edge_t *finishSurfaceGetEdge(edge_t *e, node_t *n)
{
    bor_list_t *list, *item;
    bor_mesh3_edge_t *edge;
    edge_t *s, *s2;

    s2 = NULL;
    list = borMesh3VertexEdges(&n->vert);
    BOR_LIST_FOR_EACH(list, item){
        edge = borMesh3EdgeFromVertexList(item);

        if (borMesh3EdgeFacesLen(edge) == 0)
            return NULL;

        s = bor_container_of(edge, edge_t, edge);
        if (s != e && borMesh3EdgeFacesLen(edge) == 1){
            // there is more than two border edges incidenting with n
            if (s2 != NULL)
                return NULL;
            s2 = s;
        }
    }

    return s2;
}

static int finishSurfaceNewFace(bor_gsrm_t *g, edge_t *e)
{
    bor_mesh3_edge_t *edge;
    bor_mesh3_vertex_t *vs[2];
    node_t *ns[2], *ns2[2];
    edge_t *es[2];
    edge_t *e2, *e_new;

    // get start and end points
    vs[0] = borMesh3EdgeVertex(&e->edge, 0);
    vs[1] = borMesh3EdgeVertex(&e->edge, 1);
    ns[0] = bor_container_of(vs[0], node_t, vert);
    ns[1] = bor_container_of(vs[1], node_t, vert);

    es[0] = finishSurfaceGetEdge(e, ns[0]);
    es[1] = finishSurfaceGetEdge(e, ns[1]);

    if (es[0] != NULL && es[1] != NULL){
        // try to create both faces

        // obtain opossite nodes than wich already have from edge e
        vs[0] = borMesh3EdgeOtherVertex(&es[0]->edge, &ns[0]->vert);
        vs[1] = borMesh3EdgeOtherVertex(&es[1]->edge, &ns[1]->vert);
        ns2[0] = bor_container_of(vs[0], node_t, vert);
        ns2[1] = bor_container_of(vs[1], node_t, vert);

        e_new = NULL;
        edge = borMesh3VertexCommonEdge(&ns[0]->vert, &ns2[1]->vert);
        if (edge){
            e2 = bor_container_of(edge, edge_t, edge);
        }else{
            e_new = e2 = edgeNew(g, ns[0], ns2[1]);
        }
        if (finishSurfaceTriangle(g, e) != 0){
            if (e_new)
                edgeDel(g, e_new);
            return -1;
        }

        e_new = NULL;
        if (!borMesh3VertexCommonEdge(&ns2[0]->vert, &ns2[1]->vert)){
            e_new = edgeNew(g, ns2[0], ns2[1]);
        }

        if (finishSurfaceTriangle(g, e2) != 0){
            if (e_new)
                edgeDel(g, e_new);
            return -1;
        }

        return 0;
    }else if (es[0] != NULL){
        vs[0] = borMesh3EdgeOtherVertex(&es[0]->edge, &ns[0]->vert);
        ns2[0] = bor_container_of(vs[0], node_t, vert);

        e_new = NULL;
        edge = borMesh3VertexCommonEdge(&ns[1]->vert, &ns2[0]->vert);
        if (!edge){
            e_new = edgeNew(g, ns[1], ns2[0]);
        }
        if (finishSurfaceTriangle(g, e) != 0){
            if (e_new)
                edgeDel(g, e_new);
            return -1;
        }
        return 0;

    }else if (es[1] != NULL){
        vs[1] = borMesh3EdgeOtherVertex(&es[1]->edge, &ns[1]->vert);
        ns2[1] = bor_container_of(vs[1], node_t, vert);

        e_new = NULL;
        edge = borMesh3VertexCommonEdge(&ns[0]->vert, &ns2[1]->vert);
        if (!edge){
            e_new = edgeNew(g, ns[0], ns2[1]);
        }
        if (finishSurfaceTriangle(g, e) != 0){
            if (e_new)
                edgeDel(g, e_new);
            return -1;
        }
        return 0;
    }

    return -1;
}


static int errHeapLT(const bor_pairheap_node_t *_n1,
                     const bor_pairheap_node_t *_n2,
                     void *data)
{
    bor_gsrm_t *g = (bor_gsrm_t *)data;
    node_t *n1, *n2;

    n1 = bor_container_of(_n1, node_t, err_heap);
    n2 = bor_container_of(_n2, node_t, err_heap);

    nodeFixError(g, n1);
    nodeFixError(g, n2);
    return n1->err > n2->err;
}
