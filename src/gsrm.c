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

#include <fermat/gsrm.h>
#include <fermat/alloc.h>
#include <fermat/dbg.h>

/** Print progress */
#define PR_PROGRESS(g) \
    ferTimerStop(&g->timer); \
    ferTimerPrintElapsed(&(g)->timer, stderr, " n: %d/%d, e: %d, f: %d\r", \
                         ferMesh3VerticesLen((g)->mesh), \
                         (g)->params.max_nodes, \
                         ferMesh3EdgesLen((g)->mesh), \
                         ferMesh3FacesLen((g)->mesh)); \
    fflush(stderr)

#define PR_PROGRESS_PREFIX(g, prefix) \
    ferTimerStop(&(g)->timer); \
    ferTimerPrintElapsed(&(g)->timer, stderr, prefix " n: %d/%d, e: %d, f: %d\n", \
                         ferMesh3VerticesLen((g)->mesh), \
                         (g)->params.max_nodes, \
                         ferMesh3EdgesLen((g)->mesh), \
                         ferMesh3FacesLen((g)->mesh)); \
    fflush(stderr)


struct _node_t {
    fer_vec3_t *v; /*!< Position of node (weight vector) */

    fer_real_t err;               /*!< Error counter */
    unsigned long err_cycle;      /*!< Last cycle in which were .err changed */
    fer_pairheap_node_t err_heap; /*!< Connection into error heap */

    fer_mesh3_vertex_t vert; /*!< Vertex in mesh */
    fer_nn_el_t nn;          /*!< Struct for NN search */
};
typedef struct _node_t node_t;

struct _edge_t {
    int age;      /*!< Age of edge */

    fer_mesh3_edge_t edge; /*!< Edge in mesh */
};
typedef struct _edge_t edge_t;

struct _face_t {
    fer_mesh3_face_t face; /*!< Face in mesh */
};
typedef struct _face_t face_t;

struct _fer_gsrm_cache_t {
    fer_vec3_t *is;     /*!< Input signal */
    node_t *nearest[2]; /*!< Two nearest nodes */

    node_t **common_neighb;    /*!< Array of common neighbors */
    size_t common_neighb_size; /*!< Size of .common_neighb - num allocated
                                    bytes */
    size_t common_neighb_len;  /*!< Number of nodes in .common_neighb */

    size_t err_counter_mark;      /*!< Contains mark used for accumalet
                                       error counter. It holds how many
                                       times were applied parameter alpha */
    fer_real_t err_counter_scale; /*!< Accumulated error counter - alpha^mark */

    fer_real_t pp_min, pp_max; /*!< Min and max area2 of face - used in
                                    postprocessing */
};
typedef struct _fer_gsrm_cache_t fer_gsrm_cache_t;


/** Allocates and deallocates cache */
static fer_gsrm_cache_t *cacheNew(void);
static void cacheDel(fer_gsrm_cache_t *c);


/** --- Node functions --- */
/** Creates new node and sets its weight to given vector */
static node_t *nodeNew(fer_gsrm_t *g, const fer_vec3_t *v);
/** Deletes node */
static void nodeDel(fer_gsrm_t *g, node_t *n);
/** Deletes node - proposed for ferMesh3Del2() function */
static void nodeDel2(fer_mesh3_vertex_t *v, void *data);
/** Fixes node's error counter, i.e. applies correct beta^(n * lambda) */
_fer_inline void nodeFixError(fer_gsrm_t *gng, node_t *n);
/** Increment error counter */
_fer_inline void nodeIncError(fer_gsrm_t *gng, node_t *n, fer_real_t inc);
/** Scales error counter */
_fer_inline void nodeScaleError(fer_gsrm_t *gng, node_t *n, fer_real_t scale);


/** --- Edge functions --- */
/** Creates new edge as connection between two given nodes */
static edge_t *edgeNew(fer_gsrm_t *g, node_t *n1, node_t *n2);
/** Deletes edge */
static void edgeDel(fer_gsrm_t *g, edge_t *e);
/** Deteles edge - proposed for ferMesh3Del2() function */
static void edgeDel2(fer_mesh3_edge_t *v, void *data);

/** --- Face functions --- */
static face_t *faceNew(fer_gsrm_t *g, edge_t *e, node_t *n);
static void faceDel(fer_gsrm_t *g, face_t *e);
static void faceDel2(fer_mesh3_face_t *v, void *data);


static int init(fer_gsrm_t *g);
static void adapt(fer_gsrm_t *g);
static void newNode(fer_gsrm_t *g);

/* Initializes mesh with three random nodes from input */
static void meshInit(fer_gsrm_t *g);

static void drawInputPoint(fer_gsrm_t *g);
/** Performes Extended Competitive Hebbian Learning */
static void echl(fer_gsrm_t *g);
static void echlConnectNodes(fer_gsrm_t *g);
static void echlMove(fer_gsrm_t *g);
static void echlUpdate(fer_gsrm_t *g);
/** Creates new node */
static void createNewNode(fer_gsrm_t *g);

/** Initializes mesh with three random nodes from input */
static void meshInit(fer_gsrm_t *g);
/** Choose random input signal and stores it in cache */
static void drawInputPoint(fer_gsrm_t *g);


/** --- ECHL functions --- */
/** Performs ECHL algorithm */
static void echl(fer_gsrm_t *g);
/** Gathers common neighbors of n1 and n2 and stores them in cache. */
static void echlCommonNeighbors(fer_gsrm_t *g, node_t *n1, node_t *n2);
/** Remove edge if it is inside thales sphere of n1, n2 and theirs common
 *  neighbors */
static void echlRemoveThales(fer_gsrm_t *g, edge_t *e, node_t *n1, node_t *n2);
/** Removes all edges that connect common neighbors between each other */
static void echlRemoveNeighborsEdges(fer_gsrm_t *g);
/** Create faces between given edge and common neighbors stored in cache's
 *  .common_neighb array */
static void echlCreateFaces(fer_gsrm_t *g, edge_t *e);
/** Connect winner nodes and if they are already connected update that
 *  connection */
static void echlConnectNodes(fer_gsrm_t *g);
/** Moves node towards input signal by given factor */
_fer_inline void echlMoveNode(fer_gsrm_t *g, node_t *n, fer_real_t k);
/** Move winner nodes towards input signal */
static void echlMove(fer_gsrm_t *g);
/** Updates all edges emitating from winning node */
static void echlUpdate(fer_gsrm_t *g);


/** -- Create New Node functions --- */
/** Performs "Create New Node" operation */
static void createNewNode(fer_gsrm_t *g);
/** Returns node with highest error counter */
static node_t *nodeWithHighestErrCounter(fer_gsrm_t *g);
/** Returns node with highests error counter that is neighbor of sq */
static node_t *nodesNeighborWithHighestErrCounter(fer_gsrm_t *g, node_t *sq);
/** Actually creates new node between sq and sf */
static node_t *createNewNode2(fer_gsrm_t *g, node_t *sq, node_t *sf);


/** --- Topology learning --- */
static void learnTopology(fer_gsrm_t *g);


/** --- Postprocessing functions --- */
/** Returns (via min, max, avg arguments) minimum, maximum and average area
 *  of faces in a mesh. */
static void faceAreaStat(fer_gsrm_t *g, fer_real_t *min, fer_real_t *max,
                         fer_real_t *avg);
/** Deletes incorrect faces from mesh */
static void delIncorrectFaces(fer_gsrm_t *g);
/** Deletes incorrect edges from mesh */
static void delIncorrectEdges(fer_gsrm_t *g);
/** Merges all edges that can be merged */
static void mergeEdges(fer_gsrm_t *g);
/** Tries to finish (triangulate) surface */
static void finishSurface(fer_gsrm_t *g);
/** Deletes lonely nodes, edges and faces */
static void delLonelyNodesEdgesFaces(fer_gsrm_t *g);
/** Embed triangles everywhere it can */
static void finishSurfaceEmbedTriangles(fer_gsrm_t *g);
/** Returns true if all internal angles of face is smaller than
 *  g->params.max_angle */
static int faceCheckAngle(fer_gsrm_t *g, fer_mesh3_vertex_t *v1,
                          fer_mesh3_vertex_t *v2, fer_mesh3_vertex_t *v3);
/** Deletes one of triangles (the triangles are considered to have dihedral
 *  angle smaller than g->params.min_dangle.
 *  First is deleted face that have less incidenting faces. If both have
 *  same, faces with smaller area is deleted. */
static void delFacesDangle(fer_gsrm_t *g, face_t *f1, face_t *f2);
/** Returns true if given edge can't be used for face creation */
static int edgeNotUsable(fer_mesh3_edge_t *e);
/** Tries to finish triangle incidenting with e.
 *  It's assumend that e has already one incidenting face. */
static int finishSurfaceTriangle(fer_gsrm_t *g, edge_t *e);
/** Tries to create completely new face */
static int finishSurfaceNewFace(fer_gsrm_t *g, edge_t *e);

static int errHeapLT(const fer_pairheap_node_t *_n1,
                     const fer_pairheap_node_t *_n2,
                     void *data);

void ferGSRMParamsInit(fer_gsrm_params_t *params)
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

    ferNNParamsInit(&params->nn);
    params->nn.type = FER_NN_GUG;
    params->nn.gug.dim = 3;
    params->nn.vptree.dim = 3;
    params->nn.linear.dim = 3;

    params->unoptimized_err = 0;
}

fer_gsrm_t *ferGSRMNew(const fer_gsrm_params_t *params)
{
    fer_gsrm_t *g;

    g = FER_ALLOC(fer_gsrm_t);
    g->params = *params;

    // initialize point cloude (input signals)
    g->is = ferPCNew(3);

    // init 3D mesh
    g->mesh = ferMesh3New();

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

void ferGSRMDel(fer_gsrm_t *g)
{
    if (g->c)
        cacheDel(g->c);

    if (g->is)
        ferPCDel(g->is);

    if (g->mesh)
        ferMesh3Del2(g->mesh, nodeDel2, (void *)g,
                              edgeDel2, (void *)g,
                              faceDel2, (void *)g);

    if (g->nn)
        ferNNDel(g->nn);

    if (g->beta_n)
        FER_FREE(g->beta_n);
    if (g->beta_lambda_n)
        FER_FREE(g->beta_lambda_n);
    if (g->err_heap)
        ferPairHeapDel(g->err_heap);

    FER_FREE(g);
}

size_t ferGSRMAddInputSignals(fer_gsrm_t *g, const char *fn)
{
    return ferPCAddFromFile(g->is, fn);
}

int ferGSRMRun(fer_gsrm_t *g)
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
                && fer_unlikely(cycle == FER_GSRM_PROGRESS_REFRESH)){
            PR_PROGRESS(g);
            cycle = 0;
        }

        g->cycle++;
    } while (ferMesh3VerticesLen(g->mesh) < g->params.max_nodes);

    if (g->params.verbosity >= 1){
        PR_PROGRESS(g);
        fprintf(stderr, "\n");
        fflush(stderr);
    }

    learnTopology(g);

    return 0;
}


int ferGSRMPostprocess(fer_gsrm_t *g)
{
    fer_real_t min, max, avg;

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


static int init(fer_gsrm_t *g)
{
    size_t i;
    fer_real_t maxbeta;
    fer_real_t aabb[6];

    // check if there are some input signals
    if (ferPCLen(g->is) <= 3){
        DBG2("No input signals!");
        return -1;
    }

    g->cycle = 1L;

    // initialize error heap
    if (!g->params.unoptimized_err){
        if (g->err_heap)
            ferPairHeapDel(g->err_heap);
        g->err_heap = ferPairHeapNew(errHeapLT, (void *)g);
    }

    // precompute beta^n
    if (g->beta_n)
        FER_FREE(g->beta_n);

    g->beta_n = FER_ALLOC_ARR(fer_real_t, g->params.lambda);
    g->beta_n[0] = g->params.beta;
    for (i = 1; i < g->params.lambda; i++){
        g->beta_n[i] = g->beta_n[i - 1] * g->params.beta;
    }

    // precompute beta^(n * lambda)
    if (g->beta_lambda_n)
        FER_FREE(g->beta_lambda_n);

    maxbeta = g->beta_n[g->params.lambda - 1];

    g->beta_lambda_n_len = 1000;
    g->beta_lambda_n = FER_ALLOC_ARR(fer_real_t, g->beta_lambda_n_len);
    g->beta_lambda_n[0] = maxbeta;
    for (i = 1; i < g->beta_lambda_n_len; i++){
        g->beta_lambda_n[i] = g->beta_lambda_n[i - 1] * maxbeta;
    }

    // initialize cache
    if (!g->c)
        g->c = cacheNew();

    // initialize NN search structure
    if (g->nn)
        ferNNDel(g->nn);
    ferPCAABB(g->is, aabb);
    g->params.nn.linear.dim = 3;
    g->params.nn.vptree.dim = 3;
    g->params.nn.gug.dim    = 3;
    g->params.nn.gug.aabb   = aabb;
    g->nn = ferNNNew(&g->params.nn);

    // first shuffle of all input signals
    ferPCPermutate(g->is);
    // and initialize its iterator
    ferPCItInit(&g->isit, g->is);


    // start timer
    ferTimerStart(&g->timer);

    // initialize mesh with three random nodes
    meshInit(g);

    return 0;

}

static void adapt(fer_gsrm_t *g)
{
    drawInputPoint(g);
    echl(g);
}

static void newNode(fer_gsrm_t *g)
{
    createNewNode(g);
}

static fer_gsrm_cache_t *cacheNew(void)
{
    fer_gsrm_cache_t *c;

    c = FER_ALLOC(fer_gsrm_cache_t);
    c->nearest[0] = c->nearest[1] = NULL;

    c->common_neighb_size = 3;
    c->common_neighb = FER_ALLOC_ARR(node_t *, c->common_neighb_size);
    c->common_neighb_len = 0;

    c->err_counter_mark = 0;
    c->err_counter_scale = FER_ONE;

    return c;
}

static void cacheDel(fer_gsrm_cache_t *c)
{
    FER_FREE(c->common_neighb);
    FER_FREE(c);
}



/** --- Node functions --- **/
static node_t *nodeNew(fer_gsrm_t *g, const fer_vec3_t *v)
{
    node_t *n;

    n = FER_ALLOC(node_t);
    n->v = ferVec3Clone(v);

    // initialize mesh's vertex struct with weight vector
    ferMesh3VertexSetCoords(&n->vert, n->v);

    // initialize cells struct with its own weight vector
    ferNNElInit(g->nn, &n->nn, (fer_vec_t *)n->v);

    // add node into mesh
    ferMesh3AddVertex(g->mesh, &n->vert);
    // and add node into cells
    ferNNAdd(g->nn, &n->nn);

    // set error counter
    n->err = FER_ZERO;
    if (!g->params.unoptimized_err){
        n->err_cycle = g->cycle;
        ferPairHeapAdd(g->err_heap, &n->err_heap);
    }

    //DBG("n: %lx, vert: %lx (%g %g %g)", (long)n, (long)&n->vert,
    //    ferVec3X(&n->v), ferVec3Y(&n->v), ferVec3Z(&n->v));

    return n;
}

_fer_inline void nodeFixError(fer_gsrm_t *g, node_t *n)
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

_fer_inline void nodeIncError(fer_gsrm_t *g, node_t *n, fer_real_t inc)
{
    nodeFixError(g, n);
    n->err += inc;
    ferPairHeapUpdate(g->err_heap, &n->err_heap);
}

_fer_inline void nodeScaleError(fer_gsrm_t *g, node_t *n, fer_real_t scale)
{
    nodeFixError(g, n);
    n->err *= scale;
    ferPairHeapUpdate(g->err_heap, &n->err_heap);
}

static void nodeDel(fer_gsrm_t *g, node_t *n)
{
    fer_list_t *list, *item, *item_tmp;
    fer_mesh3_edge_t *edge;
    edge_t *e;
    int res;

    // remove node from mesh
    if (fer_unlikely(ferMesh3VertexEdgesLen(&n->vert) > 0)){
        // remove edges first
        list = ferMesh3VertexEdges(&n->vert);
        FER_LIST_FOR_EACH_SAFE(list, item, item_tmp){
            edge = ferMesh3EdgeFromVertexList(item);
            e = fer_container_of(edge, edge_t, edge);

            edgeDel(g, e);
        }
    }

    // then vertex
    res = ferMesh3RemoveVertex(g->mesh, &n->vert);
    if (fer_unlikely(res != 0)){
        DBG2("Node couldn't be removed from mesh - this shouldn't happen!");
        exit(-1);
    }

    ferVec3Del(n->v);

    // remove node from cells
    ferNNRemove(g->nn, &n->nn);

    // remove from error heap
    if (!g->params.unoptimized_err){
        ferPairHeapRemove(g->err_heap, &n->err_heap);
    }

    // Note: no need of deallocation of .vert and .cells
    FER_FREE(n);
}

static void nodeDel2(fer_mesh3_vertex_t *v, void *data)
{
    fer_gsrm_t *g = (fer_gsrm_t *)data;
    node_t *n;
    n = fer_container_of(v, node_t, vert);

    ferVec3Del(n->v);

    // remove node from cells
    ferNNRemove(g->nn, &n->nn);

    FER_FREE(n);
}



/** --- Edge functions --- **/
static edge_t *edgeNew(fer_gsrm_t *g, node_t *n1, node_t *n2)
{
    edge_t *e;

    e = FER_ALLOC(edge_t);
    e->age = 0;

    ferMesh3AddEdge(g->mesh, &e->edge, &n1->vert, &n2->vert);

    //DBG("e: %lx, edge: %lx", (long)e, (long)&e->edge);

    return e;
}

static void edgeDel(fer_gsrm_t *g, edge_t *e)
{
    fer_mesh3_face_t *face;
    int res;

    // first remove incidenting faces
    while ((face = ferMesh3EdgeFace(&e->edge, 0)) != NULL){
        faceDel(g, fer_container_of(face, face_t, face));
    }

    // then remove edge itself
    res = ferMesh3RemoveEdge(g->mesh, &e->edge);
    if (fer_unlikely(res != 0)){
        DBG2("Can't remove edge - this shouldn'h happen!");
        exit(-1);
    }

    FER_FREE(e);
}

static void edgeDel2(fer_mesh3_edge_t *edge, void *data)
{
    edge_t *e;
    e = fer_container_of(edge, edge_t, edge);
    FER_FREE(e);
}




/** --- Face functions --- **/
static face_t *faceNew(fer_gsrm_t *g, edge_t *e, node_t *n)
{
    face_t *f;
    fer_mesh3_edge_t *e2, *e3;
    int res;

    e2 = ferMesh3VertexCommonEdge(ferMesh3EdgeVertex(&e->edge, 0), &n->vert);
    e3 = ferMesh3VertexCommonEdge(ferMesh3EdgeVertex(&e->edge, 1), &n->vert);
    if (fer_unlikely(!e2 || !e3)){
        DBG2("Can't create face because *the* three nodes are not connected "
             " - this shouldn't happen!");
        return NULL;
    }

    f = FER_ALLOC(face_t);

    res = ferMesh3AddFace(g->mesh, &f->face, &e->edge, e2, e3);
    if (fer_unlikely(res != 0)){
        FER_FREE(f);
        return NULL;
    }

    // TODO: check if face already exists

    return f;
}

static void faceDel(fer_gsrm_t *g, face_t *f)
{
    ferMesh3RemoveFace(g->mesh, &f->face);
    FER_FREE(f);
}

static void faceDel2(fer_mesh3_face_t *face, void *data)
{
    face_t *f;
    f = fer_container_of(face, face_t, face);
    FER_FREE(f);
}




static void meshInit(fer_gsrm_t *g)
{
    fer_vec3_t *v;
    size_t i;

    for (i = 0; i < 3; i++){
        // obtain input signal
        v = (fer_vec3_t *)ferPCItGet(&g->isit);

        // create new node
        nodeNew(g, v);

        // move to next point
        ferPCItNext(&g->isit);
    }
}

static void drawInputPoint(fer_gsrm_t *g)
{
    if (ferPCItEnd(&g->isit)){
        // if iterator is at the end permutate point cloud again
        ferPCPermutate(g->is);
        // and re-initialize iterator
        ferPCItInit(&g->isit, g->is);
    }
    g->c->is = (fer_vec3_t *)ferPCItGet(&g->isit);
    ferPCItNext(&g->isit);
}



/** --- ECHL functions --- **/
static void decreaseAllErrors(fer_gsrm_t *g)
{
    fer_list_t *list, *item;
    fer_mesh3_vertex_t *v;
    node_t *n;

    list = ferMesh3Vertices(g->mesh);
    FER_LIST_FOR_EACH(list, item){
        v = FER_LIST_ENTRY(item, fer_mesh3_vertex_t, list);
        n = fer_container_of(v, node_t, vert);
        n->err = n->err * g->params.beta;
    }
}


static void echl(fer_gsrm_t *g)
{
    fer_nn_el_t *el[2];

    // 1. Find two nearest nodes
    ferNNNearest(g->nn, (const fer_vec_t *)g->c->is, 2, el);
    g->c->nearest[0] = fer_container_of(el[0], node_t, nn);
    g->c->nearest[1] = fer_container_of(el[1], node_t, nn);

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

static void echlCommonNeighbors(fer_gsrm_t *g, node_t *n1, node_t *n2)
{
    fer_list_t *list1, *list2;
    fer_list_t *item1, *item2;
    fer_mesh3_edge_t *edge1, *edge2;
    fer_mesh3_vertex_t *o1, *o2;
    node_t *n;
    size_t len;

    // allocate enough memory for common neighbors
    if (g->c->common_neighb_size < ferMesh3VertexEdgesLen(&n1->vert)
            && g->c->common_neighb_size < ferMesh3VertexEdgesLen(&n2->vert)){
        len = ferMesh3VertexEdgesLen(&n1->vert);
        len = FER_MIN(len, ferMesh3VertexEdgesLen(&n2->vert));

        g->c->common_neighb = FER_REALLOC_ARR(g->c->common_neighb, node_t *, len);
        g->c->common_neighb_size = len;
    }

    list1 = ferMesh3VertexEdges(&n1->vert);
    list2 = ferMesh3VertexEdges(&n2->vert);
    len = 0;
    FER_LIST_FOR_EACH(list1, item1){
        edge1 = ferMesh3EdgeFromVertexList(item1);
        o1 = ferMesh3EdgeVertex(edge1, 0);
        if (o1 == &n1->vert)
            o1 = ferMesh3EdgeVertex(edge1, 1);

        FER_LIST_FOR_EACH(list2, item2){
            edge2 = ferMesh3EdgeFromVertexList(item2);
            o2 = ferMesh3EdgeVertex(edge2, 0);
            if (o2 == &n2->vert)
                o2 = ferMesh3EdgeVertex(edge2, 1);

            if (o1 == o2){
                n = fer_container_of(o1, node_t, vert);
                g->c->common_neighb[len] = n;
                len++;
            }
        }
    }

    g->c->common_neighb_len = len;
}

static void echlRemoveThales(fer_gsrm_t *g, edge_t *e, node_t *n1, node_t *n2)
{
    node_t *nb;
    size_t i, len;

    len = g->c->common_neighb_len;
    for (i=0; i < len; i++){
        nb = g->c->common_neighb[i];

        if (ferVec3Angle(n1->v, nb->v, n2->v) > M_PI_2){
            // remove edge
            edgeDel(g, e);
            return;
        }
    }
}

static void echlRemoveNeighborsEdges(fer_gsrm_t *g)
{
    size_t i, j, len;
    fer_mesh3_edge_t *edge;
    node_t **ns;
    edge_t *e;

    ns = g->c->common_neighb;
    len = g->c->common_neighb_len;
    if (len == 0)
        return;

    for (i = 0; i < len; i++){
        for (j = i + 1; j < len; j++){
            edge = ferMesh3VertexCommonEdge(&ns[i]->vert, &ns[j]->vert);
            if (edge != NULL){
                e = fer_container_of(edge, edge_t, edge);
                edgeDel(g, e);
            }
        }
    }
}

static void echlCreateFaces(fer_gsrm_t *g, edge_t *e)
{
    size_t i, len;
    node_t **ns;

    len = FER_MIN(g->c->common_neighb_len, 2);
    ns = g->c->common_neighb;
    for (i = 0; i < len; i++){
        faceNew(g, e, ns[i]);
    }
}

static void echlConnectNodes(fer_gsrm_t *g)
{
    fer_mesh3_edge_t *edge;
    edge_t *e;
    node_t *n1, *n2;

    n1 = g->c->nearest[0];
    n2 = g->c->nearest[1];

    // get edge connecting n1 and n2
    e = NULL;
    edge = ferMesh3VertexCommonEdge(&n1->vert, &n2->vert);
    if (edge){
        e = fer_container_of(edge, edge_t, edge);
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

_fer_inline void echlMoveNode(fer_gsrm_t *g, node_t *n, fer_real_t k)
{
    fer_vec3_t v;

    // compute shifting
    ferVec3Sub2(&v, g->c->is, n->v);
    ferVec3Scale(&v, k);

    // move node
    ferVec3Add(n->v, &v);

    // update node in search structure
    ferNNUpdate(g->nn, &n->nn);
}

static void echlMove(fer_gsrm_t *g)
{
    fer_list_t *list, *item;
    fer_mesh3_edge_t *edge;
    fer_mesh3_vertex_t *wvert, *vert;
    node_t *wn;
    fer_real_t err;

    wn = g->c->nearest[0];
    wvert = &wn->vert;

    // move winning node
    echlMoveNode(g, wn, g->params.eb);

    // increase error counter
    if (!g->params.unoptimized_err){
        err  = ferVec3Dist2(wn->v, g->c->is);
        err *= g->beta_n[g->params.lambda - g->step];
        nodeIncError(g, wn, err);
    }else{
        wn->err += ferVec3Dist2(wn->v, g->c->is);
    }

    // move nodes connected with the winner
    list = ferMesh3VertexEdges(wvert);
    FER_LIST_FOR_EACH(list, item){
        edge = ferMesh3EdgeFromVertexList(item);
        vert = ferMesh3EdgeOtherVertex(edge, wvert);

        echlMoveNode(g, fer_container_of(vert, node_t, vert), g->params.en);
    }
}

static void echlUpdate(fer_gsrm_t *g)
{
    node_t *wn;
    edge_t *e;
    fer_list_t *list, *item, *tmp_item;
    fer_mesh3_edge_t *edge;
    fer_mesh3_vertex_t *vert;

    wn = g->c->nearest[0];

    list = ferMesh3VertexEdges(&wn->vert);
    FER_LIST_FOR_EACH_SAFE(list, item, tmp_item){
        edge = ferMesh3EdgeFromVertexList(item);
        e = fer_container_of(edge, edge_t, edge);

        // increment age of edge
        e->age++;

        // if age of edge is above treshold remove edge and nodes which
        // remain unconnected
        if (e->age > g->params.age_max){
            // get other node than winning one
            vert = ferMesh3EdgeOtherVertex(edge, &wn->vert);

            // delete edge
            edgeDel(g, e);

            // check if n is connected in mesh, if not delete it
            if (ferMesh3VertexEdgesLen(vert) == 0){
                nodeDel(g, fer_container_of(vert, node_t, vert));
            }
        }
    }

    // check if winning node remains connected
    if (ferMesh3VertexEdgesLen(&wn->vert) == 0){
        nodeDel(g, wn);
    }
}




/** --- Create New Node functions --- **/
static node_t *nodeWithHighestErrCounterLinear(fer_gsrm_t *g)
{
    fer_list_t *list, *item;
    fer_mesh3_vertex_t *v;
    node_t *n;
    fer_real_t err;
    node_t *max;

    max = NULL;
    err = -FER_REAL_MAX;

    list = ferMesh3Vertices(g->mesh);
    FER_LIST_FOR_EACH(list, item){
        v = FER_LIST_ENTRY(item, fer_mesh3_vertex_t, list);
        n = fer_container_of(v, node_t, vert);

        if (n->err > err){
            max = n;
            err = n->err;
        }
    }

    return max;
}

static node_t *nodeWithHighestErrCounter(fer_gsrm_t *g)
{
    fer_pairheap_node_t *max;
    node_t *maxn;

    if (g->params.unoptimized_err){
        return nodeWithHighestErrCounterLinear(g);
    }

    max  = ferPairHeapMin(g->err_heap);
    maxn = fer_container_of(max, node_t, err_heap);

    return maxn;
}

static node_t *nodesNeighborWithHighestErrCounter(fer_gsrm_t *g, node_t *sq)
{
    fer_list_t *list, *item;
    fer_mesh3_edge_t *edge;
    fer_mesh3_vertex_t *other_vert;
    fer_real_t max_err;
    node_t *n, *max_n;


    max_err = -FER_REAL_MAX;
    max_n   = NULL;
    list = ferMesh3VertexEdges(&sq->vert);
    FER_LIST_FOR_EACH(list, item){
        edge = ferMesh3EdgeFromVertexList(item);
        other_vert = ferMesh3EdgeOtherVertex(edge, &sq->vert);
        n = fer_container_of(other_vert, node_t, vert);

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

static node_t *createNewNode2(fer_gsrm_t *g, node_t *sq, node_t *sf)
{
    node_t *sr;
    fer_vec3_t v;

    ferVec3Add2(&v, sq->v, sf->v);
    ferVec3Scale(&v, FER_REAL(0.5));

    sr = nodeNew(g, &v);

    return sr;
}

static void createNewNode(fer_gsrm_t *g)
{
    node_t *sq, *sf, *sr;
    fer_mesh3_edge_t *edge;


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
    edge = ferMesh3VertexCommonEdge(&sq->vert, &sf->vert);
    if (edge){
        edgeDel(g, fer_container_of(edge, edge_t, edge));
    }

    // create new node
    sr = createNewNode2(g, sq, sf);

    // set up error counters of sq, sf and sr
    if (!g->params.unoptimized_err){
        nodeScaleError(g, sq, g->params.alpha);
        nodeScaleError(g, sf, g->params.alpha);
        sr->err  = sq->err + sf->err;
        sr->err /= FER_REAL(2.);
        sr->err_cycle = g->cycle;
        ferPairHeapUpdate(g->err_heap, &sr->err_heap);
    }else{
        sq->err *= g->params.alpha;
        sf->err *= g->params.alpha;
        sr->err  = sq->err + sf->err;
        sr->err /= FER_REAL(2.);
    }

    // create edges sq-sr and sf-sr
    edgeNew(g, sq, sr);
    edgeNew(g, sf, sr);
}




/** --- Topology learning --- */
static void learnTopology(fer_gsrm_t *g)
{
    fer_vec_t *is;
    fer_pc_it_t pcit;
    fer_nn_el_t *el[2];

    // for each input point
    ferPCItInit(&pcit, g->is);
    while (!ferPCItEnd(&pcit)){
        is = ferPCItGet(&pcit);

        // 1. Find two nearest nodes
        ferNNNearest(g->nn, is, 2, el);
        g->c->nearest[0] = fer_container_of(el[0], node_t, nn);
        g->c->nearest[1] = fer_container_of(el[1], node_t, nn);

        // 2. Connect winning nodes
        echlConnectNodes(g);

        ferPCItNext(&pcit);
    }
}

/** --- Postprocessing functions --- */
static void faceAreaStat(fer_gsrm_t *g, fer_real_t *_min, fer_real_t *_max,
                         fer_real_t *_avg)
{
    fer_real_t area, min, max, avg;
    fer_list_t *list, *item;
    fer_mesh3_face_t *face;

    max = avg = FER_ZERO;
    min = FER_REAL_MAX;
    list = ferMesh3Faces(g->mesh);
    FER_LIST_FOR_EACH(list, item){
        face = FER_LIST_ENTRY(item, fer_mesh3_face_t, list);
        area = ferMesh3FaceArea2(face);

        if (area < min){
            min = area;
        }

        if (area > max){
            max = area;
        }

        avg += area;
    }

    avg /= (fer_real_t)ferMesh3FacesLen(g->mesh);

    *_min = min;
    *_max = max;
    *_avg = avg;
}

static void delIncorrectFaces(fer_gsrm_t *g)
{
    fer_mesh3_vertex_t *vs[4];
    fer_list_t *list, *item, *item_tmp;
    fer_mesh3_face_t *face, *faces[2];
    fer_mesh3_edge_t *edge;
    face_t *f, *fs[2];
    fer_real_t dangle;

    // iterate over all faces
    list = ferMesh3Faces(g->mesh);
    FER_LIST_FOR_EACH_SAFE(list, item, item_tmp){
        face = FER_LIST_ENTRY(item, fer_mesh3_face_t, list);
        ferMesh3FaceVertices(face, vs);

        // check internal angle of face
        if (!faceCheckAngle(g, vs[0], vs[1], vs[2])){
            f = fer_container_of(face, face_t, face);
            faceDel(g, f);
        }
    }

    // iterate over all edges
    list = ferMesh3Edges(g->mesh);
    FER_LIST_FOR_EACH_SAFE(list, item, item_tmp){
        edge = FER_LIST_ENTRY(item, fer_mesh3_edge_t, list);

        if (ferMesh3EdgeFacesLen(edge) == 2){
            // get incidenting faces
            faces[0] = ferMesh3EdgeFace(edge, 0);
            faces[1] = ferMesh3EdgeFace(edge, 1);

            // get end points of edge
            vs[0] = ferMesh3EdgeVertex(edge, 0);
            vs[1] = ferMesh3EdgeVertex(edge, 1);

            // get remaining two points defining the two faces
            vs[2] = ferMesh3FaceOtherVertex(faces[0], vs[0], vs[1]);
            vs[3] = ferMesh3FaceOtherVertex(faces[1], vs[0], vs[1]);

            // check dihedral angle between faces and if it is smaller than
            // treshold delete one of them
            dangle = ferVec3DihedralAngle(vs[2]->v, vs[0]->v, vs[1]->v, vs[3]->v);
            if (dangle < g->params.min_dangle){
                fs[0] = fer_container_of(faces[0], face_t, face);
                fs[1] = fer_container_of(faces[1], face_t, face);
                delFacesDangle(g, fs[0], fs[1]);
            }
        }
    }
}

static void delIncorrectEdges(fer_gsrm_t *g)
{
    int madechange;
    fer_list_t *list, *item, *item_tmp;
    fer_mesh3_edge_t *edge;
    fer_mesh3_vertex_t *vs[2];
    edge_t *e;

    madechange = 1;
    while (madechange){
        madechange = 0;

        list = ferMesh3Edges(g->mesh);
        FER_LIST_FOR_EACH_SAFE(list, item, item_tmp){
            edge = FER_LIST_ENTRY(item, fer_mesh3_edge_t, list);
            vs[0] = ferMesh3EdgeVertex(edge, 0);
            vs[1] = ferMesh3EdgeVertex(edge, 1);

            if (ferMesh3VertexEdgesLen(vs[0]) == 1
                    || ferMesh3VertexEdgesLen(vs[1]) == 1
                    || edgeNotUsable(edge)){
                e = fer_container_of(edge, edge_t, edge);
                edgeDel(g, e);
                madechange = 1;
            }
        }
    }
}

static void mergeEdges(fer_gsrm_t *g)
{
    int madechange;
    fer_list_t *list, *item, *item_tmp, *list2;
    fer_mesh3_vertex_t *vert;
    fer_mesh3_vertex_t *vs[2];
    fer_mesh3_edge_t *edge[2];
    fer_real_t angle;
    edge_t *e[2];
    node_t *n[2];

    madechange = 1;
    while (madechange){
        madechange = 0;

        list = ferMesh3Vertices(g->mesh);
        FER_LIST_FOR_EACH_SAFE(list, item, item_tmp){
            vert = FER_LIST_ENTRY(item, fer_mesh3_vertex_t, list);
            if (ferMesh3VertexEdgesLen(vert) == 2){
                // get incidenting edges
                list2 = ferMesh3VertexEdges(vert);
                edge[0] = ferMesh3EdgeFromVertexList(ferListNext(list2));
                edge[1] = ferMesh3EdgeFromVertexList(ferListPrev(list2));

                // only edges that don't incident with any face can be
                // merged
                if (ferMesh3EdgeFacesLen(edge[0]) == 0
                        && ferMesh3EdgeFacesLen(edge[1]) == 0){
                    // get and points of edges
                    vs[0] = ferMesh3EdgeOtherVertex(edge[0], vert);
                    vs[1] = ferMesh3EdgeOtherVertex(edge[1], vert);

                    // compute angle between edges and check if it is big
                    // enough to perform merging
                    angle = ferVec3Angle(vs[0]->v, vert->v, vs[1]->v);
                    if (angle > g->params.angle_merge_edges){
                        // finally, we can merge edges
                        e[0] = fer_container_of(edge[0], edge_t, edge);
                        e[1] = fer_container_of(edge[1], edge_t, edge);
                        n[0] = fer_container_of(vert, node_t, vert);

                        // first, remove edges
                        edgeDel(g, e[0]);
                        edgeDel(g, e[1]);

                        // then remove node
                        nodeDel(g, n[0]);

                        // and finally create new node
                        n[0] = fer_container_of(vs[0], node_t, vert);
                        n[1] = fer_container_of(vs[1], node_t, vert);
                        edgeNew(g, n[0], n[1]);
                        madechange = 1;
                    }
                }
            }
        }
    }
}

static void finishSurface(fer_gsrm_t *g)
{
    int madechange;
    fer_list_t *list, *item;
    fer_mesh3_edge_t *edge;
    edge_t *e;

    madechange = 1;
    while (madechange){
        madechange = 0;

        list = ferMesh3Edges(g->mesh);
        FER_LIST_FOR_EACH(list, item){
            edge = FER_LIST_ENTRY(item, fer_mesh3_edge_t, list);

            // if it is border edge
            if (ferMesh3EdgeFacesLen(edge) == 1){
                e = fer_container_of(edge, edge_t, edge);

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

static void delLonelyNodesEdgesFaces(fer_gsrm_t *g)
{
    fer_list_t *list, *item, *item_tmp;
    fer_mesh3_face_t *face;
    fer_mesh3_edge_t *edge;
    fer_mesh3_vertex_t *vert;
    fer_mesh3_edge_t *es[3];
    face_t *f;
    edge_t *e;
    node_t *n;

    list = ferMesh3Faces(g->mesh);
    FER_LIST_FOR_EACH_SAFE(list, item, item_tmp){
        face = FER_LIST_ENTRY(item, fer_mesh3_face_t, list);
        es[0] = ferMesh3FaceEdge(face, 0);
        es[1] = ferMesh3FaceEdge(face, 1);
        es[2] = ferMesh3FaceEdge(face, 2);

        if (ferMesh3EdgeFacesLen(es[0]) == 1
                && ferMesh3EdgeFacesLen(es[1]) == 1
                && ferMesh3EdgeFacesLen(es[2]) == 1){
            f = fer_container_of(face, face_t, face);
            faceDel(g, f);
        }

    }


    list = ferMesh3Edges(g->mesh);
    FER_LIST_FOR_EACH_SAFE(list, item, item_tmp){
        edge = FER_LIST_ENTRY(item, fer_mesh3_edge_t, list);
        if (ferMesh3EdgeFacesLen(edge) == 0){
            e = fer_container_of(edge, edge_t, edge);
            edgeDel(g, e);
        }
    }


    list = ferMesh3Vertices(g->mesh);
    FER_LIST_FOR_EACH_SAFE(list, item, item_tmp){
        vert = FER_LIST_ENTRY(item, fer_mesh3_vertex_t, list);
        if (ferMesh3VertexEdgesLen(vert) == 0){
            n = fer_container_of(vert, node_t, vert);
            nodeDel(g, n);
        }
    }
}

static void finishSurfaceEmbedTriangles(fer_gsrm_t *g)
{
    fer_list_t *list, *item;
    fer_mesh3_edge_t *es[3];
    fer_mesh3_vertex_t *vs[3];
    edge_t *e;
    node_t *n[3];
    size_t i;

    list = ferMesh3Edges(g->mesh);
    FER_LIST_FOR_EACH(list, item){
        es[0] = FER_LIST_ENTRY(item, fer_mesh3_edge_t, list);

        vs[0] = ferMesh3EdgeVertex(es[0], 0);
        vs[1] = ferMesh3EdgeVertex(es[0], 1);
        n[0]  = fer_container_of(vs[0], node_t, vert);
        n[1]  = fer_container_of(vs[1], node_t, vert);

        echlCommonNeighbors(g, n[0], n[1]);

        for (i = 0; i < g->c->common_neighb_len; i++){
            n[2] = g->c->common_neighb[i];

            es[1] = ferMesh3VertexCommonEdge(vs[0], &n[2]->vert);
            if (!es[1] || ferMesh3EdgeFacesLen(es[1]) != 1)
                continue;

            es[2] = ferMesh3VertexCommonEdge(vs[1], &n[2]->vert);
            if (!es[2] || ferMesh3EdgeFacesLen(es[2]) != 1)
                continue;

            e = fer_container_of(es[0], edge_t, edge);
            faceNew(g, e, n[2]);
            break;
        }
    }
}

static int faceCheckAngle(fer_gsrm_t *g, fer_mesh3_vertex_t *v1,
                          fer_mesh3_vertex_t *v2, fer_mesh3_vertex_t *v3)
{
    if (ferVec3Angle(v1->v, v2->v, v3->v) > g->params.max_angle
            || ferVec3Angle(v2->v, v3->v, v1->v) > g->params.max_angle
            || ferVec3Angle(v3->v, v1->v, v2->v) > g->params.max_angle)
        return 0;
    return 1;
}

static void delFacesDangle(fer_gsrm_t *g, face_t *f1, face_t *f2)
{
    int f1_neighbors, f2_neighbors;
    size_t i;

    f1_neighbors = f2_neighbors = 0;
    for (i = 0; i < 3; i++){
        if (ferMesh3EdgeFacesLen(ferMesh3FaceEdge(&f1->face, i)) == 2)
            f1_neighbors++;
        if (ferMesh3EdgeFacesLen(ferMesh3FaceEdge(&f2->face, i)) == 2)
            f2_neighbors++;
    }

    if (f1_neighbors < f2_neighbors){
        faceDel(g, f1);
    }else if (f1_neighbors > f2_neighbors){
        faceDel(g, f2);
    }else{
        if (ferMesh3FaceArea2(&f1->face) < ferMesh3FaceArea2(&f2->face)){
            faceDel(g, f1);
        }else{
            faceDel(g, f2);
        }
    }
}

static int edgeNotUsable(fer_mesh3_edge_t *e)
{
    fer_list_t *list, *item;
    fer_mesh3_vertex_t *vs[2];
    fer_mesh3_edge_t *edge;
    size_t i, usable_edges;

    vs[0] = ferMesh3EdgeVertex(e, 0);
    vs[1] = ferMesh3EdgeVertex(e, 1);
    for (i = 0; i < 2; i++){
        usable_edges = 0;

        list = ferMesh3VertexEdges(vs[0]);
        FER_LIST_FOR_EACH(list, item){
            edge = ferMesh3EdgeFromVertexList(item);
            if (ferMesh3EdgeFacesLen(edge) < 2)
                usable_edges++;
        }

        if (usable_edges == 1)
            return 1;
    }

    return 0;
}

static int finishSurfaceTriangle(fer_gsrm_t *g, edge_t *e)
{
    size_t i;
    int ret;
    fer_mesh3_face_t *face;
    fer_mesh3_edge_t *es[2];
    fer_mesh3_vertex_t *vs[3];
    node_t *n[3];
    node_t *s;
    fer_real_t dangle;

    ret = -1;

    // get nodes of already existing face
    face = ferMesh3EdgeFace(&e->edge, 0);
    vs[0] = ferMesh3EdgeVertex(&e->edge, 0);
    vs[1] = ferMesh3EdgeVertex(&e->edge, 1);
    vs[2] = ferMesh3FaceOtherVertex(face, vs[0], vs[1]);

    n[0] = fer_container_of(vs[0], node_t, vert);
    n[1] = fer_container_of(vs[1], node_t, vert);
    n[2] = fer_container_of(vs[2], node_t, vert);
    echlCommonNeighbors(g, n[0], n[1]);

    // all common neighbors of n[0,1] form triangles
    for (i = 0; i < g->c->common_neighb_len; i++){
        if (ferMesh3EdgeFacesLen(&e->edge) == 2)
            break;

        s = g->c->common_neighb[i];
        if (s != n[2]){
            // check angle
            if (!faceCheckAngle(g, vs[0], vs[1], &s->vert))
                continue;

            // check dihedral angle
            dangle = ferVec3DihedralAngle(vs[2]->v, vs[0]->v, vs[1]->v, s->v);
            if (dangle < g->params.min_dangle)
                continue;

            // check if face can be created inside triplet of edges
            es[0] = ferMesh3VertexCommonEdge(vs[0], &s->vert);
            es[1] = ferMesh3VertexCommonEdge(vs[1], &s->vert);
            if ((es[0] && ferMesh3EdgeFacesLen(es[0]) == 2)
                    || (es[1] && ferMesh3EdgeFacesLen(es[1]) == 2))
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
    fer_list_t *list, *item;
    fer_mesh3_edge_t *edge;
    edge_t *s, *s2;

    s2 = NULL;
    list = ferMesh3VertexEdges(&n->vert);
    FER_LIST_FOR_EACH(list, item){
        edge = ferMesh3EdgeFromVertexList(item);

        if (ferMesh3EdgeFacesLen(edge) == 0)
            return NULL;

        s = fer_container_of(edge, edge_t, edge);
        if (s != e && ferMesh3EdgeFacesLen(edge) == 1){
            // there is more than two border edges incidenting with n
            if (s2 != NULL)
                return NULL;
            s2 = s;
        }
    }

    return s2;
}

static int finishSurfaceNewFace(fer_gsrm_t *g, edge_t *e)
{
    fer_mesh3_edge_t *edge;
    fer_mesh3_vertex_t *vs[2];
    node_t *ns[2], *ns2[2];
    edge_t *es[2];
    edge_t *e2, *e_new;

    // get start and end points
    vs[0] = ferMesh3EdgeVertex(&e->edge, 0);
    vs[1] = ferMesh3EdgeVertex(&e->edge, 1);
    ns[0] = fer_container_of(vs[0], node_t, vert);
    ns[1] = fer_container_of(vs[1], node_t, vert);

    es[0] = finishSurfaceGetEdge(e, ns[0]);
    es[1] = finishSurfaceGetEdge(e, ns[1]);

    if (es[0] != NULL && es[1] != NULL){
        // try to create both faces

        // obtain opossite nodes than wich already have from edge e
        vs[0] = ferMesh3EdgeOtherVertex(&es[0]->edge, &ns[0]->vert);
        vs[1] = ferMesh3EdgeOtherVertex(&es[1]->edge, &ns[1]->vert);
        ns2[0] = fer_container_of(vs[0], node_t, vert);
        ns2[1] = fer_container_of(vs[1], node_t, vert);

        e_new = NULL;
        edge = ferMesh3VertexCommonEdge(&ns[0]->vert, &ns2[1]->vert);
        if (edge){
            e2 = fer_container_of(edge, edge_t, edge);
        }else{
            e_new = e2 = edgeNew(g, ns[0], ns2[1]);
        }
        if (finishSurfaceTriangle(g, e) != 0){
            if (e_new)
                edgeDel(g, e_new);
            return -1;
        }

        e_new = NULL;
        if (!ferMesh3VertexCommonEdge(&ns2[0]->vert, &ns2[1]->vert)){
            e_new = edgeNew(g, ns2[0], ns2[1]);
        }

        if (finishSurfaceTriangle(g, e2) != 0){
            if (e_new)
                edgeDel(g, e_new);
            return -1;
        }

        return 0;
    }else if (es[0] != NULL){
        vs[0] = ferMesh3EdgeOtherVertex(&es[0]->edge, &ns[0]->vert);
        ns2[0] = fer_container_of(vs[0], node_t, vert);

        e_new = NULL;
        edge = ferMesh3VertexCommonEdge(&ns[1]->vert, &ns2[0]->vert);
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
        vs[1] = ferMesh3EdgeOtherVertex(&es[1]->edge, &ns[1]->vert);
        ns2[1] = fer_container_of(vs[1], node_t, vert);

        e_new = NULL;
        edge = ferMesh3VertexCommonEdge(&ns[0]->vert, &ns2[1]->vert);
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


static int errHeapLT(const fer_pairheap_node_t *_n1,
                     const fer_pairheap_node_t *_n2,
                     void *data)
{
    fer_gsrm_t *g = (fer_gsrm_t *)data;
    node_t *n1, *n2;

    n1 = fer_container_of(_n1, node_t, err_heap);
    n2 = fer_container_of(_n2, node_t, err_heap);

    nodeFixError(g, n1);
    nodeFixError(g, n2);
    return n1->err > n2->err;
}
