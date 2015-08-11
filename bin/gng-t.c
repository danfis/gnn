#include <stdio.h>
#include <signal.h>
#include <boruvka/timer.h>
#include <boruvka/pc.h>
#include <boruvka/gug.h>
#include <boruvka/alloc.h>
#include <boruvka/vec3.h>
#include <gnn/gng-t.h>

struct _node_t {
    gnn_gngt_node_t node;

    bor_vec_t *w;
    bor_gug_el_t gug;

    int _id;
};
typedef struct _node_t node_t;

int dim;
bor_real_t target;
BOR_VEC(tmpv, 10);
int dump = 0, dump_num = 0;

gnn_gngt_params_t params;
gnn_gngt_ops_t ops;
gnn_gngt_t *gng;

bor_gug_params_t gug_params;
bor_gug_t *gug;

bor_timer_t timer;

bor_pc_t *pc;
bor_pc_it_t pcit;

static int terminate(void *data);
static void callback(void *data);
static const void *input_signal(void *data);
static gnn_gngt_node_t *new_node(const void *is, void *);
static gnn_gngt_node_t *new_node_between(const gnn_gngt_node_t *n1,
                                         const gnn_gngt_node_t *n2, void *);
static void del_node(gnn_gngt_node_t *n, void *);
static void nearest(const void *input_signal,
                    gnn_gngt_node_t **n1,
                    gnn_gngt_node_t **n2, void *);
static bor_real_t dist2(const void *input_signal,
                        const gnn_gngt_node_t *node, void *);
static void move_towards(gnn_gngt_node_t *node,
                         const void *input_signal,
                         bor_real_t fraction, void *);
static void dumpSVT(gnn_gngt_t *gng, FILE *out, const char *name);

static void sigDump(int sig);

int main(int argc, char *argv[])
{
    size_t size;
    bor_real_t aabb[30];

    if (argc < 4){
        fprintf(stderr, "Usage: %s dim file.pts target\n", argv[0]);
        return -1;
    }

    dim = atoi(argv[1]);
    target = atof(argv[3]);

    // read input points
    pc = borPCNew(dim);
    size = borPCAddFromFile(pc, argv[2]);
    fprintf(stderr, "Added %d points from %s\n", (int)size, argv[2]);
    borPCPermutate(pc);
    borPCItInit(&pcit, pc);


    // create NN search structure
    borGUGParamsInit(&gug_params);
    gug_params.dim         = atoi(argv[1]);
    gug_params.num_cells   = 0;
    gug_params.max_dens    = 0.1;
    gug_params.expand_rate = 1.5;
    borPCAABB(pc, aabb);
    gug_params.aabb = aabb;
    gug = borGUGNew(&gug_params);

    // create GNG-T
    gnnGNGTParamsInit(&params);
    params.target = target;
    //params.age_max = 1000;
    //params.lambda = 10000;

    gnnGNGTOpsInit(&ops);
    ops.new_node         = new_node;
    ops.new_node_between = new_node_between;
    ops.del_node         = del_node;
    ops.input_signal     = input_signal;
    ops.nearest          = nearest;
    ops.dist2            = dist2;
    ops.move_towards     = move_towards;
    ops.terminate        = terminate;
    ops.callback         = callback;
    ops.callback_period = 300;
    ops.data = NULL;

    gng = gnnGNGTNew(&ops, &params);

    signal(SIGINT, sigDump);

    borTimerStart(&timer);
    gnnGNGTRun(gng);
    callback(NULL);
    fprintf(stderr, "\n");

    dumpSVT(gng, stdout, NULL);

    gnnGNGTDel(gng);
    borGUGDel(gug);
    borPCDel(pc);

    return 0;
}


static int terminate(void *data)
{
    FILE *fout;
    static char fn[1000];

    if (dump){
        dump = 0;

        sprintf(fn, "gng-t-%06d.svt", dump_num++);
        printf("\nDumping network to `%s'...\n", fn);
        fout = fopen(fn, "w");
        dumpSVT(gng, fout, NULL);
        fclose(fout);
    }

    return 0;
    return gnnGNGTNodesLen(gng) >= 1000;
}

static void callback(void *data)
{
    size_t nodes_len;

    nodes_len = gnnGNGTNodesLen(gng);

    borTimerStopAndPrintElapsed(&timer, stderr, " n: %d, avg err: %f, target: %f\r",
            nodes_len, gnnGNGTAvgErr(gng), target);
}

static const void *input_signal(void *data)
{
    const bor_vec_t *v;

    if (borPCItEnd(&pcit)){
        borPCPermutate(pc);
        borPCItInit(&pcit, pc);
    }
    v = borPCItGet(&pcit);
    borPCItNext(&pcit);
    return (const void *)v;
}

static gnn_gngt_node_t *new_node(const void *is, void *_)
{
    node_t *n;

    n = BOR_ALLOC(node_t);
    n->w = borVecClone(dim, (const bor_vec_t *)is);
    borGUGElInit(&n->gug, n->w);
    borGUGAdd(gug, &n->gug);

    return &n->node;
}

static gnn_gngt_node_t *new_node_between(const gnn_gngt_node_t *_n1,
                                         const gnn_gngt_node_t *_n2, void *_)
{
    node_t *n1 = bor_container_of(_n1, node_t, node);
    node_t *n2 = bor_container_of(_n2, node_t, node);

    borVecAdd2(dim, tmpv, n1->w, n2->w);
    borVecScale(dim, tmpv, BOR_REAL(0.5));

    return new_node((const void *)tmpv, NULL);
}

static void del_node(gnn_gngt_node_t *_n, void *_)
{
    node_t *n = bor_container_of(_n, node_t, node);

    borGUGRemove(gug, &n->gug);
    borVecDel(n->w);
    free(n);
}

static void nearest(const void *is,
                    gnn_gngt_node_t **n1,
                    gnn_gngt_node_t **n2, void *_)
{
    bor_gug_el_t *els[2];
    node_t *ns[2];

    borGUGNearest(gug, (const bor_vec_t *)is, 2, els);
    ns[0] = bor_container_of(els[0], node_t, gug);
    ns[1] = bor_container_of(els[1], node_t, gug);
    *n1 = &ns[0]->node;
    *n2 = &ns[1]->node;
}

static bor_real_t dist2(const void *is,
                        const gnn_gngt_node_t *node, void *_)
{
    node_t *n = bor_container_of(node, node_t, node);

    return borVecDist2(dim, (const bor_vec_t *)is, n->w);
}

static void move_towards(gnn_gngt_node_t *node,
                         const void *is,
                         bor_real_t fraction, void *_)
{
    node_t *n;

    n = bor_container_of(node, node_t, node);

    borVecSub2(dim, tmpv, (const bor_vec_t *)is, n->w);
    borVecScale(dim, tmpv, fraction);
    borVecAdd(dim, n->w, tmpv);

    borGUGUpdate(gug, &n->gug);
}

static void dumpSVT(gnn_gngt_t *gng, FILE *out, const char *name)
{
    bor_list_t *list, *item;
    bor_net_node_t *nn;
    gnn_gngt_node_t *gn;
    node_t *n;
    bor_net_edge_t *e;
    size_t i, id1, id2;

    if (dim != 2 && dim != 3)
        return;

    fprintf(out, "--------\n");

    if (name)
        fprintf(out, "Name: %s\n", name);

    fprintf(out, "Points:\n");
    list = gnnGNGTNodes(gng);
    i = 0;
    BOR_LIST_FOR_EACH(list, item){
        gn = gnnGNGTNodeFromList(item);
        n  = bor_container_of(gn, node_t, node);

        n->_id = i++;
        if (dim == 2){
            borVec2Print((const bor_vec2_t *)n->w, out);
        }else{
            borVec3Print((const bor_vec3_t *)n->w, out);
        }
        fprintf(out, "\n");
    }


    fprintf(out, "Edges:\n");
    list = gnnGNGTEdges(gng);
    BOR_LIST_FOR_EACH(list, item){
        e = BOR_LIST_ENTRY(item, bor_net_edge_t, list);

        nn = borNetEdgeNode(e, 0);
        gn = gnnGNGTNodeFromNet(nn);
        n  = bor_container_of(gn, node_t, node);
        id1 = n->_id;

        nn = borNetEdgeNode(e, 1);
        gn = gnnGNGTNodeFromNet(nn);
        n  = bor_container_of(gn, node_t, node);
        id2 = n->_id;
        fprintf(out, "%d %d\n", (int)id1, (int)id2);
    }

    fprintf(out, "--------\n");
}

static void sigDump(int sig)
{
    dump = 1;
}

