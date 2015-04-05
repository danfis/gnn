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


#include <opts.h>
#include <boruvka/parse.h>
#include <gnn/gsrm.h>

#ifdef BOR_SINGLE
# define OPTS_REAL OPTS_FLOAT
#else /* BOR_SINGLE */
# define OPTS_REAL OPTS_DOUBLE
#endif /* BOR_SINGLE */


#define DUMP_TRIANGLES_FN_LEN 100
static gnn_gsrm_params_t params;
static gnn_gsrm_t *gsrm;
static const char *is_fn = NULL;
static const char *outfile_fn;
static FILE *dump_triangles = NULL;
static char dump_triangles_fn[DUMP_TRIANGLES_FN_LEN + 1] = "";
static int no_postprocess = 0;

static int pargc;
static char **pargv;

static void usage(int argc, char *argv[], const char *opt_msg);
static void readOptions(int argc, char *argv[]);
static void printAttrs(void);

int main(int argc, char *argv[])
{
    bor_mesh3_t *mesh;
    size_t islen;
    FILE *outfile;
    bor_timer_t timer;

    readOptions(argc, argv);

    gsrm = gnnGSRMNew(&params);

    printAttrs();

    // open output file
    if (outfile_fn == NULL){
        outfile = stdout;
    }else{
        outfile = fopen(outfile_fn, "w");
        if (outfile == NULL){
            fprintf(stderr, "Can't open '%s' for writing!\n", outfile_fn);
            return -1;
        }
    }

    borTimerStart(&timer);
    borTimerStopAndPrintElapsed(&timer, stderr, " Reading input signals:\n");
    borTimerStopAndPrintElapsed(&timer, stderr, "   -- '%s'...\n", is_fn);
    islen = gnnGSRMAddInputSignals(gsrm, is_fn);
    borTimerStopAndPrintElapsed(&timer, stderr, "     --  Added %d input signals.\n", islen);
    fprintf(stderr, "\n");

    if (gnnGSRMRun(gsrm) == 0){
        if (!no_postprocess)
            gnnGSRMPostprocess(gsrm);

        borTimerStart(&timer);

        mesh = gnnGSRMMesh(gsrm);
        borMesh3DumpSVT(mesh, outfile, "Result");

        if (params.verbosity >= 2){
            fprintf(stderr, "\n");
            borTimerStopAndPrintElapsed(&timer, stderr, " Mesh dumped to '%s'.\n",
                                        (outfile == stdout ? "stdout" : outfile_fn));
        }

        if (dump_triangles != NULL){
            borMesh3DumpTriangles(mesh, dump_triangles);
            fclose(dump_triangles);

            if (params.verbosity >= 2){
                borTimerStopAndPrintElapsed(&timer, stderr,
                                            " Mesh dumped as triangles into '%s'.\n",
                                            dump_triangles_fn);
            }
        }
    }

    gnnGSRMDel(gsrm);


    // close output file
    if (outfile != stdout)
        fclose(outfile);

    return 0;
}

static void optHelp(const char *l, char s)
{
    usage(pargc, pargv, NULL);
}

static void optIncVerbosity(const char *l, char s)
{
    params.verbosity += 1;
}

static void optDumpTriangles(const char *l, char s, const char *val)
{
    dump_triangles = fopen(val, "w");
    if (dump_triangles == NULL)
        usage(pargc, pargv, "can't open file for dump-triangles");
    strncpy(dump_triangles_fn, val, DUMP_TRIANGLES_FN_LEN);
}

static void optNN(const char *l, char s)
{
    if (strcmp(l, "nn-gug") == 0){
        params.nn.type = BOR_NN_GUG;
    }else if (strcmp(l, "nn-vptree") == 0){
        params.nn.type = BOR_NN_VPTREE;
    }else if (strcmp(l, "nn-linear") == 0){
        params.nn.type = BOR_NN_LINEAR;
    }
}

static void optOutput(const char *l, char s, const char *val)
{
    if (strcmp(val, "stdout") == 0){
        outfile_fn = NULL;
    }else{
        outfile_fn = val;
    }
}

static void readOptions(int argc, char *argv[])
{
    int i;

    pargc = argc;
    pargv = argv;

    gnnGSRMParamsInit(&params);
    params.verbosity = 1;
    params.nn.gug.num_cells = 0;
    params.nn.gug.max_dens = 0.1;
    params.nn.gug.expand_rate = 1.5;

    optsAdd("help",             'h', OPTS_NONE,  NULL, OPTS_CB(optHelp));
    optsAdd("verbose",          'v', OPTS_NONE,  NULL, OPTS_CB(optIncVerbosity));
    optsAdd("epsilon-n",         0, OPTS_REAL,   (void *)&params.en, NULL);
    optsAdd("epsilon-b",         0, OPTS_REAL,   (void *)&params.eb, NULL);
    optsAdd("lambda",            0, OPTS_SIZE_T, (void *)&params.lambda, NULL);
    optsAdd("beta",              0, OPTS_REAL,   (void *)&params.beta, NULL);
    optsAdd("alpha",             0, OPTS_REAL,   (void *)&params.alpha, NULL);
    optsAdd("age-max",           0, OPTS_INT,    (void *)&params.age_max, NULL);
    optsAdd("max-nodes",         0, OPTS_SIZE_T, (void *)&params.max_nodes, NULL);
    optsAdd("min-dangle",        0, OPTS_REAL,   (void *)&params.min_dangle, NULL);
    optsAdd("max-angle",         0, OPTS_REAL,   (void *)&params.max_angle, NULL);
    optsAdd("angle-merge-edges", 0, OPTS_REAL,   (void *)&params.angle_merge_edges, NULL);
    optsAdd("dump-triangles",    0, OPTS_STR,    NULL, OPTS_CB(optDumpTriangles));
    optsAdd("nn-gug",            0, OPTS_NONE,   NULL, OPTS_CB(optNN));
    optsAdd("nn-vptree",         0, OPTS_NONE,   NULL, OPTS_CB(optNN));
    optsAdd("nn-linear",         0, OPTS_NONE,   NULL, OPTS_CB(optNN));
    optsAdd("vptree-max-size",   0, OPTS_INT,    (void *)&params.nn.vptree.maxsize, NULL);
    optsAdd("gug-max-dens",      0, OPTS_REAL,   (void *)&params.nn.gug.max_dens, NULL);
    optsAdd("gug-expand-rate",   0, OPTS_REAL,   (void *)&params.nn.gug.expand_rate, NULL);
    optsAdd("unoptimized-err",   0, OPTS_NONE,   (void *)&params.unoptimized_err, NULL);
    optsAdd("no-postprocess",    0, OPTS_NONE,   (void *)&no_postprocess, NULL);
    optsAdd("output",           'o', OPTS_STR,   NULL, OPTS_CB(optOutput));

    if (opts(&argc, argv) != 0){
        usage(argc, argv, NULL);
    }

    if (argc > 2){
        for (i = 1; i < argc; i++){
            if (argv[i][0] == '-' && argv[i][1] == '-'){
                fprintf(stderr, "Unknown option %s\n", argv[i]);
            }
        }
        usage(argc, argv, NULL);
    }else if (argc <= 1){
        usage(argc, argv, "filename must be specified");
    }
    is_fn = argv[argc - 1];
}

static void usage(int argc, char *argv[], const char *opt_msg)
{
    if (opt_msg != NULL){
        fprintf(stderr, "%s\n", opt_msg);
    }

    fprintf(stderr, "\n");
    fprintf(stderr, "Usage %s [ options ] filename\n", argv[0]);
    fprintf(stderr, "   Options: --epsilon-b float  Winner learning rate\n");
    fprintf(stderr, "            --epsilon-n float  Winner's neighbors learning rate\n");
    fprintf(stderr, "            --lambda    int    Steps in cycle\n");
    fprintf(stderr, "            --beta      float  Error counter decreasing rate\n");
    fprintf(stderr, "            --alpha     float  Error counter decreasing rate\n");
    fprintf(stderr, "            --age-max   int\n");
    fprintf(stderr, "            --max-nodes int    Stop Criterium\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "            --min-dangle        float  Minimal dihedral angle between faces\n");
    fprintf(stderr, "            --max-angle         float  Maximal angle in cusp of face\n");
    fprintf(stderr, "            --angle-merge-edges float  Minimal angle between edges to merge them\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "            --unoptimized-err   Turn off optimization of error handling\n");
    fprintf(stderr, "            --no-postprocess    Turn off postprocessing\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "            --nn-gug                  Use Growing Uniform Grid for NN search (default choise)\n");
    fprintf(stderr, "            --nn-vptree               Use VP-Tree for NN search\n");
    fprintf(stderr, "            --nn-linear               Use linear NN search\n");
    fprintf(stderr, "            --vptree-max-size  int    Maximal number of elements in leaf node\n");
    fprintf(stderr, "            --gug-max-dens     float  Maximal density\n");
    fprintf(stderr, "            --gug-expand-rate  float  Expand rate\n");
    fprintf(stderr, "\n");

    fprintf(stderr, "\n");

    fprintf(stderr, "            --outfile / -o   filename Filename where will be dumped resulting mesh (stdout is default)\n");
    fprintf(stderr, "            --dump-triangles filename Filename where will be stored triangles from reconstructed object.\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "            -v / -vv / ...  Increases verbosity\n");
    fprintf(stderr, "\n");

    exit(-1);
}

void printAttrs(void)
{
    const gnn_gsrm_params_t *param;

    param = gnnGSRMParams(gsrm);

    fprintf(stderr, "Attributes:\n");
    fprintf(stderr, "    lambda:    %d\n", (int)param->lambda);
    fprintf(stderr, "    eb:        %f\n", (float)param->eb);
    fprintf(stderr, "    en:        %f\n", (float)param->en);
    fprintf(stderr, "    alpha      %f\n", (float)param->alpha);
    fprintf(stderr, "    beta:      %f\n", (float)param->beta);
    fprintf(stderr, "    age_max:   %d\n", (int)param->age_max);
    fprintf(stderr, "    max nodes: %d\n", (int)param->max_nodes);
    fprintf(stderr, "\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "    min d. angle:  %f\n", (float)param->min_dangle);
    fprintf(stderr, "    max angle:     %f\n", (float)param->max_angle);
    fprintf(stderr, "    ang. merge e.: %f\n", (float)param->angle_merge_edges);
    fprintf(stderr, "\n");
    fprintf(stderr, "    input signals: %s\n", is_fn);
    fprintf(stderr, "\n");
    fprintf(stderr, "    outfile: %s\n", (outfile_fn == NULL ? "stdout" : outfile_fn));
    fprintf(stderr, "\n");
    fprintf(stderr, "VP-Tree:\n");
    fprintf(stderr, "    maxsize: %d\n", (int)param->nn.vptree.maxsize);
    fprintf(stderr, "GUG:\n");
    fprintf(stderr, "    num cells:   %d\n", (int)param->nn.gug.num_cells);
    fprintf(stderr, "    max dens:    %f\n", (float)param->nn.gug.max_dens);
    fprintf(stderr, "    expand rate: %f\n", (float)param->nn.gug.expand_rate);
    fprintf(stderr, "\n");
}

