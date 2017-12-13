/* 
 * File:   main.c
 * Author: Alexander Cech (08900070)
 *
 * Created on November 28, 2017, 5:16 PM
 */

// Define in BITONIC_PARTYPE in makefile:

#if BITONIC_PARTYPE == 1
#define BITONIC_USE_OPENMP
#elif BITONIC_PARTYPE == 2
#define BITONIC_USE_CILK
#elif BITONIC_PARTYPE == 3
#define BITONIC_USE_MPI
#endif

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <string.h>
#include <assert.h>

#ifdef BITONIC_USE_OPENMP
#include <omp.h>
#endif

#ifdef BITONIC_USE_CILK
#include <cilk/cilk.h>
#endif

//#define ARTIFICIAL_COMPARE_TIME_US 10
//#define ARTIFICIAL_SWAP_TIME_US    20
#define DEFAULT_CILK_CUTOFF 100


typedef int dataitem;

int compare_dataitem(const void *p1, const void *p2) {
    const dataitem *di1 = (const dataitem *) p1;
    const dataitem *di2 = (const dataitem *) p2;
    return (*di1 > *di2) - (*di1 < *di2);
}

void bail(char *msg) {
    if (msg) fprintf(stderr, "%s\n", msg);
    exit(1);
}

void swap_dataitems(dataitem *a1, dataitem *a2) {
    #ifdef ARTIFICIAL_SWAP_TIME_US
        usleep(ARTIFICIAL_SWAP_TIME_US);
    #endif
    dataitem tmp = *a1; *a1 = *a2; *a2 = tmp;
}


void compare_and_swap(dataitem *a1, dataitem *a2) {
    #ifdef ARTIFICIAL_COMPARE_TIME_US
        usleep(ARTIFICIAL_COMPARE_TIME_US);
    #endif
    if (compare_dataitem(a1, a2) > 0) 
        swap_dataitems(a1, a2);
}

int reverse_data_array(dataitem *a, int n) {
    for (int i = 0; i < n/2; i++) {
        swap_dataitems(&a[i], &a[n-1-i]);
    }
    return n/2; // work
}

void print_data_array(char *info, dataitem *data, int size, int maxtoprint) {
    if (info) printf("%s", info);
    for (int i = 0; i < size; i++) {
        if (maxtoprint && (i >= maxtoprint)) { printf(" ..."); break; }
        printf(" %d", data[i]);
    }
    printf("\n");
}

void print_data_array_ex(char *info, int start, dataitem *base, int size, int maxtoprint) {
    if (info) printf("%s", info);
    for (int i = 0; i < size; i++) {
        if (maxtoprint && (i >= maxtoprint)) { printf(" ..."); break; }
        if (start+i >= 0) printf(" %d", base[start+i]); else printf(" X");
    }
    printf("\n");
}

int debug_level = 0;

// <editor-fold desc="----- OpenMP implementation">
// ---------- OpenMP implementation ----------
#ifdef BITONIC_USE_OPENMP
int openmp_bitonic_split_rev(dataitem *a_base, int start, int len, int rstart, int rlen) {
    if (debug_level >= 10) printf("tid %d: bitonic_split_rev(%p, %d, %d, %d, %d)\n", omp_get_thread_num(), a_base, start, len, rstart, rlen);

    //if (debug_level >= 20) print_data_array_ex(" before: ", start, a_base, len, 20);
    
    int half = len >> 1;
    //if (len & 1) start_b++; // never can happen here

    for (int i = 0; i < half; i++) {
        int ix1 = start+i, ix2 = start+half+i;
      
        if (debug_level >= 15) printf("tid %d: in loop i=%d\n", omp_get_thread_num(), i);
        
        if (ix1 >= rstart && ix1 < rstart+rlen) ix1 = rstart+rlen-1-(ix1-rstart);
        if (ix2 >= rstart && ix2 < rstart+rlen) ix2 = rstart+rlen-1-(ix2-rstart);
        
        if (ix1 >= 0 && ix2 >= 0)
            compare_and_swap(&a_base[ix1], &a_base[ix2]);
    }
    
    //if (debug_level >= 20) print_data_array_ex(" after : ", start, a_base, len, 20);
    
    return half; // work
}

int openmp_bitonic_merge_rev(dataitem *a, int len, int rlen) {
    int work = 0;

    // calc depth = lg len:
    int depth = 0;
    for (int t = len; t > 1; t >>= 1, depth++); 
    if ((1 << depth) < len) depth++;
    
    // fake sequence (think: filled with minimum elements at start) + real sequence is 2^depth long
    int totalseqlen = 1 << depth;
    int dummyseqlen = totalseqlen - len;
    
    int data_n = len - rlen;
    
    for (int d=0, numsplits=1, seqlen=totalseqlen; d<depth; d++, seqlen/=2, numsplits*=2) {
        #pragma omp parallel
        {
            #pragma omp for collapse(2)
            for (int split = 0; split < numsplits; split++) {
                for (int i = 0; i < (seqlen >> 1); i++) {
                    int splitstart  = split * seqlen - dummyseqlen;
                    int rstart,rlen;
                    if (splitstart < data_n && data_n < splitstart + seqlen)  {
                        int left = data_n - splitstart, right = seqlen - left;
                        // view shorter part of sequence as reversed for split
                        if (left > right) {
                            rstart = splitstart + left; rlen = right;
                        }  else {
                            rstart = splitstart;        rlen = left;
                        }
                    } else {
                        // classic split
                        rstart = 0; rlen = 0;
                    }
                    
                    int ix1 = splitstart+i, ix2 = splitstart+(seqlen >> 1)+i;

                    if (debug_level >= 15) printf("tid %d: in depth=%d, split=%d, loop i=%d\n", omp_get_thread_num(), d, split, i);

                    if (ix1 >= rstart && ix1 < rstart+rlen) ix1 = rstart+rlen-1-(ix1-rstart);
                    if (ix2 >= rstart && ix2 < rstart+rlen) ix2 = rstart+rlen-1-(ix2-rstart);

                    if (ix1 >= 0 && ix2 >= 0)
                        compare_and_swap(&a[ix1], &a[ix2]);
                }
                
                
                /*
                int splitstart  = split * seqlen - dummyseqlen;
                // if transition n to m is in this split sequence...
                if (splitstart < data_n && data_n < splitstart + seqlen)  {
                    int left = data_n - splitstart, right = seqlen - left;
                    // view shorter part of sequence as reversed for split
                    if (left > right) {
                        openmp_bitonic_split_rev(a, splitstart, seqlen, splitstart + left, right);
                    }  else {
                        openmp_bitonic_split_rev(a, splitstart, seqlen, splitstart,        left );
                    }
                } else {
                    // classic split
                    openmp_bitonic_split_rev(a, splitstart, seqlen, 0, 0);
                }
                */
            }
        }
    }
    
    return 0;
}

#endif
// </editor-fold>

// <editor-fold desc="----- cilk implementation">
// ---------- cilk implementation ----------
#ifdef BITONIC_USE_CILK
int cilk_cutoff = DEFAULT_CILK_CUTOFF;

// "reverse" bitonic split (the rlen items from rstart are considered reversed)
// if len == 0 this is just a classic split
int cilk_bitonic_split_rev(dataitem *a, int len, int rstart, int rlen) {
    if (debug_level >= 10) printf("bitonic_split_rev(%p, %d, %d)\n", a, len, rlen);

    if (debug_level >= 20) print_data_array(" before: ", a, len, 20);
    
    int half = len >> 1, start_b = half;
    if (len & 1) start_b++;

    cilk_for (int i = 0; i < half; i++) {
        int ix1 = i, ix2 = start_b+i;
        
        if (ix1 >= rstart && ix1 < rstart+rlen) ix1 = rstart+rlen-1-(ix1-rstart);
        if (ix2 >= rstart && ix2 < rstart+rlen) ix2 = rstart+rlen-1-(ix2-rstart);
        
        compare_and_swap(&a[ix1], &a[ix2]);
    }
    
    if (debug_level >= 20) print_data_array(" after : ", a, len, 20);
    
    return half; // work
}

// consider the last rlen items reversed
int cilk_bitonic_merge_rev(dataitem *a, int len, int rlen) {
    if (len <= 1) return 0;
    
    int work = 0;

    if (debug_level >= 10) printf("bitonic_merge_rev(%p, %d, %d)\n", a, len, rlen);
    
    assert(rlen >= 0 && rlen < len);
    
    int len_a = len >> 1;
    int next_rlen_a, next_rlen_b;
    
    if (rlen > 0) {
        int left = len - rlen, right = rlen;

        // only for the split, view the shorter part of sequence as reversed, 
        // so that the center item is never reversed
        if (left > right) {
            // right side is reversed
            if ((len & 1) && compare_dataitem(&a[len_a], &a[left]) < 0) len_a++;  
            work += cilk_bitonic_split_rev(a, len, left, right);
        } else {
            // left side is reversed
            if ((len & 1) && compare_dataitem(&a[len_a], &a[len-1]) < 0) len_a++;  
            work += cilk_bitonic_split_rev(a, len, 0, left);
        }
        
        int len_b = len - len_a;
        next_rlen_a = rlen > len_b ? rlen - len_b : 0;
        next_rlen_b = rlen < len_b ? rlen         : 0;
    } else {
        // this is a classic merge, could just call bitonic_merge_classic(a, len)
        if ((len & 1) && (compare_dataitem(&a[len_a], &a[len-1]) < 0)) len_a++;   
        work += cilk_bitonic_split_rev(a, len, 0, 0);
        next_rlen_a = next_rlen_b = 0;
    }
    
    int work1,work2;
    if (len_a > cilk_cutoff) {
        work1 = cilk_spawn cilk_bitonic_merge_rev(a, len_a, next_rlen_a);
    } else {
        work1 =            cilk_bitonic_merge_rev(a, len_a, next_rlen_a);
    }
    work2 = cilk_bitonic_merge_rev(a + len_a, len - len_a, next_rlen_b);

    cilk_sync;
    work += work1 + work2;
    
    return work;
}
#endif
// </editor-fold>

// <editor-fold desc="----- Serial implementations">
// ---------- serial implementations ----------

// "classic" bitonic split (data is a bitonic sequence)
int bitonic_split_classic(dataitem *a, int len) {
    int half = len >> 1, start_b = half;
    if (len & 1) start_b++;
    for (int i = 0; i < half; i++) {
        compare_and_swap(&a[i], &a[start_b+i]);
    }
    return half; // work
}

// "classic" bitonic merge (data is a bitonic sequence)
int bitonic_merge_classic(dataitem *a, int len) {
    if (len <= 1) return 0;
    
    int work = 0;
    
    int len_a = len >> 1;
    if (len & 1) {
        if (compare_dataitem(&a[len_a], &a[len-1]) < 0) len_a++;
    }
    
    work += bitonic_split_classic(a, len);
    work += bitonic_merge_classic(a, len_a);
    work += bitonic_merge_classic(a + len_a, len - len_a);
    
    return work;
}

// "reverse" bitonic split (the rlen items from rstart are considered reversed)
// if len == 0 this is just a classic split
int bitonic_split_rev(dataitem *a, int len, int rstart, int rlen) {
    if (debug_level >= 10) printf("bitonic_split_rev(%p, %d, %d)\n", a, len, rlen);

    if (debug_level >= 20) print_data_array(" before: ", a, len, 20);
    
    int half = len >> 1, start_b = half;
    if (len & 1) start_b++;

    for (int i = 0; i < half; i++) {
        int ix1 = i, ix2 = start_b+i;
        
        if (ix1 >= rstart && ix1 < rstart+rlen) ix1 = rstart+rlen-1-(ix1-rstart);
        if (ix2 >= rstart && ix2 < rstart+rlen) ix2 = rstart+rlen-1-(ix2-rstart);
        
        //printf("cmp ix %d %d\n",ix1,ix2);
        
        compare_and_swap(&a[ix1], &a[ix2]);
    }
    
    if (debug_level >= 20) print_data_array(" after : ", a, len, 20);
    
    return half; // work
}

// consider the last rlen items reversed
int bitonic_merge_rev(dataitem *a, int len, int rlen) {
    if (len <= 1) return 0;
    
    int work = 0;

    if (debug_level >= 10) printf("bitonic_merge_rev(%p, %d, %d)\n", a, len, rlen);
    
    assert(rlen >= 0 && rlen <= len);
    
    int len_a = len >> 1;
    int next_rlen_a, next_rlen_b;
    
    if (rlen > 0) {
        int left = len - rlen, right = rlen;

        // only for the split, view the shorter part of sequence as reversed, 
        // so that the center item is never reversed
        if (left > right) {
            // right side is reversed
            if ((len & 1) && compare_dataitem(&a[len_a], &a[left]) < 0) len_a++;  
            work += bitonic_split_rev(a, len, left, right);
        } else {
            // left side is reversed
            if ((len & 1) && compare_dataitem(&a[len_a], &a[len-1]) < 0) len_a++;  
            work += bitonic_split_rev(a, len, 0, left);
        }
        
        int len_b = len - len_a;
        next_rlen_a = rlen > len_b ? rlen - len_b : 0;
        next_rlen_b = rlen < len_b ? rlen         : 0;
    } else {
        // this is a classic merge, could just call bitonic_merge_classic(a, len)
        if ((len & 1) && (compare_dataitem(&a[len_a], &a[len-1]) < 0)) len_a++;
        work += bitonic_split_rev(a, len, 0, 0);
        next_rlen_a = next_rlen_b = 0;
    }
    
    work += bitonic_merge_rev(a,         len_a,       next_rlen_a);
    work += bitonic_merge_rev(a + len_a, len - len_a, next_rlen_b);
    
    return work;
}

// </editor-fold>


void check_sorted(dataitem *data, int size) {
    if (size < 2) return;
    for (int i = 1; i < size; i++) {
        if (data[i] < data[i-1]) {
            char buf[200];
            sprintf(buf, "** check_sorted failed: [%d]=%d [%d]=%d **",i-1,data[i-1],i,data[i]);
            bail(buf);
        }
    }
}

// ---------- Time measurement functions ----------
#ifdef BITONIC_USE_OPENMP
typedef double measure_context;

void start_measure(measure_context *t1) {
    *t1 = omp_get_wtime();
}

long end_measure(measure_context *t1) {
    double t2 = omp_get_wtime();
    return ((t2 - *t1) * 1000.0);
}
#else
typedef struct timespec measure_context;

void start_measure(measure_context *tp1) {
    clock_gettime(CLOCK_REALTIME, tp1);
}

// returns # of milliseconds since start_measure
long end_measure(measure_context *tp1) {
    struct timespec tp2;
    clock_gettime(CLOCK_REALTIME, &tp2);
    return (tp2.tv_sec - tp1->tv_sec) * 1000 +
           ((tp2.tv_nsec / 1000000) - (tp1->tv_nsec / 1000000));
}
#endif
// --------------------------------------------------

struct datainfo {
    dataitem *data, *backup;
    int n,m;
};

int verbose = 0, meascnt = 10; // global

void restore_data_backup(struct datainfo *di) {
    memcpy(di->data, di->backup, (di->n+di->m) * sizeof(dataitem)); // restore backup
}

void runtest(char *label, struct datainfo *di, int (* testfunc)(struct datainfo *)) {
    measure_context t0;
    double sum_millis = 0;
    double sum_work = 0;
   
    for (int i = 0; i < meascnt; i++) {
        restore_data_backup(di);
        start_measure(&t0);
        sum_work += testfunc(di);
        sum_millis += end_measure(&t0);
        if (verbose && i==0) print_data_array(label, di->data, di->n + di->m, 50);
        check_sorted(di->data, di->n + di->m);
    }

    printf("%s: %.0f ms, %.0f ops\n", label, sum_millis/meascnt, sum_work/meascnt);
}

int testcase_serial_reversal(struct datainfo *di) {
    int work = 0;
    work += reverse_data_array(&di->data[di->n], di->m); // needs second array sorted descending
    work += bitonic_merge_classic(di->data, di->n + di->m);
    return work;
}

int testcase_serial_noreversal(struct datainfo *di) {
    return bitonic_merge_rev(di->data, di->n + di->m, di->m);
}

#ifdef BITONIC_USE_OPENMP
int testcase_openmp_reversal(struct datainfo *di) {
    int work = 0;
    // FIXME - parallelize reversal
    work += reverse_data_array(&di->data[di->n], di->m); // needs second array sorted descending
    work += openmp_bitonic_merge_rev(di->data, di->n + di->m, 0);
    return work;
}

int testcase_openmp_noreversal(struct datainfo *di) {
    return openmp_bitonic_merge_rev(di->data, di->n + di->m, di->m);
}

#endif

#ifdef BITONIC_USE_CILK
int testcase_cilk_reversal(struct datainfo *di) {
    int work = 0;
    // FIXME - parallelize reversal
    work += reverse_data_array(&di->data[di->n], di->m); // needs second array sorted descending
    work += cilk_bitonic_merge_rev(di->data, di->n + di->m, 0);
    return work;
}

int testcase_cilk_noreversal(struct datainfo *di) {
    return cilk_bitonic_merge_rev(di->data, di->n + di->m, di->m);
}
#endif

void usage(char *prg) {
    fprintf(stderr, 
#ifdef BITONIC_USE_OPENMP
        "-- BITONIC OPENMP VARIANT --\n"
#endif
#ifdef BITONIC_USE_CILK
        "-- BITONIC CILK VARIANT --\n"
#endif
#ifdef BITONIC_USE_MPI
        "-- BITONIC MPI VARIANT --\n"
#endif
        "Compiled with gcc %d.%d.%d\n\n"
        "Usage: %s [options]\n"
        "Valid options are:\n"
        "-n <value>    size of 1st ordered sequence [default 100]\n"
        "-m <value>    size of 2nd ordered sequence [default 100]\n"
        "-r <value>    random seed number [default = time(0)]\n"
        "-l <value>    low  bound of random data values [default 0]\n"
        "-h <value>    high bound of random data values [default 999]\n"
        "-t <testcase> testcase to run [default 0 = all]\n"
        "-c <value>    average measurements over <value> runs [default 10]\n"
        "-p <value>    number of threads to use (OpenMP)\n"
        "-u <value>    cutoff value (cilk)\n"
        "-v            verbose\n"
        "-d <level>    debug level\n", 
        __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__,
        prg
    );
    exit(1);
}

int main(int argc, char** argv) {

    int n = 100, m = 100, seed = time(0), 
        rnd_lo = 0, rnd_hi = 999, testcase = 0;
    
#ifdef BITONIC_USE_OPENMP
    int maxthreads = omp_get_max_threads();
    int numthreads = maxthreads;
#endif
#ifdef BITONIC_USE_CILK
    int numthreads = 4;
#endif
    
    // parse arguments
    opterr = 1;
    int c;
    while ((c = getopt(argc, argv, "n:m:r:l:h:t:c:p:u:vd:")) != -1) {
        switch (c) {
            case 'n': n          = atoi(optarg); break;
            case 'm': m          = atoi(optarg); break;
            case 'r': seed       = atoi(optarg); break;
            case 'l': rnd_lo     = atoi(optarg); break;
            case 'h': rnd_hi     = atoi(optarg); break;
            case 't': testcase   = atoi(optarg); break;
            case 'c': meascnt    = atoi(optarg); break;
            case 'p': numthreads = atoi(optarg); break;
#ifdef BITONIC_USE_CILK            
            case 'u': cilk_cutoff= atoi(optarg); break;
#endif
            case 'v': verbose    = 1;            break;
            case 'd': debug_level= atoi(optarg); break;
            default:  usage(argv[0]);
        }
    }
    if (optind < argc || n < 0 || m < 0 || rnd_lo > rnd_hi || meascnt < 1) usage(argv[0]);
#ifdef BITONIC_USE_OPENMP
    if (numthreads > maxthreads) {
        fprintf(stderr, "numthreads invalid, maximum is %d\n", maxthreads);
        bail(NULL);
    }
    omp_set_num_threads(numthreads);
#endif
    
    srand(seed);
    
    printf("n=%d m=%d seed=%d range=[%d..%d] meascnt=%d" 
        #ifdef  BITONIC_USE_OPENMP
           " p=%d" 
        #endif    
        #ifdef  BITONIC_USE_CILK
           " cutoff=%d" 
        #endif    
           "\n"
            , n, m, seed, rnd_lo, rnd_hi, meascnt
        #ifdef  BITONIC_USE_OPENMP
            , numthreads
        #endif    
        #ifdef  BITONIC_USE_CILK
            , cilk_cutoff
        #endif    
            );
    
    // create and initialize data buffer
    dataitem *data = malloc((n+m) * sizeof(dataitem));
    if (! data) bail("Cannot allocate memory");

    for (int i = 0; i < n+m; i++) data[i] = rnd_lo + rand() % (rnd_hi+1-rnd_lo);
    qsort(&data[0], n, sizeof(dataitem), compare_dataitem);
    qsort(&data[n], m, sizeof(dataitem), compare_dataitem);

    // keep a backup of the initialized data array for comparative tests
    dataitem *backup = malloc((n+m) * sizeof(dataitem));
    if (! backup) bail("Cannot allocate memory");
    memcpy(backup, data, (n+m) * sizeof(dataitem));
    
    if (verbose) {
        print_data_array("A =", &data[0], n, 30);
        print_data_array("B =", &data[n], m, 30);
    }
    
    // ------------------------------------------------------------

    struct datainfo datainfo;
    datainfo.data   = data;
    datainfo.backup = backup;
    datainfo.n      = n;
    datainfo.m      = m;
    
    struct timespec t0;
    long millis;
    
    // test method from lecture (needs explicit reversal)
    if (!testcase || testcase == 1) runtest("Serial ", &datainfo, testcase_serial_reversal);

    // test improved version (without reversing array B first)
    //if (!testcase || testcase == 2) runtest("T2 ", &datainfo, testcase_serial_noreversal);

#ifdef BITONIC_USE_OPENMP
    // test openmp with explicit reversal
    //if (!testcase || testcase == 3) runtest("T3 ", &datainfo, testcase_openmp_reversal);

    // test openmp without reversal
    if (!testcase || testcase == 2) runtest("OpenMP ", &datainfo, testcase_openmp_noreversal);
#endif
    
#ifdef  BITONIC_USE_CILK
    // test cilk with explicit reversal
    // if (!testcase || testcase == 3) runtest("T3 ", &datainfo, testcase_cilk_reversal);

    // test cilk without reversal
    if (!testcase || testcase == 2) runtest("Cilk   ", &datainfo, testcase_cilk_noreversal);
#endif

    
    return (EXIT_SUCCESS);
}
