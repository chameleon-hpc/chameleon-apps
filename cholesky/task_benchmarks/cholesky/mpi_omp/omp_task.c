
#include "common.h"

static void omp_potrf(double * const A, int ts, int ld);
static void omp_trsm(double *A, double *B, int ts, int ld);
static void omp_gemm(double *A, double *B, double *C, int ts, int ld);
static void omp_syrk(double *A, double *B, int ts, int ld);

double do_cholesky_ser(const int ts, const int nt, double* A[nt][nt])
{
    double time = get_time();
#pragma omp parallel
#pragma omp single
    for (int k = 0; k < nt; k++) {
#pragma omp task depend(out:A[k][k])
{
        omp_potrf(A[k][k], ts, ts);
#ifdef DEBUG
        if (mype == 0) printf("potrf:out:A[%d][%d]\n", k, k);
#endif
}
        for (int i = k + 1; i < nt; i++) {
#pragma omp task depend(in:A[k][k]) depend(out:A[k][i])
{
            omp_trsm(A[k][k], A[k][i], ts, ts);
#ifdef DEBUG
            if (mype == 0) printf("trsm :in:A[%d][%d]:out:A[%d][%d]\n", k, k, k, i);
#endif
}
        }
        for (int i = k + 1; i < nt; i++) {
            for (int j = k + 1; j < i; j++) {
#pragma omp task depend(in:A[k][i], A[k][j]) depend(out:A[j][i])
{
                omp_gemm(A[k][i], A[k][j], A[j][i], ts, ts);
#ifdef DEBUG
                if (mype == 0) printf("gemm :in:A[%d][%d]:A[%d][%d]:out:A[%d][%d]\n", k, i, k, j, j, i);
#endif
}
            }
#pragma omp task depend(in:A[k][i]) depend(out:A[i][i])
{
            omp_syrk(A[k][i], A[i][i], ts, ts);
#ifdef DEBUG
            if (mype == 0) printf("syrk :in:A[%d][%d]:out:A[%d][%d]\n", k, i, i, i);
#endif
}
        }
    }
#pragma omp taskwait
    return get_time() - time;
}

static void omp_potrf(double * const A, int ts, int ld)
{
    static int INFO;
    static const char L = 'L';
    LAPACKE_dpotrf(LAPACK_COL_MAJOR, L, ts, A, ts);
    /* dpotrf_(&L, &ts, A, &ld, &INFO); */
}
static void omp_trsm(double *A, double *B, int ts, int ld)
{
    static char LO = 'L', TR = 'T', NU = 'N', RI = 'R';
    static double DONE = 1.0;
    cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit, ts, ts, DONE, A, ld, B, ld);
    /* dtrsm_(&RI, &LO, &TR, &NU, &ts, &ts, &DONE, A, &ld, B, &ld ); */
}
static void omp_gemm(double *A, double *B, double *C, int ts, int ld)
{
    static const char TR = 'T', NT = 'N';
    static double DONE = 1.0, DMONE = -1.0;
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, ts, ts, ts, DMONE, A, ld, B, ld, DONE, C, ld);
    /* dgemm_(&NT, &TR, &ts, &ts, &ts, &DMONE, A, &ld, B, &ld, &DONE, C, &ld); */
}
static void omp_syrk(double *A, double *B, int ts, int ld)
{
    static char LO = 'L', NT = 'N';
    static double DONE = 1.0, DMONE = -1.0;
    cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans, ts, ts, DMONE, A, ld, DONE, B, ld);
    /* dsyrk_(&LO, &NT, &ts, &ts, &DMONE, A, &ld, &DONE, B, &ld ); */
}
