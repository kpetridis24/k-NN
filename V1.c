
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "cblas.h"
#include <math.h>
#include <mpi.h>

void kNN(double *X, double *Y, int m, int d, int n, double *ndist, int *nidx, int k, int loop, int pnum);
void V1(double *Xbuffer, int m, int d, int k, int pid, int num_prcss);
void update(double *ndist, int *nidx, double elemD, int loc, int pos, int k);
int  binary_Search  (double a[], double item, int low, int high);
void insertion_Sort (double *a, int *b, int size);
struct timespec t0, t1;

struct knnresult
{
  int    *nidx;    /* Indices (0-based) of nearest neighbors [m-by-k] */
  double *ndist;   /* Distance of nearest neighbors          [m-by-k] */
  int     m;       /* Number of query points                 [scalar] */
  int     k;       /* Number of nearest neighbors            [scalar] */

};


int main(int argc, char **argv){
    
    int pnum, pid;
    MPI_Init( &argc, &argv );
    MPI_Comm_size(MPI_COMM_WORLD, &pnum);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Status stat;
    int m = atoi( argv[1] ),          
        d = atoi( argv[2] ),            
        k = atoi( argv[3] ),
        err;   

    int chunk  = m * d / pnum,
        prcs_m = m     / pnum;  
        
    double *Xbuffer = malloc(chunk * sizeof(double));

    if(pid == 0)
    {
        double *X = (double *)malloc(m * d * sizeof(double)); 
        for(int i = 0; i < m * d; i++) X[i] = ((double)(rand())) / ((double)(RAND_MAX));
        for(int i = 0; i < chunk; i++) Xbuffer[i] = X[i];
        for(int p = 1; p < pnum; p++)
           MPI_Send(X+p*chunk, chunk, MPI_DOUBLE, p, 0, MPI_COMM_WORLD);
    }
    else MPI_Recv(Xbuffer, chunk, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &stat);
    
    V1(Xbuffer, m, d, k, pid, pnum);
    
    MPI_Finalize();
    return 0;
}


/* Calculates the distributed KNN across MPI processes */
void V1(double *Xbuffer, int m, int d, int k, int pid, int num_prcss)
{
    int loop,
        err=0,
        xchunk   = m * d / num_prcss, 
        prcs_m   = m     / num_prcss;
       
    MPI_Status recvstat;
    MPI_Request req, req2;
    int f1=0, f2=0;
    struct knnresult knn;
    knn.ndist = (double *)malloc((prcs_m * k) * sizeof(double));
    knn.nidx  = (int    *)malloc((prcs_m * k) * sizeof(int   ));

    double *D = (double *)malloc(prcs_m * prcs_m * sizeof(double)),
           *Ybuffer = malloc((xchunk+1) * sizeof(double));

    for(int a = 0; a < xchunk; a++) Ybuffer[a] = Xbuffer[a];       
    clock_gettime(CLOCK_REALTIME, &t0);

    /* Calculate and update k-NN, Send, Receive */
    for(loop = 0; loop < num_prcss; loop++)
    {   
        kNN(Xbuffer, Ybuffer, prcs_m, d, prcs_m, knn.ndist, knn.nidx, k, loop, num_prcss);

        if(pid != num_prcss - 1)
            MPI_Isend(Ybuffer, xchunk, MPI_DOUBLE, pid+1, pid, MPI_COMM_WORLD, &req);
        else MPI_Isend(Ybuffer, xchunk, MPI_DOUBLE, 0, pid, MPI_COMM_WORLD, &req);
        
        if(pid != 0)
            MPI_Recv(Ybuffer, xchunk, MPI_DOUBLE, pid-1, pid-1, MPI_COMM_WORLD, &recvstat);
        if(pid == 0)
            MPI_Recv(Ybuffer, xchunk, MPI_DOUBLE, num_prcss-1, num_prcss-1, MPI_COMM_WORLD, &recvstat);
            
    }

    clock_gettime(CLOCK_REALTIME, &t1);
    double duration = ((t1.tv_sec-t0.tv_sec)*1000000+(t1.tv_nsec-t0.tv_nsec)/1000)/1000000.0;
    printf("~ Duration: %f sec\n", duration);
    
}


void update(double *ndist, int *nidx, double elemD, int loc, int pos, int k){
    
    if(loc == k - 1) 
    {   
        ndist[loc] = elemD;
        nidx [loc] = pos;
    }
    else if(loc < k - 1)
    {
        for(int z = k - 1; z > loc; z--) 
        {
            ndist[z] = ndist[z-1];
            nidx [z] = nidx [z-1];
        }
        ndist[loc] = elemD;
        nidx [loc] = pos;
    }
}


/* Calculates distance and saves the kNN */
void kNN(double *X, double *Y, int m, int d, int n, double *ndist, int *nidx, int k, int loop, int pnum){

    int loc, offst, cnt, pos;
    double *D = (double *)malloc(m * n * sizeof(double)),
           *sumx = malloc(m*d * sizeof(double)),
           *sumy = malloc(n*d * sizeof(double)),
            elemD;
       
    for(int i = 0; i < m; i++) sumx[i] = cblas_ddot(d, X+i*d, 1, X+i*d, 1);
    for(int j = 0; j < n; j++) sumy[j] = cblas_ddot(d, Y+j*d, 1, Y+j*d, 1);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, d, -2, X, d, Y, d, 0, D, n);

    for(int i = 0; i < m; i++)
    {   
        cnt   = 0;
        offst = i * k;

        for(int j = 0; j < n; j++)
        {
            D[j+i*n] += sumx[i] + sumy[j];
            elemD = sqrt(fabs( D[j+i*n] ));
            pos = j;

            if(loop == 0)
            {
                if(cnt < k)
                {
                    ndist[offst + cnt] = elemD;
                    nidx [offst + cnt] = pos;
                    if(cnt == k - 1) insertion_Sort(ndist + offst, nidx + offst, k);
                    cnt++;
                }
                else
                {
                    loc = binary_Search(ndist + offst, elemD, 0, k);
                    update(ndist + offst, nidx + offst, elemD, loc, pos, k);
                }
            }
            else 
            {   
                loc = binary_Search(ndist + offst, elemD, 0, k);
                update(ndist + offst, nidx + offst, elemD, loc, pos, k);
            }
        }
    }
}


/* Binary search */
int binary_Search(double a[], double item, int low, int high){

    if (high <= low) 
        return (item > a[low])?  (low + 1): low; 
  
    int mid = (low + high)/2; 
  
    if(item == a[mid]) 
        return mid+1; 
  
    if(item > a[mid]) 
        return binary_Search(a, item, mid+1, high); 

    return binary_Search(a, item, low, mid-1); 
} 
  

/* Binary insertion sort */
void insertion_Sort(double *a, int *b, int size) 
{ 
    int i, loc, j, k;
    double selected;
  
    for (i = 0; i < size; i++) 
    { 
        j = i - 1; 
        selected = a[i]; 
  
        loc = binary_Search(a, selected, 0, j); 
        
        while (j >= loc) 
        { 
            a[j+1] = a[j]; 
            b[j+1] = b[j];
            j--; 
        } 
        a[j+1] = selected; 
        b[j+1] = i;
    } 
} 



