
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "cblas.h"
#include <math.h>

typedef struct knnresult
{
  int    *nidx;    /* Indices (0-based) of nearest neighbors [m-by-k] */
  double *ndist;   /* Distance of nearest neighbors          [m-by-k] */
  int     m;       /* Number of query points                 [scalar] */
  int     k;       /* Number of nearest neighbors            [scalar] */
} knnresult;


/* Returns the location to insert element in list */
int binary_Search(double *a, double item, int low, int high) 
{ 
    if (high <= low) 
        return (item > a[low]) ? (low + 1) : low; 
  
    int mid = (low + high) / 2; 
  
    if(item == a[mid]) 
        return mid+1; 
  
    if(item > a[mid]) 
        return binary_Search(a, item, mid+1, high); 
    else
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


knnresult kNN(double *X, double *Y, int n, int m, int d, int k){

   /*   int *TMP;
        int tmp;
        TMP = X;
        X   = Y;    <---- USE THIS IF YOU WANT TO FIND K-NN OF Y ARRAY, WORKS FOR X BY DEFAULT
        Y   = TMP;
        tmp = n;
        n   = m;
        m   = tmp;   */

    knnresult knn;
    knn.ndist = malloc(k * m * sizeof(double));
    knn.nidx  = malloc(k * m * sizeof(int));
    knn.m     = m;
    knn.k     = k;

    double *ndist = knn.ndist;
    int    *nidx  = knn.nidx;

    double pos, elemD;
    int loc, cnt;

    double *D = (double *)malloc(m * n * sizeof(double));
    double sumx[ n ],
           sumy[ m ];


    for(int i = 0; i < n; i++)
        sumx[i] = cblas_ddot(d, X+i*d, 1, X+i*d, 1);
    
    for(int i = 0; i < m; i++)
        sumy[i] = cblas_ddot(d, Y+i*d, 1, Y+i*d, 1);
    
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, m, d, -2, X, d, Y, d, 0, D, m);


    for(int i = 0; i < n; i++)
    {   
        cnt = 0;
        for(int j = 0; j < m; j++)
        {
            D[j+i*m] += sumx[i] + sumy[j];
            elemD =  sqrt(fabs( D[j+i*m] ));
            pos = j;
          
            if(cnt < k)
            {   
                ndist[i*k+cnt] = elemD;
                nidx [i*k+cnt] = pos;
                if(cnt == k - 1) insertion_Sort(ndist+i*k, nidx+i*k, k);
                cnt++;
            }
            else 
            {  
                loc = binary_Search(ndist+i*k, elemD, 0, k); 
                if(loc == k - 1) 
                {
                    ndist[i*k+loc] = elemD;
                    nidx [i*k+loc] = pos;
                }
                else if(loc < k - 1)
                {
                    for(int z = i*k+k-1; z > i*k+loc; z--) 
                    {
                        ndist[z] = ndist[z-1];
                        nidx [z] = nidx [z-1];
                    }
                    ndist[i*k+loc] = elemD;
                    nidx [i*k+loc] = pos;
                }
            }
        }
    }
    return knn;
}
      

void main(int argc, char **argv){

    int n = atoi(argv[1]),
        m = atoi(argv[2]),
        d = atoi(argv[3]),
        k = atoi(argv[4]);

    knnresult knn;
    knn.nidx  = malloc(n * k * sizeof(int   ));
    knn.ndist = malloc(n * k * sizeof(double));
    double *X = malloc(n * d * sizeof(double)),
           *Y = malloc(m * d * sizeof(double));

    for(int i = 0; i < n*d; i++) X[i] = ((double)(rand())) / ((double)(RAND_MAX));
    for(int i = 0; i < m*d; i++) Y[i] = ((double)(rand())) / ((double)(RAND_MAX));
    
    kNN(X, Y, n, m, d, k);

}
      

