
#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>
#include <time.h>
#include <mpi.h>
#include <math.h>

struct timespec t0, t1;

typedef struct knnresult
{
  int    *nidx;    /* Indices (0-based) of nearest neighbors [m-by-k] */
  double *ndist;   /* Distance of nearest neighbors          [m-by-k] */
  int     m;       /* Number of query points                 [scalar] */
  int     k;       /* Number of nearest neighbors            [scalar] */
} knnresult;

typedef struct node
{
	struct node *left;
	struct node *right;
    struct node *parent;
	double *vp;
    double *kids;
    double pdist;
    double mu;
    int    isLeaf;
    int    isLeft;
    int    isRight; 
} node;

typedef struct query
{
    double   *coor;
    knnresult knn;
    int       cnt;
} query;


double* calc_D(double *X, double *Y, int m, int d, int n)
{  
    double *D = (double *)malloc(m * n * sizeof(double));
    double prdct1[ m ];
    double prdct2[ n ];
    
    for(int i = 0; i < m; i++) prdct1[i] = cblas_ddot(d, X+i*d, 1, X+i*d, 1);
    for(int i = 0; i < n; i++) prdct2[i] = cblas_ddot(d, Y+i*d, 1, Y+i*d, 1);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, d, -2, X, d, Y, d, 0, D, n);

    for(int i = 0; i < m; i++)
        for(int j = 0; j < n; j++)
            D[j+i*n] += prdct1[i] + prdct2[j];

    return D;
}


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
int* insertion_Sort(double *a, int size) 
{ 
    int i, loc, j, k;
    double selected;
    int *aPos = (int *)malloc(size * sizeof(int));  
  
    for (i = 0; i < size; i++) 
    { 
        j = i - 1; 
        selected = a[i]; 
  
        /* Find correct location to insert item */
        loc = binary_Search(a, selected, 0, j); 
        
        /* Move elements --> */ 
        while (j >= loc) 
        { 
            a[j+1]    = a[j]; 
            aPos[j+1] = aPos[j];
            j--; 
        } 
        a[j+1]    = selected; 
        aPos[j+1] = i;
    } 

    return aPos;
} 


/* Updates kNN */
void update(double *ndist, double *Pdist, int loop, int k){

    int loc;
    if(loop == 0) for(int z = 0; z < k; z++) ndist[z] = Pdist[z];
    else
    {
        for(int v = 0; v < k; v++)
        {
            loc = binary_Search(ndist, Pdist[v], 0, k);
            if(loc == k - 1) ndist[loc] = Pdist[v];
            else if(loc < k - 1){
                for(int c = k - 1; c > loc; c--) ndist[c] = ndist[c-1];
                ndist[loc] = Pdist[v];
            }
        }
    }

}


/* Returns pointer to tree's root */
node *createVPT(double *S, int n, int d, node *par){

    node *T = (node *)malloc(sizeof(node));
    T -> kids = malloc(d * 3 * sizeof(double));
    T -> vp   = malloc(d     * sizeof(double));
    T -> parent = par;
    
    for(int i = 0; i < d; i++) T -> vp[i] = S[i];
    int bound = 2 * d + 1;
    
    if(n < bound)
    {    
        for(int j = d, j2 = 0; j < d * 3; j++, j2++) T -> kids[j2] = S[j]; 
        T -> isLeaf  = 1;
        return T;
    }
    else T -> isLeaf = 0;

    double *sL    = malloc(n * d * sizeof(double)),
           *sR    = malloc(n * d * sizeof(double)),
           *D     = malloc(n     * sizeof(double)); 
    int    *temp  = malloc(n     * sizeof(int));    
    int     size1 = 0,
            size2 = 0,
            mid;

    D = calc_D(S, T -> vp, n, d, 1);
    temp = insertion_Sort(D, n);
  
    (n % 2 != 0) ? (mid = n / 2) : (mid = (n / 2) + 1);
    T -> mu = D[mid];
  
    for(int i = 1, i2 = 0; i < mid; i++, i2+=d)
    {
        for(int u = 0; u < d; u++) sL[i2+u] = S[d * temp[i] + u];
        size1++;
    }
    for(int i = mid, i2 = 0; i < n; i++, i2+=d)
    {   
        for(int u = 0; u < d; u++) sR[i2+u] = S[d * temp[i] + u];
        size2++;
    } 
 
    T -> left  = createVPT(sL, size1, d, T);
    T -> left  -> isLeft  = 1;

    T -> right = createVPT(sR, size2, d, T);
    T -> right -> isRight = 1;
    
    return T;
}


void checkIntersection(node *T, query p, int d, int k, double *ndist, double farthest);


/* VP tree searcher */
node *searchVPT(node *T, query p, int d, int k, double *ndist){
  
    int loc, intersection;
    double *D = malloc(sizeof(double)), farthest, elemD; 
    D = calc_D(T -> vp, p.coor, 1, d, 1);
    T -> pdist = sqrt(fabs( D[0] ));
    elemD = sqrt(fabs( D[0] ));
  
    if(p.cnt < k && T -> isLeaf != 1)
    { 
        ndist[p.cnt] = elemD;
        p.cnt++;
    }
    else if(p.cnt >= k && T->isLeaf != 1)
    {  
        insertion_Sort(ndist, k);
        loc = binary_Search(ndist, elemD, 0, k);
        if(loc == k - 1) ndist[loc] = elemD;
        else if(loc < k - 1)
        {
            for(int i = k - 1; i > loc; i--) ndist[i] = ndist[i-1];
            ndist[loc] = elemD;
        }
    }
    
    if(T -> isLeaf == 1)
    {  
        insertion_Sort(ndist, p.cnt);
        double *Dtemp = malloc(3 * sizeof(double));
        Dtemp = calc_D(T -> kids, p.coor, 3, d, 1);
        if(p.cnt == k - 1)
            for(int i = 0; i < 3; i++)
            {
                loc = binary_Search(ndist, Dtemp[i], 0, p.cnt);
                if(loc < p.cnt)
                {   
                    if(Dtemp[i] != 0){
                        for(int y = p.cnt - 1; y > loc; y--) ndist[y] = ndist[y-1];
                        ndist[loc] = Dtemp[i];
                    }
                }
            }
    
        if(p.cnt < k)
        {   
            int g = 0;
            ndist[p.cnt++] = elemD;
            insertion_Sort(Dtemp, 3);
            while(p.cnt < k && g < 3) ndist[p.cnt++] = Dtemp[g++];
            insertion_Sort(ndist, k);
        }
    }
    if(T -> isLeaf != 1)
    {
        if(D[0] < T -> mu)
            return searchVPT(T -> left, p, d, k, ndist);
        else return searchVPT(T -> right, p, d, k, ndist);
    }

    farthest = ndist[p.cnt - 1];
    checkIntersection(T, p, d, k, ndist, farthest);
}


/* Climbs the tree upwards checking for intersection */
void checkIntersection(node *T, query p, int d, int k, double *ndist, double farthest){

    int intersection = 0;
    (T->pdist < (T->mu) + farthest && T->pdist > (T->mu) - farthest) ? (intersection = 1) : (intersection = 0);
 
    if( intersection )
    {   
        if((T -> isLeft == 1) && (T -> parent -> right != NULL))
            searchVPT(T -> parent -> right, p, d, k, ndist);
    }
    else if (T -> parent != NULL){
        checkIntersection(T -> parent, p, d, k, ndist, farthest);intersection=0;}
}


/* Spawns new node */
node *newNode(double *nvp, double *nkids, int hasKids, double med, int isLf, int isLft, int isRht, int d){

    node *temp   = (node   *)malloc(    sizeof(node  ));
    temp -> vp   = (double *)malloc(d * sizeof(double));
    
    for(int i = 0; i < d; i++) temp -> vp[i] = nvp[i];
    temp -> mu      = med;
    temp -> isLeaf  = isLf;
    temp -> isLeft  = isLft;
    temp -> isRight = isRht;
    temp -> left  = NULL;
    temp -> right = NULL;

    if( hasKids )
    {
        temp -> kids = (double *)malloc(d * 3 * sizeof(double));
        for(int j = 0; j < 3 * d; j++) temp -> kids[j] = nkids[j];
    }
    return temp;
}


/* Serializes VP tree into list */
int SerializeVPT(node *T, double *list, int *j, int d){

    int y, k, i = *j;
    if(T -> isLeaf == 1)
    {   
        list[i++] = -1.0;
        for(int u = 0; u < d; u++, i++) list[i] = T->vp[u];
        for(int j = 0; j < d * 3; j++, i++) list[i] = T->kids[j];
        list[i++] = T->mu;
        list[i++] = T->isLeaf;
        list[i++] = T->isLeft;
        list[i++] = T->isRight;

        *j = i;
        return i;
    }
    for(int u = 0; u < d; u++, i++) list[i] = T->vp[u];
    list[i++] = T->mu;
    list[i++] = T->isLeaf;
    list[i++] = T->isLeft;
    list[i++] = T->isRight;

    y = SerializeVPT(T -> left , list, &i, d);
    k = SerializeVPT(T -> right, list, &y, d);

    *j = k;
    return k;
}


/* Deserializes tree from list */
node *deSerializeVPT(node *T, double *list, int ptr, int *len, int d){
    
    double val, val2, mu, isLeaf, isLeft, isRight,
          *vp   = malloc(d * sizeof(double)),
          *kids = malloc(d * 3 * sizeof(double));

    node *par = malloc(sizeof(node));
    par = T;

    val = list[(*len)++];
    vp[0] = val;

    if(*len > ptr) return NULL;
    else if(val == -1)
    {   
        for(int i = 0; i < d; i++, (*len)++) vp[i] = list[*len];
        for(int i = 0; i < d * 3; i++, (*len)++) kids[i] = list[*len];
        mu      = list[(*len)++];
        isLeaf  = list[(*len)++];
        isLeft  = list[(*len)++];
        isRight = list[(*len)++];
        T = newNode(vp, kids, 1, mu, isLeaf, isLeft, isRight, d);
        T -> parent = par;
    }
    else
    {    
        for(int i = 1; i < d; i++, (*len)++) vp[i] = list[*len];
        mu      = list[(*len)++];
        isLeaf  = list[(*len)++];
        isLeft  = list[(*len)++];
        isRight = list[(*len)++];
        T = newNode(vp, kids, 0, mu, isLeaf, isLeft, isRight, d);
    
        node *tempPar = malloc(sizeof(node));
        tempPar = T;
        T -> parent = par;
     
        T -> left = deSerializeVPT(T -> left, list, ptr, len, d);
        T -> left -> parent = tempPar;
      
        T -> right = deSerializeVPT(T -> right, list, ptr, len, d);    
        T -> right -> parent = tempPar;
    }
    return T;
}


/* Sends serialized tree */
void VPTsend(node *root, int rank, int pnum, int size, int d){

    MPI_Request req;
    double *list = malloc(size * 10 * sizeof(double));
    int *i = malloc(sizeof(int)), len;
    *i = 0;

    SerializeVPT(root, list, i, d);
    len = *i;

    if(rank != pnum - 1)
    {
        MPI_Isend(&len, 1, MPI_INT, rank+1, rank, MPI_COMM_WORLD, &req);
        MPI_Isend(list, len, MPI_DOUBLE, rank+1, rank, MPI_COMM_WORLD, &req);
    }
    else
    {
        MPI_Isend(&len, 1, MPI_INT, 0, rank, MPI_COMM_WORLD, &req);
        MPI_Isend(list, len, MPI_DOUBLE, 0, rank, MPI_COMM_WORLD, &req);
    }
}


/* Receives VP tree */
void VPTrecv(node *root, int rank, int pnum, int d, MPI_Status stat, MPI_Status stat2){

    int len;
    int *ptr = malloc(sizeof(int));
    *ptr = 0;

    if(rank != 0)
    {   
        MPI_Recv(&len, 1, MPI_INT, rank-1, rank-1, MPI_COMM_WORLD, &stat);
        double rxbuffer[ len ];
        MPI_Recv(&rxbuffer, len, MPI_DOUBLE, rank-1, rank-1, MPI_COMM_WORLD, &stat2);
        root = NULL;
        root = deSerializeVPT(root, rxbuffer, len, ptr, d);
    }
    else if(rank == 0)
    {
        MPI_Recv(&len, 1, MPI_INT, pnum-1, pnum-1, MPI_COMM_WORLD, &stat);
        double rxbuffer[ len ]; 
        MPI_Recv(&rxbuffer, len, MPI_DOUBLE, pnum-1, pnum-1, MPI_COMM_WORLD, &stat2);
        root = NULL;
        root = deSerializeVPT(root, rxbuffer, len, ptr, d);
    }
}


/* Calculates the distributed k-NN with VP tree */
knnresult V2_distrAllkNN(double *X, int m, int d, int k, int rank, int pnum){

    int loop;
    knnresult knn;
    knn.ndist       = malloc((k * m )* sizeof(double));
    node *root      = malloc(        sizeof(node  ));
    double *temp    = malloc((k)     * sizeof(double));
    root -> isLeft  = 0;
    root -> isRight = 0;
    query p;
    p.coor = malloc(d * sizeof(double));
    p.cnt = 0;

    root = createVPT(X, m, d, NULL);
    MPI_Status stat, stat2;

    for(loop = 0; loop < pnum; loop++)
    {
        for(int e = 0, e2 = 0; e < m; e++, e2 += d) 
        {   
            p.cnt = 0;
            for(int u = 0; u < d; u++) p.coor[u] = X[d*e+u];
            searchVPT(root, p, d, k, temp);
            update(knn.ndist + k * e, temp, loop, k);
            temp = malloc(k * sizeof(double));
        }
        VPTsend(root, rank, pnum, m, d);
        VPTrecv(root, rank, pnum, d, stat, stat2);
    }
}


void main(int argc, char **argv){

    int pnum, rank;
    MPI_Init( &argc, &argv );
    MPI_Comm_size(MPI_COMM_WORLD, &pnum);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Status stat;
    MPI_Request req;
    int m = atoi( argv[1] ),
        d = atoi( argv[2] ),
        k = atoi( argv[3] );
        
    int loop, chunk = m * d / pnum, len = m / pnum, prcs_m = m / pnum;
    double *Xbuffer = malloc(chunk * sizeof(double));

    if(rank == 0)
    {
        double *X = (double *)malloc(m * d * sizeof(double)); 
        for(int i = 0; i < m * d; i++) X[i] = ((double)(rand())) / ((double)(RAND_MAX));
        for(int i = 0; i < chunk; i++) Xbuffer[i] = X[i];
        for(int p = 1; p < pnum; p++)
            MPI_Isend(X+p*chunk, chunk, MPI_DOUBLE, p, 0, MPI_COMM_WORLD, &req);
    }
    else MPI_Recv(Xbuffer, chunk, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &stat);
    
    clock_gettime(CLOCK_REALTIME, &t0);
    V2_distrAllkNN(Xbuffer, prcs_m, d, k, rank, pnum);
    clock_gettime(CLOCK_REALTIME, &t1);
   
   
        double duration = ((t1.tv_sec-t0.tv_sec)*1000000+(t1.tv_nsec-t0.tv_nsec)/1000)/1000000.0;
        printf("~ Duration: %f sec\n", duration);
    

    free(Xbuffer);
    MPI_Finalize();
}

