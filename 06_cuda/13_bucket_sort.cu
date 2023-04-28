#include <cstdio>
#include <cstdlib>
#include <vector>


__global__ void add(int* d_bucket, int* d_key, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        atomicAdd(&d_bucket[d_key[i]], 1);
    }
}

__global__ void sort(int* d_bucket, int* d_key, int n, int range) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = 0;
    for (int k=0; k<range; k++) {
        for (int l=0; l<d_bucket[k]; l++) {
            if (i == j) {
                d_key[i] = k;
            }
            j++;
        }
    }
}


int main() {
  int n = 50;
  int range = 5;

  int *d_key;
  cudaMallocManaged(&d_key, n*sizeof(int));

  for (int i=0; i<n; i++) {
    d_key[i] = rand() % range;
    printf("%d ", d_key[i]);
  }
  printf("\n");

  int *d_bucket;
  
  cudaMallocManaged(&d_bucket, n*sizeof(int));
  for (int i=0; i<range; i++) {
    d_bucket[i] = 0;
  }

  add<<<1, n>>>(d_bucket, d_key, n);
  cudaDeviceSynchronize();

  sort<<<1, n>>>(d_bucket, d_key, n, range);
  cudaDeviceSynchronize();
  
  for (int i=0; i<n; i++) {
    printf("%d ",d_key[i]);
  }
  printf("\n");
  cudaFree(d_bucket);
  cudaFree(d_key);
}
