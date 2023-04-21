#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>

int main() {
  const int N = 8;
  float x[N], y[N], m[N], fx[N], fy[N], is[N], tmp[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
    is[i] = i;
    tmp[i] = 0;
  }
  __m256 xvec = _mm256_load_ps(x);
  __m256 yvec = _mm256_load_ps(y);
  __m256 mvec = _mm256_load_ps(m);
  __m256 isvec = _mm256_load_ps(is);
  __m256 zeros = _mm256_setzero_ps();

  for(int i=0; i<N; i++) {
    __m256 xi = _mm256_set1_ps(x[i]);
    __m256 yi = _mm256_set1_ps(y[i]);
    __m256 ivec = _mm256_set1_ps(i);
    __m256 mask = _mm256_cmp_ps(ivec, isvec, _CMP_NEQ_OQ);

    __m256 rxvec = _mm256_sub_ps(xi, xvec);
    __m256 ryvec = _mm256_sub_ps(yi, yvec);
    __m256 rsqvec = _mm256_add_ps(_mm256_mul_ps(rxvec, rxvec), _mm256_mul_ps(ryvec, ryvec));
    __m256 rrvec = _mm256_blendv_ps(zeros, _mm256_rsqrt_ps(rsqvec), mask);
    __m256 fxivec = _mm256_mul_ps(_mm256_mul_ps(rxvec, mvec), _mm256_mul_ps(_mm256_mul_ps(rrvec, rrvec), rrvec));
    __m256 fyivec = _mm256_mul_ps(_mm256_mul_ps(ryvec, mvec), _mm256_mul_ps(_mm256_mul_ps(rrvec, rrvec), rrvec));

    fxivec = _mm256_hadd_ps(fxivec, _mm256_permute2f128_ps(fxivec, fxivec, 1));
    fxivec = _mm256_hadd_ps(fxivec, fxivec);
    fxivec = _mm256_hadd_ps(fxivec, fxivec);
    _mm256_store_ps(tmp, fxivec);
    fx[i] = -1 * tmp[0];

    fyivec = _mm256_hadd_ps(fyivec, _mm256_permute2f128_ps(fyivec, fyivec, 1));
    fyivec = _mm256_hadd_ps(fyivec, fyivec);
    fyivec = _mm256_hadd_ps(fyivec, fyivec);
    _mm256_store_ps(tmp, fyivec); 
    fy[i] = -1 * tmp[0];

    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
