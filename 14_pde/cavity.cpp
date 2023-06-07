#include <cstdlib>
#include <cstdio>
#include <vector>
#include <chrono>
#include <immintrin.h>
using namespace std;
typedef vector<vector<float> > matrix;

int main() {
    const int nx = 162;
    const int ny = 162;
    int nt = 100;
    int nit = 50;
    float dx = 2. / (nx - 1);
    float dy = 2. / (ny - 1);
    float dt = 0.001;
    float rho = 1;
    float nu = 0.02;
    vector<float> x(nx);
    vector<float> y(ny);
    
    for (int i=0; i<nx; i++) {
        x[i] = i * dx;
    }
    for (int j=0; j<ny; j++) {
        y[j] = j * dy;
    }

    matrix u(ny, vector<float>(nx, 0));
    matrix v(ny, vector<float>(nx, 0));
    matrix b(ny, vector<float>(nx, 0));
    matrix p(ny, vector<float>(nx, 0));
    matrix un(ny, vector<float>(nx, 0));
    matrix vn(ny, vector<float>(nx, 0));
    matrix pn(ny, vector<float>(nx, 0));

    for (int n=0; n<nt; n++) {
        auto tic = chrono::steady_clock::now();
#pragma acc parallel loop
        for (int j=1; j<ny-1; j++) {
            for (int i=1; i<nx-1; i++) {
                b[j][i] = rho * (1 / dt * (
                    (u[j][i+1] - u[j][i-1]) / (2 * dx) + (v[j+1][i] - v[j-1][i]) / (2 * dy)
                ) - (
                    ((u[j][i+1] - u[j][i-1]) / (2 * dx)) * ((u[j][i+1] - u[j][i-1]) / (2 * dx))
                ) - 2 * (
                    (u[j+1][i] - u[j-1][i]) / (2 * dy) * (v[j][i+1] - v[j][i-1]) / (2 * dx)
                ) - (
                    ((v[j+1][i] - v[j-1][i]) / (2 * dy)) * ((v[j+1][i] - v[j-1][i]) / (2 * dy))
                ));
            }
        }
#pragma acc parallel loop
        for (int it=0; it<nit; it++){
            for (int j=0; j<ny; j++) {
                for (int i=0; i<nx; i++) {
                    pn[j][i] = p[j][i];
                }
            }
#pragma omp parallel for
            for (int j=1; j<ny-1; j++) {
                for (int i=1; i<nx-1; i++) {
                    p[j][i] = (
                        dy*dy * (pn[j][i+1] + pn[j][i-1]) + dx*dx * (pn[j+1][i] + pn[j-1][i]) - b[j][i] * dx*dx * dy*dy
                    ) / (
                        2 * (dx*dx + dy*dy)
                    );
                }
            }
            for (int j=1; j<ny-1; j++) {
                p[j][nx-1] = p[j][nx-2];
                p[j][0] = p[j][1];
            }
            for (int i=1; i<nx-1; i++) {
                p[0][i] = p[1][i];
                p[ny-1][i] = p[ny-2][i];
            }
        }

#pragma acc parallel loop
        for (int j=0; j<ny; j++) {
            for (int i=0; i<nx; i++) {
                un[j][i] = u[j][i];
                vn[j][i] = v[j][i];
            }
        }


        for (int j=1; j<ny-1; j++) {
            for (int i=1; i<nx-1; i+=8) {
                __m256 uji = _mm256_loadu_ps(&un[j][i]);
                __m256 ujim1 = _mm256_loadu_ps(&un[j][i-1]);
                __m256 ujip1 = _mm256_loadu_ps(&un[j][i+1]);
                __m256 ujm1i = _mm256_loadu_ps(&un[j-1][i]);
                __m256 ujp1i = _mm256_loadu_ps(&un[j+1][i]);
                __m256 pjim1 = _mm256_loadu_ps(&p[j][i-1]);
                __m256 pjip1 = _mm256_loadu_ps(&p[j][i+1]);
                __m256 uvec;
                uvec = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(_mm256_sub_ps(_mm256_sub_ps(
                                    uji,
                                    _mm256_mul_ps(uji, _mm256_mul_ps(_mm256_div_ps(_mm256_set1_ps(dt), _mm256_set1_ps(dx)), _mm256_sub_ps(uji, ujim1)))
                                ),
                                _mm256_mul_ps(uji, _mm256_mul_ps(_mm256_div_ps(_mm256_set1_ps(dt), _mm256_set1_ps(dy)), _mm256_sub_ps(uji, ujm1i)))
                            ),
                            _mm256_mul_ps(_mm256_div_ps(_mm256_set1_ps(dt), (_mm256_mul_ps(_mm256_set1_ps(2), _mm256_mul_ps(_mm256_set1_ps(rho), _mm256_set1_ps(dx))))),  _mm256_sub_ps(pjip1, pjim1))
                        ),
                        _mm256_mul_ps(_mm256_div_ps(_mm256_mul_ps(_mm256_set1_ps(nu), _mm256_set1_ps(dt)), _mm256_mul_ps(_mm256_set1_ps(dx), _mm256_set1_ps(dx))), _mm256_sub_ps(ujip1, _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(2), uji), ujim1)))
                    ),
                    _mm256_mul_ps(_mm256_div_ps(_mm256_mul_ps(_mm256_set1_ps(nu), _mm256_set1_ps(dt)), _mm256_mul_ps(_mm256_set1_ps(dy), _mm256_set1_ps(dy))), _mm256_sub_ps(ujp1i, _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(2), uji), ujm1i)))
                );
                _mm256_storeu_ps(&u[j][i], uvec);
                
                __m256 vji = _mm256_loadu_ps(&vn[j][i]);
                __m256 vjim1 = _mm256_loadu_ps(&vn[j][i-1]);
                __m256 vjip1 = _mm256_loadu_ps(&vn[j][i+1]);
                __m256 vjm1i = _mm256_loadu_ps(&vn[j-1][i]);
                __m256 vjp1i = _mm256_loadu_ps(&vn[j+1][i]);
                __m256 pjm1i = _mm256_loadu_ps(&p[j-1][i]);
                __m256 pjp1i = _mm256_loadu_ps(&p[j+1][i]);
                __m256 vvec;
                vvec = _mm256_add_ps(_mm256_add_ps(_mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(
                                    vji,
                                    _mm256_mul_ps(_mm256_mul_ps(vji, _mm256_div_ps(_mm256_set1_ps(dt), _mm256_set1_ps(dx))), _mm256_sub_ps(vji, vjim1))
                                ),
                                _mm256_mul_ps(_mm256_mul_ps(vji, _mm256_div_ps(_mm256_set1_ps(dt), _mm256_set1_ps(dy))), _mm256_sub_ps(vji, vjm1i))
                            ),
                            _mm256_mul_ps(_mm256_div_ps(_mm256_set1_ps(dt), _mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(2), _mm256_set1_ps(rho)), _mm256_set1_ps(dx))), _mm256_sub_ps(pjp1i, pjm1i))
                        ),
                        _mm256_mul_ps(_mm256_div_ps(_mm256_mul_ps(_mm256_set1_ps(nu), _mm256_set1_ps(dt)), _mm256_mul_ps(_mm256_set1_ps(dx), _mm256_set1_ps(dx))), _mm256_sub_ps(vjip1, _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(2), vji), vjim1)))
                    ),
                    _mm256_mul_ps(_mm256_div_ps(_mm256_mul_ps(_mm256_set1_ps(nu), _mm256_set1_ps(dt)), _mm256_mul_ps(_mm256_set1_ps(dy), _mm256_set1_ps(dy))), _mm256_sub_ps(vjp1i, _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(2), vji), vjm1i)))
                );
                _mm256_storeu_ps(&v[j][i], vvec);
            }
        }

        for (int j=1; j<ny-1; j++) {
            u[j][0] = 0;
            u[j][nx-1] = 0;
            v[j][0] = 0;
            v[j][nx-1] = 0;
        }
        for (int i=1; i<nx-1; i++) {
            u[0][i] = 0;
            u[nx-1][i] = 1;
            v[0][i] = 0;
            v[nx-1][i] = 0;
        }
        auto toc = chrono::steady_clock::now();
        double time = chrono::duration<double>(toc - tic).count();
        printf("step=%d: %lf [s]\n", n, time);
    }
}
