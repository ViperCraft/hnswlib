#pragma once
#include "hnswlib.h"

namespace hnswlib {

static float
InnerProduct(const void *pVect1, const void *pVect2, const void *qty_ptr) {
    size_t qty = *((size_t *) qty_ptr);
    float res = 0;
    for (unsigned i = 0; i < qty; i++) {
        res += ((float *) pVect1)[i] * ((float *) pVect2)[i];
    }
    return res;
}

static float
InnerProductDistance(const void *pVect1, const void *pVect2, const void *qty_ptr) {
    return 1.0f - InnerProduct(pVect1, pVect2, qty_ptr);
}

__attribute__((target("avx")))
static float
InnerProductSIMD16ExtAVX_ALIGNED(const void *pVect1v, const void *pVect2v, const void *pEnd1v ) {
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    const float *pEnd1 = (float*)pEnd1v;

    __m256 v1, v2;
    // server processors had much more underloading ALU than LS buffers
    // for using less dependency(two sums) will show me better results
    __m256 sum = _mm256_set1_ps(0), sum2 = _mm256_set1_ps(0);

    while (pVect1 < pEnd1) {
        v1 = _mm256_load_ps(pVect1);
        pVect1 += 8;
        v2 = _mm256_loadu_ps(pVect2);
        pVect2 += 8;
        sum = _mm256_add_ps(sum, _mm256_mul_ps(v1, v2));

        v1 = _mm256_load_ps(pVect1);
        pVect1 += 8;
        v2 = _mm256_loadu_ps(pVect2);
        pVect2 += 8;
        sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(v1, v2));
    }

    return 1.f - _mm256_reduce_add_ps(_mm256_add_ps(sum, sum2));
}

__attribute__((target("avx,fma")))
static float
InnerProductSIMD16ExtAVXFMA_ALIGNED(const void *pVect1v, const void *pVect2v, const void *pEnd1v ) {
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    const float *pEnd1 = (float*)pEnd1v;

    __m256 v1, v2;
    // server processors had much more underloading ALU than LS buffers
    // for using less dependency(two sums) will show me better results
    __m256 sum = _mm256_set1_ps(0), sum2 = _mm256_set1_ps(0);

    while (pVect1 < pEnd1) {
        v1 = _mm256_load_ps(pVect1);
        v2 = _mm256_loadu_ps(pVect2);
        sum = _mm256_fmadd_ps(v1, v2, sum);

        v1 = _mm256_load_ps(pVect1 + 8);
        v2 = _mm256_loadu_ps(pVect2 + 8);
        sum2 = _mm256_fmadd_ps(v1, v2, sum2);

        pVect1 += 16;
        pVect2 += 16;
    }

    return 1.f - _mm256_reduce_add_ps(_mm256_add_ps(sum, sum2));
}

__attribute__((target("default")))
static float
InnerProductSIMD16Ext_ALIGNED(const void *pVect1v, const void *pVect2v, const void *pEnd1v) {
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    const float *pEnd1 = (float*)pEnd1v;

    __m128 v1, v2;
    __m128 sum = _mm_set1_ps(0);

    while (pVect1 < pEnd1) {
        //_mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);
        v1 = _mm_load_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum = _mm_add_ps(sum, _mm_mul_ps(v1, v2));

        v1 = _mm_load_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum = _mm_add_ps(sum, _mm_mul_ps(v1, v2));

        v1 = _mm_load_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum = _mm_add_ps(sum, _mm_mul_ps(v1, v2));

        v1 = _mm_load_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum = _mm_add_ps(sum, _mm_mul_ps(v1, v2));
    }

    sum = _mm_hadd_ps (sum, sum);
    sum = _mm_hadd_ps (sum, sum);
    return  1.f - _mm_cvtss_f32 (sum);
}

__attribute__((target("avx512f")))
static float
InnerProductSIMD16ExtAVX3_ALIGNED(const void *pVect1v, const void *pVect2v, const void *pEnd1v ) {
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    const float *pEnd1 = (float*)pEnd1v;

    __m512 v1, v2;
    __m512 sum = _mm512_set1_ps(0);

    while (pVect1 < pEnd1) {
        v1 = _mm512_load_ps(pVect1);
        pVect1 += 16;
        v2 = _mm512_loadu_ps(pVect2);
        pVect2 += 16;
        sum = _mm512_fmadd_ps(v1, v2, sum);
    }

    return 1.f - _mm512_reduce_add_ps(sum);
}

#if defined(USE_AVX)

// Favor using AVX if available.
static float
InnerProductSIMD4ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float PORTABLE_ALIGN32 TmpRes[8];
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);

    size_t qty16 = qty / 16;
    size_t qty4 = qty / 4;

    const float *pEnd1 = pVect1 + 16 * qty16;
    const float *pEnd2 = pVect1 + 4 * qty4;

    __m256 sum256 = _mm256_set1_ps(0);

    while (pVect1 < pEnd1) {
        //_mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);

        __m256 v1 = _mm256_loadu_ps(pVect1);
        pVect1 += 8;
        __m256 v2 = _mm256_loadu_ps(pVect2);
        pVect2 += 8;
        sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v1, v2));

        v1 = _mm256_loadu_ps(pVect1);
        pVect1 += 8;
        v2 = _mm256_loadu_ps(pVect2);
        pVect2 += 8;
        sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v1, v2));
    }

    __m128 v1, v2;
    __m128 sum_prod = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));

    while (pVect1 < pEnd2) {
        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
    }

    _mm_store_ps(TmpRes, sum_prod);
    float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
    return sum;
}

static float
InnerProductDistanceSIMD4ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    return 1.0f - InnerProductSIMD4ExtAVX(pVect1v, pVect2v, qty_ptr);
}

#endif

#if defined(USE_SSE)

static float
InnerProductSIMD4ExtSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float PORTABLE_ALIGN32 TmpRes[8];
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);

    size_t qty16 = qty / 16;
    size_t qty4 = qty / 4;

    const float *pEnd1 = pVect1 + 16 * qty16;
    const float *pEnd2 = pVect1 + 4 * qty4;

    __m128 v1, v2;
    __m128 sum_prod = _mm_set1_ps(0);

    while (pVect1 < pEnd1) {
        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
    }

    while (pVect1 < pEnd2) {
        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
    }

    _mm_store_ps(TmpRes, sum_prod);
    float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];

    return sum;
}

static float
InnerProductDistanceSIMD4ExtSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    return 1.0f - InnerProductSIMD4ExtSSE(pVect1v, pVect2v, qty_ptr);
}

#endif


#if defined(USE_AVX512)

static float
InnerProductSIMD16ExtAVX512(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float PORTABLE_ALIGN64 TmpRes[16];
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);

    size_t qty16 = qty / 16;


    const float *pEnd1 = pVect1 + 16 * qty16;

    __m512 sum512 = _mm512_set1_ps(0);

    size_t loop = qty16 / 4;

    while (loop--) {
        __m512 v1 = _mm512_loadu_ps(pVect1);
        __m512 v2 = _mm512_loadu_ps(pVect2);
        pVect1 += 16;
        pVect2 += 16;

        __m512 v3 = _mm512_loadu_ps(pVect1);
        __m512 v4 = _mm512_loadu_ps(pVect2);
        pVect1 += 16;
        pVect2 += 16;

        __m512 v5 = _mm512_loadu_ps(pVect1);
        __m512 v6 = _mm512_loadu_ps(pVect2);
        pVect1 += 16;
        pVect2 += 16;

        __m512 v7 = _mm512_loadu_ps(pVect1);
        __m512 v8 = _mm512_loadu_ps(pVect2);
        pVect1 += 16;
        pVect2 += 16;

        sum512 = _mm512_fmadd_ps(v1, v2, sum512);
        sum512 = _mm512_fmadd_ps(v3, v4, sum512);
        sum512 = _mm512_fmadd_ps(v5, v6, sum512);
        sum512 = _mm512_fmadd_ps(v7, v8, sum512);
    }

    while (pVect1 < pEnd1) {
        __m512 v1 = _mm512_loadu_ps(pVect1);
        __m512 v2 = _mm512_loadu_ps(pVect2);
        pVect1 += 16;
        pVect2 += 16;
        sum512 = _mm512_fmadd_ps(v1, v2, sum512);
    }

    float sum = _mm512_reduce_add_ps(sum512);
    return sum;
}

static float
InnerProductDistanceSIMD16ExtAVX512(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    return 1.0f - InnerProductSIMD16ExtAVX512(pVect1v, pVect2v, qty_ptr);
}

#endif

#if defined(USE_AVX)

static float
InnerProductSIMD16ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float PORTABLE_ALIGN32 TmpRes[8];
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);

    size_t qty16 = qty / 16;


    const float *pEnd1 = pVect1 + 16 * qty16;

    __m256 sum256 = _mm256_set1_ps(0);

    while (pVect1 < pEnd1) {
        //_mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);

        __m256 v1 = _mm256_loadu_ps(pVect1);
        pVect1 += 8;
        __m256 v2 = _mm256_loadu_ps(pVect2);
        pVect2 += 8;
        sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v1, v2));

        v1 = _mm256_loadu_ps(pVect1);
        pVect1 += 8;
        v2 = _mm256_loadu_ps(pVect2);
        pVect2 += 8;
        sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v1, v2));
    }

    _mm256_store_ps(TmpRes, sum256);
    float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];

    return sum;
}

static float
InnerProductDistanceSIMD16ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    return 1.0f - InnerProductSIMD16ExtAVX(pVect1v, pVect2v, qty_ptr);
}

#endif

#if defined(USE_SSE)

static float
InnerProductSIMD16ExtSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float PORTABLE_ALIGN32 TmpRes[8];
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);

    size_t qty16 = qty / 16;

    const float *pEnd1 = pVect1 + 16 * qty16;

    __m128 v1, v2;
    __m128 sum_prod = _mm_set1_ps(0);

    while (pVect1 < pEnd1) {
        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
    }
    _mm_store_ps(TmpRes, sum_prod);
    float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];

    return sum;
}

static float
InnerProductDistanceSIMD16ExtSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    return 1.0f - InnerProductSIMD16ExtSSE(pVect1v, pVect2v, qty_ptr);
}

#endif

#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
static DISTFUNC<float> InnerProductSIMD16Ext = InnerProductSIMD16ExtSSE;
static DISTFUNC<float> InnerProductSIMD4Ext = InnerProductSIMD4ExtSSE;
static DISTFUNC<float> InnerProductDistanceSIMD16Ext = InnerProductDistanceSIMD16ExtSSE;
static DISTFUNC<float> InnerProductDistanceSIMD4Ext = InnerProductDistanceSIMD4ExtSSE;

static float
InnerProductDistanceSIMD16ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    size_t qty = *((size_t *) qty_ptr);
    size_t qty16 = qty >> 4 << 4;
    float res = InnerProductSIMD16Ext(pVect1v, pVect2v, &qty16);
    float *pVect1 = (float *) pVect1v + qty16;
    float *pVect2 = (float *) pVect2v + qty16;

    size_t qty_left = qty - qty16;
    float res_tail = InnerProduct(pVect1, pVect2, &qty_left);
    return 1.0f - (res + res_tail);
}

static float
InnerProductDistanceSIMD4ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    size_t qty = *((size_t *) qty_ptr);
    size_t qty4 = qty >> 2 << 2;

    float res = InnerProductSIMD4Ext(pVect1v, pVect2v, &qty4);
    size_t qty_left = qty - qty4;

    float *pVect1 = (float *) pVect1v + qty4;
    float *pVect2 = (float *) pVect2v + qty4;
    float res_tail = InnerProduct(pVect1, pVect2, &qty_left);

    return 1.0f - (res + res_tail);
}
#endif

class InnerProductSpace : public SpaceInterface<float> {
    DISTFUNC<float> fstdistfunc_, fstdistfunc_aligned_ = nullptr;
    size_t data_size_;
    size_t dim_;

 public:
    InnerProductSpace(size_t dim) {
        __builtin_cpu_init();
        fstdistfunc_ = InnerProductDistance;
#if defined(USE_AVX) || defined(USE_SSE) || defined(USE_AVX512)
    #if defined(USE_AVX512)
        if (AVX512Capable()) {
            InnerProductSIMD16Ext = InnerProductSIMD16ExtAVX512;
            InnerProductDistanceSIMD16Ext = InnerProductDistanceSIMD16ExtAVX512;
        } else if (AVXCapable()) {
            InnerProductSIMD16Ext = InnerProductSIMD16ExtAVX;
            InnerProductDistanceSIMD16Ext = InnerProductDistanceSIMD16ExtAVX;
        }
    #elif defined(USE_AVX)
        if (AVXCapable()) {
            InnerProductSIMD16Ext = InnerProductSIMD16ExtAVX;
            InnerProductDistanceSIMD16Ext = InnerProductDistanceSIMD16ExtAVX;
        }
    #endif
    #if not defined(USE_AVX512)
    (void)AVX512Capable;
    #endif
    #if defined(USE_AVX)
        if (AVXCapable())
            InnerProductSIMD4Ext = InnerProductSIMD4ExtAVX;
    #endif
        if (dim % 16 == 0)
        {
            fstdistfunc_ = InnerProductDistanceSIMD16Ext;
            fstdistfunc_aligned_ = InnerProductSIMD16Ext_ALIGNED;
            if( __builtin_cpu_supports("avx") )
                fstdistfunc_aligned_ = InnerProductSIMD16ExtAVX_ALIGNED;
            if( __builtin_cpu_supports("fma") )
                fstdistfunc_aligned_ = InnerProductSIMD16ExtAVXFMA_ALIGNED;
            if( __builtin_cpu_supports("avx512f") )
                fstdistfunc_aligned_ = InnerProductSIMD16ExtAVX3_ALIGNED;
        }
        else if (dim % 4 == 0)
            fstdistfunc_ = InnerProductDistanceSIMD4Ext;
        else if (dim > 16)
            fstdistfunc_ = InnerProductDistanceSIMD16ExtResiduals;
        else if (dim > 4)
            fstdistfunc_ = InnerProductDistanceSIMD4ExtResiduals;
#endif
        dim_ = dim;
        data_size_ = dim * sizeof(float);
    }

    size_t get_data_size() override {
        return data_size_;
    }

    DISTFUNC<float> get_dist_func() override {
        return fstdistfunc_;
    }

    void *get_dist_func_param() override {
        return &dim_;
    }

    DISTFUNC<float> get_dist_func_aligned() override {
        return fstdistfunc_aligned_;
    }

    size_t get_dim() const override {
        return dim_;
    }

    ~InnerProductSpace() {}
private:
    static void unused() {
#if defined(__GNUC__) && !defined(NDEBUG)
        // GCC compiler has bug with undefined function
        // when has been used in debug mode (-O0)
        // this hack used only to avoid buggy behavior
        InnerProductSIMD16Ext_ALIGNED(0, 0, 0);
        InnerProductSIMD16ExtAVX_ALIGNED(0, 0, 0);
        InnerProductSIMD16ExtAVXFMA_ALIGNED(0, 0, 0);
        InnerProductSIMD16ExtAVX3_ALIGNED(0, 0, 0);
#endif
    }
};

}  // namespace hnswlib
