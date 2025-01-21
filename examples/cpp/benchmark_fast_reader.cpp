#include "../../hnswlib/hnsw_fast_reader.h"
#include <chrono>

static int const dim = 64;
static int const max_elements = 1000000;
static int const M = 16;
static int const ef_construction = 200;  // Controls index search speed/build speed tradeoff

char const *INDEX_NAMES[] = { "bench-index-l2.hnsw", "bench-index-ip.hnsw" };

struct Timelapse
{
    std::chrono::high_resolution_clock::time_point t_start, t_end;
    char const *name;
    Timelapse( char const *name) : name(name) { t_start = std::chrono::high_resolution_clock::now(); }
    ~Timelapse() {
        t_end = std::chrono::high_resolution_clock::now();
        std::cout   << "timelapse for " << name
                    << " done in " << std::chrono::duration_cast<std::chrono::milliseconds>( t_end - t_start ).count()
                    << " millis.\n";
    }
};


template<typename S>
void create_index_if_not_exists(float const *data, char const *fname)
{
    std::ifstream f(fname);
    if( !f.good() )
    {
        std::cout << "DB fname: " << fname << " not found, creating new index of " << max_elements << " items." << std::endl;
        // Initing index
        S space(dim);
        hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);

        {
            Timelapse tm("append data");
            // Add data to index
            for (int i = 0; i < max_elements; i++) {
                alg_hnsw->addPoint(data + i * dim, i);
            }
        }

        {
            Timelapse tm("saveIndex");
            // Serialize index
            alg_hnsw->saveIndex(fname);
        }
        delete alg_hnsw;
    }
}

template<bool raw_api, typename T>
static bool search( float const *vec, T *api, hnswlib::labeltype i )
{
    auto result = api->searchKnn(vec, 1);
    return result.top().second == i;
}

template<bool raw_api, typename T>
static bool search( float const *vec, hnswlib::HierarchicalNSWFastReader<float, T> *api,
    hnswlib::labeltype i )
{
    if( !raw_api )
    {
        auto result = api->searchKnn(vec, vec + dim, 1);
        return result.top().second == i;
    }
    else
    {
        auto result = api->searchKnnDistanceLimit(vec, vec + dim, std::max(1U, api->getEf()), 2.f);
        return !result.empty() ? result.front().second == i : false;
    }
}

template<typename T>
static void debug_print( T *api )
{

}

template<typename T>
static void debug_print( hnswlib::HierarchicalNSWFastReader<float, T> *api )
{
    //std::cout << "get_avg_candidate_set_size()=" << api->get_avg_candidate_set_size() << std::endl;
}

template<typename T, bool raw_api, typename S>
static void bench_recall_from_storage(char const *fname, S *space, float const *data, int ntimes )
{
    std::chrono::high_resolution_clock::time_point t_start, t_end;
    using hnswlib::labeltype;
    T *alg_hnsw;
    {
        Timelapse tm("loading index");
        alg_hnsw = new T(space, fname);
        std::cout << "Index nelements: " << alg_hnsw->getMaxElements() << ", ";
    }


    float avg_recall = 0;
    {
        Timelapse tm("benching");
        for( int t = 0; t < ntimes; ++t )
        {
            float correct = 0;
            for (labeltype i = 0; i < max_elements; i++) {
                float const *vec = data + i * dim;
                if (search<raw_api>(vec, alg_hnsw, i)) correct++;
            }
            float recall = (float)correct / max_elements;

            avg_recall += recall;
        }
    }
    std::cout << typeid(T).name() << " Recall of deserialized index: "
        << avg_recall / ntimes << std::endl;


    debug_print(alg_hnsw);

    delete alg_hnsw;
}


template<typename S>
void test( int ntimes, int dim, float const *data, char const *fname )
{
    create_index_if_not_exists<S>(data, fname);
    S space(dim);

    std::cout << "Start benchmarking for space: " << typeid(S).name() << " dim: " << dim << std::endl;

    bench_recall_from_storage<hnswlib::HierarchicalNSW<float>, false>(fname, &space, data, ntimes);
    bench_recall_from_storage<hnswlib::HierarchicalNSWFastReader<float, hnswlib::THPDataMapper>, false>(fname, &space, data, ntimes);
    bench_recall_from_storage<hnswlib::HierarchicalNSWFastReader<float, hnswlib::THPDataMapper>, true>(fname, &space, data, ntimes);
}

int main()
{
    // Generate random data
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    float* data = (float*)hnswlib::aligned_malloc(dim * max_elements * sizeof(float), 64);
    for (int i = 0; i < dim * max_elements; i++) {
        data[i] = distrib_real(rng);
    }


    //test<hnswlib::L2Space>(3, dim, data, INDEX_NAMES[0]);
    test<hnswlib::InnerProductSpace>(3, dim, data, INDEX_NAMES[1]);

    free(data);
}