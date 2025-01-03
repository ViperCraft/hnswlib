#include "../../hnswlib/hnsw_fast_reader.h"
#include <chrono>

static int const dim = 64;
static int const max_elements = 1000000;
static int const M = 16;
static int const ef_construction = 200;  // Controls index search speed/build speed tradeoff

char const INDEX_NAME[] = { "bench-index.hnsw" };

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

void create_index_if_not_exists(float const *data)
{
    // Initing index
    hnswlib::L2Space space(dim);
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
        alg_hnsw->saveIndex(INDEX_NAME);
    }
    delete alg_hnsw;
}

template<typename T>
static hnswlib::labeltype search( float const *vec, T *api )
{
    auto result = api->searchKnn(vec, 1);
    return result.top().second;
}

template<typename T>
static hnswlib::labeltype search( float const *vec, hnswlib::HierarchicalNSWFastReader<float, T> *api )
{
    auto result = api->searchKnn(vec, vec + dim, 1);
    return result.top().second;
}


template<typename T, typename S>
static void bench_recall_from_storage(S *space, float const *data, int ntimes )
{
    std::chrono::high_resolution_clock::time_point t_start, t_end;
    using hnswlib::labeltype;
    T *alg_hnsw;
    {
        Timelapse tm("loading index");
        alg_hnsw = new T(space, INDEX_NAME);
    }

    float avg_recall = 0;
    {
        Timelapse tm("benching");
        for( int t = 0; t < ntimes; ++t )
        {
            float correct = 0;
            for (labeltype i = 0; i < max_elements; i++) {
                float const *vec = data + i * dim;
                if (search(vec, alg_hnsw) == i) correct++;
            }
            float recall = (float)correct / max_elements;

            avg_recall += recall;
        }
    }
    std::cout << typeid(T).name() << " Recall of deserialized index: " 
        << avg_recall / ntimes << std::endl;

    delete alg_hnsw;
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

    std::ifstream f(INDEX_NAME);
    
    if( !f.good() )
        create_index_if_not_exists(data);

    hnswlib::L2Space space(dim);

    int ntimes = 3;

    bench_recall_from_storage<hnswlib::HierarchicalNSW<float>>(&space, data, ntimes);
    bench_recall_from_storage<hnswlib::HierarchicalNSWFastReader<float, hnswlib::THPDataMapper>>(&space, data, ntimes);

    free(data);
}