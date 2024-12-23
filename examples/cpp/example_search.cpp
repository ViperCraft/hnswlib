#include "../../hnswlib/hnswlib.h"
#include "../../hnswlib/hnsw_fast_scan.h"

template<typename T, typename S>
static void test_recall_from_storage(std::string const &hnsw_path, S *space, float const *data, int dim, int max_elements)
{
    T *alg_hnsw = new T(space, hnsw_path);
    float correct = 0;
    for (int i = 0; i < max_elements; i++) {
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(data + i * dim, 1);
        hnswlib::labeltype label = result.top().second;
        if (label == i) correct++;
    }
    float recall = (float)correct / max_elements;
    std::cout << typeid(T).name() << " Recall of deserialized index: " << recall << "\n";

    delete alg_hnsw;
}

static void test( int dim, int M, int max_elements )
{
    int ef_construction = 200;  // Controls index search speed/build speed tradeoff

    std::cout << "test: dim=" << dim << " M=" << M << " max_elements=" << max_elements
        << std::endl;

    // Initing index
    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);

    // Generate random data
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    float* data = new float[dim * max_elements];
    for (int i = 0; i < dim * max_elements; i++) {
        data[i] = distrib_real(rng);
    }

    // Add data to index
    for (int i = 0; i < max_elements; i++) {
        alg_hnsw->addPoint(data + i * dim, i);
    }

    // Query the elements for themselves and measure recall
    float correct = 0;
    for (int i = 0; i < max_elements; i++) {
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(data + i * dim, 1);
        hnswlib::labeltype label = result.top().second;
        if (label == i) correct++;
    }
    float recall = correct / max_elements;
    std::cout << "Recall: " << recall << "\n";

    // Serialize index
    std::string hnsw_path = "hnsw.bin";
    alg_hnsw->saveIndex(hnsw_path);
    delete alg_hnsw;

    // Deserialize index and check recall
    test_recall_from_storage<hnswlib::HierarchicalNSW<float>>(hnsw_path, &space, data, dim, max_elements);
    test_recall_from_storage<hnswlib::HierarchicalNSWFastReader<float>>(hnsw_path, &space, data, dim, max_elements);

    delete[] data;
}

int main() {
    int max_elements = 100000;   // Maximum number of elements, should be known beforehand
    int M = 16;                 // Tightly connected with internal dimensionality of the data
                                // strongly affects the memory consumption
    for( int dim : {16, 32, 64, 128})
        for( int max_elements : {10000, 100000})
            test(dim, M, 10000);
    return 0;
}
