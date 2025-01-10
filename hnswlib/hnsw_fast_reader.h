#pragma once

#include "hnswlib.h"
#include <stdlib.h>
#include <assert.h>
#include <memory>
#include <stdint.h>
#include "hnswalg.h"
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <stdio.h>
#include <stddef.h>
#include <fcntl.h>
#include <unistd.h>

namespace hnswlib {

struct DataMapping
{
    DataMapping( void *d, size_t s )
        : data(d)
        , data_size(s)
        , alloc_size(s)
        {}

    DataMapping( void *d, size_t data_sz, size_t alloc_sz )
        : data(d)
        , data_size(data_sz)
        , alloc_size(alloc_sz)
        {}

    DataMapping() = default;

    void *data = nullptr;
    size_t data_size = 0;
    size_t alloc_size = 0;
};

inline void* aligned_malloc( size_t sz, size_t alignment = 16 )
{
    void *out_bytes;
    if( posix_memalign(&out_bytes, alignment, sz) )
        throw std::bad_alloc();

    return out_bytes;
}

class MappingBase
{
protected:
    using Mapping = DataMapping;
    static constexpr size_t align_size( size_t sz, size_t block_siz )
    {
        return (sz + block_siz - 1) & ~(block_siz - 1);
    }
public:
    static void unmap(const Mapping & mapping) {
        if (mapping.data)
            ::munmap(const_cast<void*>(mapping.data), mapping.alloc_size);
    }

    template<typename T>
    static void readPOD( char const *&top, T &out )
    {
        out = *(T const*)top;
        top += sizeof(T);
    }

    static void mlock(const Mapping & mapping) {
        if (mapping.data)
            ::mlock(const_cast<void*>(mapping.data), mapping.alloc_size);
    }
};

class MMapDataMapper : public MappingBase {
public:
    static Mapping map(const char* filename, bool need_mlock) {
        int fd = open(filename, O_RDONLY, 0400);
        if (fd == -1) {
            return Mapping{};
        }

        struct stat fd_stat;
        if (fstat(fd, &fd_stat)) {
            close(fd);
            return Mapping{};
        }

        void *mmaped = mmap(0, fd_stat.st_size, PROT_READ, MAP_SHARED, fd, 0);
        close(fd);
        if (mmaped == MAP_FAILED) {
            return Mapping{};
        }
#if defined(MADV_DONTDUMP)
        // Exclude from a core dump those pages
        madvise(mmaped, fd_stat.st_size, MADV_DONTDUMP);
#endif

        if (need_mlock) {
            ::mlock(mmaped, fd_stat.st_size);
        }

        return Mapping(mmaped, (size_t)fd_stat.st_size);
    }

    static Mapping alloc_mem( size_t sz ) {
        void *mem = mmap(nullptr, sz, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
        if( MAP_FAILED == mem )
            throw std::bad_alloc();
        return Mapping(mem, sz);
    }
};

class THPDataMapper : public MappingBase
{
    // ordinal HUGE PAGE size for most systems!
    static size_t constexpr HUGE_PAGE_SIZE = 1UL << 21; // 2M
    static size_t constexpr HUGE_PAGE_SZ_MASK = HUGE_PAGE_SIZE - 1;
public:
    static Mapping map(const char* filename, bool need_mlock) {
        int fd = open(filename, O_RDONLY, 0400);
        if (fd == -1) {
            return Mapping{};
        }

        struct stat fd_stat;
        if (fstat(fd, &fd_stat)) {
            close(fd);
            return Mapping{};
        }

        void *mmaped = mmap(0, fd_stat.st_size, PROT_READ, MAP_SHARED, fd, 0);
        close(fd);
        if (mmaped == MAP_FAILED) {
            return Mapping{};
        }


        Mapping thp_map = alloc_mem(fd_stat.st_size);

        // copy data
        memcpy(thp_map.data, mmaped, fd_stat.st_size);

        // release file-mapping
        unmap(Mapping(mmaped, fd_stat.st_size));

        return thp_map;
    }

    static Mapping alloc_mem( size_t sz )
    {
        size_t alloc_sz;
        void *thp_map = alloc_thp_aligned(sz, alloc_sz);
        return Mapping(thp_map, sz, alloc_sz);
    }
private:
    static void* alloc_thp_aligned( size_t sz, size_t &out_mem_sz )
    {
        // align to the next page size
        sz = align_size(sz, HUGE_PAGE_SIZE);

        // sadly but in order to alloc HUGE_PAGE
        // we need always overbooking +HUGE_PAGE_SIZE(2M)
        // to make proper alignment via cutting
        void *mem = mmap(nullptr, sz + HUGE_PAGE_SIZE, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
        if( MAP_FAILED == mem )
            throw std::bad_alloc();
        // align pointer to the HUGE_PAGE_SIZE
        if( size_t(mem) % HUGE_PAGE_SIZE )
        {
            void *aligned_mem = (void *) (((size_t)mem + HUGE_PAGE_SZ_MASK) & ~HUGE_PAGE_SZ_MASK);
            // free unaligned chunk of mem
            size_t unaligned_chunk_sz = (size_t)aligned_mem - (size_t)mem;
            munmap(mem, unaligned_chunk_sz);
            mem = aligned_mem;
            out_mem_sz = sz + HUGE_PAGE_SIZE - unaligned_chunk_sz;
        }
        else
            out_mem_sz = sz + HUGE_PAGE_SIZE;

#if defined(MADV_HUGEPAGE)
        // make THP
        madvise(mem, out_mem_sz, MADV_HUGEPAGE);
#endif
        return mem;
    }
};

// much lower memory footprint than bitmapped visited list
// on very huge lists > 1M
template<typename T>
class VisitedMapT {
 public:
    T *arr;
    T val = -1;
    uint32_t sz;

    VisitedMapT(uint32_t sz, void *mem) : sz(sz) {
        arr = static_cast<T*>(mem);
    }

    void reset() {
        val++;
        if (val == 0) {
            memset(arr, 0, sizeof(T) * sz);
            val++;
        }
    }

    void set( uint32_t pos )
    {
        arr[pos] = val;
    }

    bool mark( uint32_t pos )
    {
        if( arr[pos] != val )
        {
            arr[pos] = val;
            return true;
        }

        return false;
    }

    ~VisitedMapT() {}
};


// reads filecontent directly from disk-cache w/o moving to the anon memory and allocation
// reading and search only, so temporary structures for quick expand also omitted
// to get maximum speed, can be used with HierarchicalNSW class
// NOTE: this code is completely based on code from original HierarchicalNSW class!
template<typename dist_t, typename MMapImpl = MMapDataMapper>
class HierarchicalNSWFastReader
{
public:
    using pq_result_t = std::priority_queue<std::pair<dist_t, labeltype >>;
    using result_list = std::vector<std::pair<dist_t, tableint>>;
    struct pq_top_candidates_t : std::priority_queue<std::pair<dist_t, tableint>,
        result_list, typename HierarchicalNSW<dist_t>::CompareByFirst>
    {
        pq_top_candidates_t( size_t reserve_n = 0 ) { reserve(reserve_n); }
        result_list const& get_cont() const { return this->c; }
        void reserve( size_t n ) { this->c.reserve(n); }
        void clear() { this->c.clear(); }
    };
public:

    HierarchicalNSWFastReader(
        SpaceInterface<dist_t> *s,
        const std::string &location, bool mlock = false) {
        loadIndex(location, s, mlock);
    }


    ~HierarchicalNSWFastReader() {
        MMapImpl::unmap(mapping_);
        MMapImpl::unmap(tmp_mapping_);
    }


    pq_result_t
    searchKnn(const dist_t *query_data, const dist_t *query_data_end, uint32_t k) const {
        pq_result_t result;
        if (max_elements_ == 0) return result;

        size_t max_ef = std::max(ef_, k);

        pq_top_candidates_t top_candidates(max_ef);
        searchKnnImpl(query_data, query_data_end, max_ef, top_candidates);

        while (top_candidates.size() > k) {
            top_candidates.pop();
        }

        while (top_candidates.size() > 0) {
            std::pair<dist_t, tableint> rez = top_candidates.top();
            result.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
            top_candidates.pop();
        }
        return result;
    }


    // WARN: dangerous interface, don't use me if you not a Jedi ;)
    void searchKnnImpl(const dist_t *query_data, const dist_t *query_data_end,
        size_t max_ef, pq_top_candidates_t &top_candidates) const {

        tableint ep_id = enterpoint_node_;
        dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), query_data_end);

        for (int level = maxlevel_; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                tableint *data;

                data = (unsigned int *) get_linklist(ep_id, level);
                int size = getListCount(data);

                tableint *datal = data + 1;
                for (int i = 0; i < size; i++) {
                    tableint cand = datal[i];
                    dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), query_data_end);

                    if (d < curdist) {
                        curdist = d;
                        ep_id = cand;
                        changed = true;
                    }
                }
            }
        }

        searchBaseLayerST(ep_id, query_data, query_data_end, max_ef, top_candidates);
    }


    void setEf(uint32_t ef) {
        ef_ = ef;
    }

    uint32_t getEf() const { return ef_; }

    size_t getMaxElements() const {
        return max_elements_;
    }

    // WARN: query_point must be aligned!!!
    dist_t getDistance( const dist_t *query_point, const dist_t *query_point_end, tableint to ) const {
        char const* to_data = getDataByInternalId(to);
        return fstdistfunc_(query_point, to_data, query_point_end);
    }

    inline char const* getDataByInternalId(tableint internal_id) const {
        return (internal_id * size_data_per_element_ + offsetData_);
    }
private:

    pq_top_candidates_t
    searchBaseLayerST(tableint ep_id, const dist_t *data_point, const dist_t *query_data_end,
        size_t ef, pq_top_candidates_t &top_candidates) const {
        visited_map_->reset();

        auto const *visited_array = visited_map_->arr;

        // sorry don't know better numbers, must be tuned
        pq_top_candidates_t candidate_set(256);

        dist_t lowerBound;
        char const* ep_data = getDataByInternalId(ep_id);
        dist_t dist = fstdistfunc_(data_point, ep_data, query_data_end);
        lowerBound = dist;
        top_candidates.emplace(dist, ep_id);
        candidate_set.emplace(-dist, ep_id);

        visited_map_->set(ep_id);

        while (!candidate_set.empty()) {
            auto [candidate_dist, current_node_id] = candidate_set.top();
            candidate_dist = -candidate_dist;
            if (candidate_dist > lowerBound)
                break;

            candidate_set.pop();

            int const *data = (int *) get_linklist0(current_node_id);
            size_t size = getListCount((linklistsizeint*)data);

#ifdef USE_SSE
            _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
            _mm_prefetch(getDataByInternalId(*(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char *) (data + 2), _MM_HINT_T0);
#endif

            for (size_t j = 1; j <= size; j++) {
                int candidate_id = *(data + j);
#ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(data + j + 1)), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*(data + j + 1)),
                                _MM_HINT_T0);  ////////////
#endif
                if( visited_map_->mark(candidate_id) ) {
                    ep_data = getDataByInternalId(candidate_id);
                    dist_t dist = fstdistfunc_(data_point, ep_data, query_data_end);

                    bool flag_consider_candidate =
                        top_candidates.size() < ef || lowerBound > dist;

                    if (flag_consider_candidate) {
                        candidate_set.emplace(-dist, candidate_id);
#ifdef USE_SSE
                        _mm_prefetch(get_linklist0(candidate_set.top().second), _MM_HINT_T0);
#endif
                        top_candidates.emplace(dist, candidate_id);
                        while (top_candidates.size() > ef) {
                            top_candidates.pop();
                        }

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    }
                }
            }
        }

        return top_candidates;
    }

    void loadIndex(const std::string &location, SpaceInterface<dist_t> *s, bool mlock) {

        mapping_ = MMapImpl::map(location.c_str(), mlock);

        if( nullptr == mapping_.data )
            throw std::runtime_error("failed to mmap index file: " + location);

        size_t M_{0};
        size_t maxM0_{0};
        size_t maxM_{0};
        size_t ef_construction_{0};
        size_t cur_element_count{0};  // current number of elements
        size_t offsetData, offsetLevel0, label_offset;
        size_t size_data_per_element, max_elements;
        double mult_{0.0};

        char const *input = static_cast<char const*>(mapping_.data), *end_addr = input + mapping_.data_size;

        MMapImpl::readPOD(input, offsetLevel0);
        MMapImpl::readPOD(input, max_elements);
        MMapImpl::readPOD(input, cur_element_count);

        max_elements = cur_element_count;
        max_elements_= max_elements;
        MMapImpl::readPOD(input, size_data_per_element);
        MMapImpl::readPOD(input, label_offset);
        MMapImpl::readPOD(input, offsetData);
        MMapImpl::readPOD(input, maxlevel_);
        MMapImpl::readPOD(input, enterpoint_node_);

        MMapImpl::readPOD(input, maxM_);
        MMapImpl::readPOD(input, maxM0_);
        MMapImpl::readPOD(input, M_);
        MMapImpl::readPOD(input, mult_);
        MMapImpl::readPOD(input, ef_construction_);

        fstdistfunc_ = s->get_dist_func_aligned();
        if( nullptr == fstdistfunc_ )
            throw std::runtime_error("not supported space interface or alignment of data is invalid!");

        size_data_per_element_ = size_data_per_element;

        offsetData_ = input + offsetData;
        offsetLevel0_ = input + offsetLevel0;
        label_offset_ = input + label_offset;

        input += max_elements * size_data_per_element;

        size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);

        // alloc mem for temporary structures
        // 1. visited list
        // 2. short-path for linked-lists start addr

        tmp_mapping_ = MMapImpl::alloc_mem( max_elements * sizeof(uint16_t)
            + sizeof(void *) * max_elements);

        visited_map_.reset( new VisitedMap(max_elements, tmp_mapping_.data) );
        linkLists_ = (char **) ((char*)tmp_mapping_.data + max_elements * sizeof(uint16_t) );
        ef_ = 10;
        for (size_t i = 0; i < cur_element_count; i++) {
            unsigned int linkListSize;
            MMapImpl::readPOD(input, linkListSize);
            if (linkListSize == 0) {
                linkLists_[i] = nullptr;
            } else {
                linkLists_[i] = const_cast<char*>(input);
                input += linkListSize;
            }
            if( input > end_addr )
                throw std::runtime_error("file read error");
        }
    }


    inline labeltype getExternalLabel(tableint internal_id) const {
        labeltype return_label;
        memcpy(&return_label, (internal_id * size_data_per_element_ + label_offset_), sizeof(labeltype));
        return return_label;
    }

    inline linklistsizeint *get_linklist0(tableint internal_id) const {
        return (linklistsizeint *) (internal_id * size_data_per_element_ + offsetLevel0_);
    }

    inline linklistsizeint *get_linklist(tableint internal_id, uint32_t level) const {
        return (linklistsizeint *) (linkLists_[internal_id] + (level - 1) * size_links_per_element_);
    }

    unsigned short int getListCount(linklistsizeint * ptr) const {
        return *((unsigned short int *)ptr);
    }

private:
    using VisitedMap = VisitedMapT<uint16_t>;

    uint32_t size_data_per_element_{0};
    uint32_t size_links_per_element_{0};

    char const *offsetData_{nullptr}, *offsetLevel0_{nullptr}, *label_offset_{nullptr};

    char **linkLists_{nullptr};

    DISTFUNC<dist_t> fstdistfunc_;

    std::unique_ptr<VisitedMap> visited_map_;

    uint32_t ef_{ 0 };
    tableint enterpoint_node_{0};
    int maxlevel_{0};
    uint32_t max_elements_{0};

    DataMapping mapping_, tmp_mapping_;
};

}
