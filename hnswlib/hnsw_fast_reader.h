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
protected:
    static ssize_t mmap4read( const char* filename, void * &mmaped )
    {
        int fd = open(filename, O_RDONLY, 0400);
        if (fd == -1) {
            return -1;
        }

        struct stat fd_stat;
        if (fstat(fd, &fd_stat)) {
            close(fd);
            return -1;
        }

        mmaped = mmap(0, fd_stat.st_size, PROT_READ, MAP_SHARED, fd, 0);
        close(fd);
        if (mmaped == MAP_FAILED) {
            return -1;
        }

        return fd_stat.st_size;
    }
};

class MMapDataMapper : public MappingBase {
public:
    static Mapping map(const char* filename, bool need_mlock) {
        void *mmaped;
        size_t file_size = mmap4read(filename, mmaped);
        if( (ssize_t)file_size < 0 )
            return Mapping{};

        if (need_mlock) {
            ::mlock(mmaped, file_size);
        }

        return Mapping(mmaped, file_size);
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
protected:
    // ordinal HUGE PAGE size for most systems!
    static size_t constexpr HUGE_PAGE_SIZE = 1UL << 21; // 2M
    static size_t constexpr HUGE_PAGE_SZ_MASK = HUGE_PAGE_SIZE - 1;
public:
    static Mapping map(const char* filename, bool need_mlock) {
        void *mmaped;
        size_t file_size = mmap4read(filename, mmaped);
        if( (ssize_t)file_size < 0 )
            return Mapping{};

        Mapping thp_map = alloc_mem(file_size);

        // copy data
        memcpy(thp_map.data, mmaped, file_size);

        // release file-mapping
        unmap(Mapping{mmaped, file_size});

        return thp_map;
    }

    static Mapping alloc_mem( size_t sz )
    {
        size_t alloc_sz;
        void *thp_map = alloc_thp_aligned(sz, alloc_sz);
        return Mapping(thp_map, sz, alloc_sz);
    }
protected:
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

// universal HugePages allocator to work with annon-huge-pages in the linux
// can allocate on the HugeTLBfs first, with low-cost fall-back to the THP
class HugePagesDataMapper : public THPDataMapper
{
public:
    static Mapping map(const char* filename, bool need_mlock) {
        void *mmaped;
        size_t file_size = mmap4read(filename, mmaped);
        if( (ssize_t)file_size < 0 )
            return Mapping{};

        // try to allocate on the hugetlbfs first
        void *htlb_mmaping = mmap(0, file_size, PROT_READ | PROT_WRITE,
            MAP_SHARED | MAP_ANONYMOUS | MAP_HUGETLB, 0, 0);

        Mapping hp_map {};
        
        // no way, no avail memory or not exist at all => 
        //   falling back to the THP mode
        if( htlb_mmaping == MAP_FAILED )
            hp_map = THPDataMapper::alloc_mem(file_size);
        else
            hp_map = Mapping{htlb_mmaping, file_size};

        // copy data
        memcpy(hp_map.data, mmaped, file_size);

        // release file-mapping
        unmap(Mapping(mmaped, file_size));

        return hp_map;
    }

    static Mapping alloc_mem( size_t sz )
    {
        void *htlb_mmaping = mmap(0, sz, PROT_READ | PROT_WRITE,
            MAP_SHARED | MAP_ANONYMOUS | MAP_HUGETLB, 0, 0);

        if( htlb_mmaping == MAP_FAILED )
        {
            // fall-back to THP
            return THPDataMapper::alloc_mem(sz);
        }

        return Mapping{htlb_mmaping, sz};
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

// this need to separate shared storage and temporary data
// for example under multi-threaded environment we need to 
// separate temporary contexts
template<typename MMapImpl>
class DefaultContextHolder
{
public:
    void alloc( size_t max_elements )
    {
        mapping_ = MMapImpl::alloc_mem( max_elements * sizeof(uint16_t) );
        // TODO: place visited_map in the same VMA!
        visited_map_.reset( new VisitedMap(max_elements, mapping_.data) );
    }
    uint16_t const* scan_start( uint32_t ep_id, uint32_t )
    {
        visited_map_->reset();
        uint16_t const *visited_array = visited_map_->arr;

        visited_map_->set(ep_id);

        return visited_array;
    }
    bool mark( uint32_t pos )
    {
        return visited_map_->mark(pos);
    }
    ~DefaultContextHolder()
    {
        MMapImpl::unmap(mapping_);
    }
    bool is_initialized() const
    {
        return visited_map_ != nullptr;
    }
private:
    using VisitedMap = VisitedMapT<uint16_t>;
    std::unique_ptr<VisitedMap> visited_map_;
    DataMapping mapping_;
};

// default multi-threaded implementation with lazy init
template<typename MMapImpl, typename GlobalName>
class MTContextHolder
{
    using ImplT = DefaultContextHolder<MMapImpl>;
public:
    void alloc( size_t max_elements )
    {
        // lazy init, do nothing here
    }
    uint16_t const* scan_start( uint32_t ep_id, uint32_t max_elements )
    {
        if( !impl_.is_initialized() )
        {
            impl_.alloc(max_elements);
        }

        return impl_.scan_start(ep_id, max_elements);
    }
    bool mark( uint32_t pos )
    {
        return impl_.mark(pos);
    }
private:
    static thread_local ImplT impl_;
};

template<typename MMapImpl, typename GlobalName>
thread_local typename MTContextHolder<MMapImpl, GlobalName>::ImplT 
MTContextHolder<MMapImpl, GlobalName>::impl_;


// reads filecontent directly from disk-cache w/o moving to the anon memory and allocation
// reading and search only, so temporary structures for quick expand also omitted
// to get maximum speed, can be used with HierarchicalNSW class
// NOTE: this code is completely based on code from original HierarchicalNSW class!
template<typename dist_t, typename MMapImpl = MMapDataMapper, 
    typename CtxHolder = DefaultContextHolder<MMapImpl> >
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

        while (!top_candidates.empty()) {
            auto rez = top_candidates.top();
            result.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
            top_candidates.pop();
        }
        return result;
    }

    // keeps original API semantics
    // Filter functor works as flow stopper too, return true, if you want early exit
    template<typename Filter>
    void searchKnnFilter(const dist_t *query_data,
        const dist_t *query_data_end, uint32_t k, Filter f) const {
        size_t max_ef = std::max(ef_, k);
        pq_top_candidates_t top_candidates(max_ef);
        searchKnnImpl(query_data, query_data_end, max_ef, top_candidates);

        // traverse pq as array
        for( auto p : top_candidates.get_cont() )
        {
            if( f(p) )
                break;
        }
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

                data = (uint32_t *) get_linklist(ep_id, level);
                for (tableint *it = data + 1, *end = it + getListCount(data); 
                    it != end; ++it) {
                    tableint cand = *it;
                    dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), query_data_end);

                    if (d < curdist) {
                        curdist = d;
                        ep_id = cand;
                        changed = true;
                    }
                }
            }
        }

        searchBaseLayerST(ep_id, curdist, query_data, query_data_end, max_ef, top_candidates);
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

    inline labeltype getExternalLabel(tableint internal_id) const {
        labeltype return_label;
        memcpy(&return_label, (internal_id * size_data_per_element_ + label_offset_), sizeof(labeltype));
        return return_label;
    }

private:


    pq_top_candidates_t
    searchBaseLayerST(tableint ep_id, dist_t lowerBound, const dist_t *query_data, const dist_t *query_data_end,
        size_t ef, pq_top_candidates_t &top_candidates) const {

        // average depth seems to be less than 64 in my tests
        // but you always must be stay tuned!
        pq_top_candidates_t candidate_set(64);

        top_candidates.emplace(lowerBound, ep_id);
        candidate_set.emplace(-lowerBound, ep_id);

        auto const *visited_array = ctx_.scan_start(ep_id, max_elements_);

        while (!candidate_set.empty()) {
            auto [candidate_dist, current_node_id] = candidate_set.top();
            candidate_dist = -candidate_dist;
            if (candidate_dist > lowerBound)
                break;

            candidate_set.pop();

            uint32_t const *data = (uint32_t *) get_linklist0(current_node_id);
            size_t size = getListCount((linklistsizeint*)data);
            __builtin_prefetch(visited_array + *(data + 1), 1);
            __builtin_prefetch(visited_array + *(data + 1) + 64, 1);
            __builtin_prefetch(getDataByInternalId(*(data + 1)), 0);
            // prefetch list future
            __builtin_prefetch(data + 2, 0);

            for (size_t j = 1; j <= size; j++) {
                uint32_t candidate_id = *(data + j);
                __builtin_prefetch(visited_array + *(data + j + 1), 1);
                __builtin_prefetch(getDataByInternalId(*(data + j + 1)), 0);

                if( ctx_.mark(candidate_id) ) {
                    char const* ep_data = getDataByInternalId(candidate_id);
                    dist_t dist = fstdistfunc_(query_data, ep_data, query_data_end);

                    bool flag_consider_candidate =
                        lowerBound > dist || top_candidates.size() < ef;

                    if (flag_consider_candidate) {
                        candidate_set.emplace(-dist, candidate_id);
                        __builtin_prefetch(get_linklist0(candidate_set.top().second), 0);
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

    static uint32_t scan_list( linklistsizeint *lst )
    {
        uint32_t *list_data = (uint32_t*)lst;
        size_t list_sz = getListCount(lst);
        //if( list_sz != cnt )
        //    throw std::runtime_error("list sizes not match!");

        uint32_t prev_id = 0;
        for (size_t j = 1; j <= list_sz; j++) {
            uint32_t candidate_id = *(list_data + j);

            if( prev_id > candidate_id )
                return 0;

            prev_id = candidate_id;
        }

        return 1;
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
        ctx_.alloc(max_elements);

        tmp_mapping_ = MMapImpl::alloc_mem(sizeof(void *) * max_elements);

        linkLists_ = (char **)tmp_mapping_.data;
        ef_ = 10;
        size_t total_lnk_sz = 0, non_empty = 0;
        uint32_t min_len_sz = 100500, max_len_sz = 0, ordered_list_cnt = 0;
        for (size_t i = 0; i < cur_element_count; i++) {
            uint32_t linkListSize;
            MMapImpl::readPOD(input, linkListSize);
            if (linkListSize == 0) {
                linkLists_[i] = nullptr;
            } else {
                linkLists_[i] = const_cast<char*>(input);
                input += linkListSize;
                non_empty++;
                total_lnk_sz += linkListSize;
                min_len_sz = std::min(min_len_sz, linkListSize);
                max_len_sz = std::max(max_len_sz, linkListSize);
                ordered_list_cnt += scan_list((linklistsizeint*)linkLists_[i]);
            }
            if( input > end_addr )
                throw std::runtime_error("file read error");
        }

#ifdef COLLECT_STATS_FILE

        std::cerr << "Total multi-level lists: " << cur_element_count << " full: " << non_empty 
            << " AVG lnk_sz=" << double(total_lnk_sz) / non_empty 
            << " min_len=" << min_len_sz
            << " max_len=" << max_len_sz
            << " size_links_per_element=" << size_links_per_element_
            << " offsetData=" << offsetData
            << " offsetLevel0=" << offsetLevel0
            << " ordered_list_cnt=" << ordered_list_cnt
            << std::endl;

        total_lnk_sz = 0, non_empty = 0;
        min_len_sz = 100500, max_len_sz = 0, ordered_list_cnt = 0;

        for (size_t i = 0; i < cur_element_count; i++) {
            linklistsizeint *lst = get_linklist0(i);
            uint32_t linkListSize = getListCount(lst);
            if( linkListSize )
                non_empty++;
            total_lnk_sz += linkListSize;
            min_len_sz = std::min(min_len_sz, linkListSize);
            max_len_sz = std::max(max_len_sz, linkListSize);
            ordered_list_cnt += scan_list(lst);
        }

        std::cerr << "Total LEVEL0 lists: " << cur_element_count << " full: " << non_empty 
            << " AVG lnk_sz=" << double(total_lnk_sz) / non_empty 
            << " min_len=" << min_len_sz
            << " max_len=" << max_len_sz
            << " size_data_per_element=" << size_data_per_element_
            << " ordered_list_cnt=" << ordered_list_cnt
            << std::endl;
#endif
    }

    inline linklistsizeint *get_linklist0(tableint internal_id) const {
        return (linklistsizeint *) (internal_id * size_data_per_element_ + offsetLevel0_);
    }

    inline linklistsizeint *get_linklist(tableint internal_id, uint32_t level) const {
        return (linklistsizeint *) (linkLists_[internal_id] + (level - 1) * size_links_per_element_);
    }

    static uint16_t getListCount(linklistsizeint * ptr) {
        return *((uint16_t *)ptr);
    }

private:
    uint32_t size_data_per_element_{0};
    uint32_t size_links_per_element_{0};

    char const *offsetData_{nullptr}, *offsetLevel0_{nullptr}, *label_offset_{nullptr};

    char **linkLists_{nullptr};

    DISTFUNC<dist_t> fstdistfunc_;

    uint32_t ef_{ 0 };
    tableint enterpoint_node_{0};
    int maxlevel_{0};
    uint32_t max_elements_{0};

    mutable CtxHolder ctx_;

    DataMapping mapping_, tmp_mapping_;
};

}
