/*
 * dense_heap.c — Implementation of the dense heap with logical ordering.
 *
 * Design goals:
 *  - Single contiguous mmap region; no malloc after create.
 *  - All hot paths inline-friendly and branch-minimized.
 *  - 1.5× growth strategy to keep amortized cost low.
 *  - Page-aligned arrays to avoid false sharing across cache lines.
 *  - Prefetch hints on iteration-adjacent ops.
 */

#include "dense_heap.h"

#include <assert.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h> /* sysconf */

/* -------------------------------------------------------------------------
 * Internal: page size (cached once)
 * ---------------------------------------------------------------------- */

static size_t _dh_page_size(void) {
  static size_t ps = 0;
  if (DH_UNLIKELY(ps == 0))
    ps = (size_t)sysconf(_SC_PAGESIZE);
  return ps;
}

/* Round up n to nearest multiple of align (must be power of 2) */
DH_INLINE size_t _align_up(size_t n, size_t align) {
  return (n + align - 1u) & ~(align - 1u);
}

/* -------------------------------------------------------------------------
 * Internal: compute layout for a given capacity & elem_size
 * ---------------------------------------------------------------------- */

typedef struct {
  size_t data_off;
  size_t index_off;
  size_t reverse_off;
  size_t total;
} _dh_layout_t;

static DH_COLD _dh_layout_t _dh_compute_layout(size_t capacity,
                                               size_t elem_size) {
  _dh_layout_t L;
  size_t page = _dh_page_size();

  /* header sits at offset 0; data starts after, page-aligned */
  size_t hdr_end = _align_up(sizeof(dh_header_t), DH_CACHELINE);
  L.data_off =
      _align_up(hdr_end, page); /* page-aligned for madvise potential */

  size_t data_bytes = _align_up(capacity * elem_size, DH_CACHELINE);
  size_t index_bytes = _align_up(capacity * sizeof(size_t), DH_CACHELINE);
  size_t reverse_bytes = _align_up(capacity * sizeof(size_t), DH_CACHELINE);

  L.index_off = L.data_off + data_bytes;
  L.reverse_off = L.index_off + index_bytes;
  L.total = _align_up(L.reverse_off + reverse_bytes, page);

  return L;
}

/* -------------------------------------------------------------------------
 * Internal: allocate a fresh mmap region and initialise header
 * ---------------------------------------------------------------------- */

static DH_COLD dh_status_t _dh_alloc_region(dense_heap_t *dh, size_t capacity,
                                            size_t elem_size,
                                            size_t preserve_size) {
  _dh_layout_t L = _dh_compute_layout(capacity, elem_size);

  void *base = mmap(NULL, L.total, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (DH_UNLIKELY(base == MAP_FAILED))
    return DH_ERR_NOMEM;

  /* If we're growing, copy old data + index + reverse into new region */
  if (preserve_size > 0 && dh->hdr != NULL) {
    dh_header_t *old = dh->hdr;

    /* copy data */
    memcpy((char *)base + L.data_off, (char *)old + old->data_off,
           preserve_size * elem_size);

    /* copy index */
    memcpy((char *)base + L.index_off, (char *)old + old->index_off,
           preserve_size * sizeof(size_t));

    /* copy reverse */
    memcpy((char *)base + L.reverse_off, (char *)old + old->reverse_off,
           preserve_size * sizeof(size_t));

    /* unmap old region */
    munmap(old, old->region_size);
  }

  dh_header_t *hdr = (dh_header_t *)base;
  hdr->capacity = capacity;
  hdr->size = preserve_size;
  hdr->elem_size = elem_size;
  hdr->data_off = L.data_off;
  hdr->index_off = L.index_off;
  hdr->reverse_off = L.reverse_off;
  hdr->region_size = L.total;
  hdr->magic = DH_MAGIC;

  dh->hdr = hdr;
  return DH_OK;
}

/* -------------------------------------------------------------------------
 * dense_heap_create
 * ---------------------------------------------------------------------- */

DH_COLD dh_status_t dense_heap_create(dense_heap_t *out, size_t elem_size,
                                      size_t initial_cap) {
  if (DH_UNLIKELY(!out || elem_size == 0))
    return DH_ERR_INVAL;

  if (initial_cap == 0)
    initial_cap = DH_DEFAULT_CAPACITY;

  out->hdr = NULL;
  return _dh_alloc_region(out, initial_cap, elem_size, 0);
}

/* -------------------------------------------------------------------------
 * dense_heap_destroy
 * ---------------------------------------------------------------------- */

DH_COLD dh_status_t dense_heap_destroy(dense_heap_t *dh) {
  if (DH_UNLIKELY(!dh || !dh->hdr))
    return DH_ERR_INVAL;

  assert(dh->hdr->magic == DH_MAGIC);
  int r = munmap(dh->hdr, dh->hdr->region_size);
  dh->hdr = NULL;
  return (r == 0) ? DH_OK : DH_ERR_NOMEM;
}

/* -------------------------------------------------------------------------
 * dense_heap_reserve  (grow only)
 * ---------------------------------------------------------------------- */

DH_COLD dh_status_t dense_heap_reserve(dense_heap_t *dh, size_t new_cap) {
  if (DH_UNLIKELY(!dh || !dh->hdr))
    return DH_ERR_INVAL;
  if (new_cap <= dh->hdr->capacity)
    return DH_OK;

  return _dh_alloc_region(dh, new_cap, dh->hdr->elem_size, dh->hdr->size);
}

/* -------------------------------------------------------------------------
 * Internal: grow by 1.5x (at least by 1)
 * ---------------------------------------------------------------------- */

static DH_NOINLINE dh_status_t _dh_grow(dense_heap_t *dh) {
  size_t old_cap = dh->hdr->capacity;

  /* overflow check */
  if (DH_UNLIKELY(old_cap > SIZE_MAX / DH_GROW_NUM))
    return DH_ERR_OVERFLOW;

  size_t new_cap = old_cap * DH_GROW_NUM / DH_GROW_DEN;
  if (new_cap <= old_cap)
    new_cap = old_cap + 1; /* handle tiny capacities */

  return _dh_alloc_region(dh, new_cap, dh->hdr->elem_size, dh->hdr->size);
}

/* -------------------------------------------------------------------------
 * dense_heap_push  — O(1) amortised
 * ---------------------------------------------------------------------- */

DH_HOT dh_status_t dense_heap_push(dense_heap_t *dh, const void *elem) {
  if (DH_UNLIKELY(!dh || !dh->hdr || !elem))
    return DH_ERR_INVAL;

  assert(dh->hdr->magic == DH_MAGIC);

  /* grow if needed */
  if (DH_UNLIKELY(dh->hdr->size == dh->hdr->capacity)) {
    dh_status_t s = _dh_grow(dh);
    if (DH_UNLIKELY(s != DH_OK))
      return s;
  }

  dh_header_t *hdr = dh->hdr;
  size_t phys = hdr->size;
  size_t es = hdr->elem_size;
  char *data = (char *)hdr + hdr->data_off;
  size_t *idx = (size_t *)((char *)hdr + hdr->index_off);
  size_t *rev = (size_t *)((char *)hdr + hdr->reverse_off);

  /* copy element into physical slot */
  memcpy(data + phys * es, elem, es);

  /* logical position == current size (append at logical end) */
  idx[phys] = phys;
  rev[phys] = phys;

  hdr->size = phys + 1;
  return DH_OK;
}

/* -------------------------------------------------------------------------
 * dense_heap_get / get_ptr / set  — O(1)
 * ---------------------------------------------------------------------- */

DH_HOT dh_status_t dense_heap_get(const dense_heap_t *dh, size_t logical_idx,
                                  void *out_elem) {
  if (DH_UNLIKELY(!dh || !dh->hdr || !out_elem))
    return DH_ERR_INVAL;

  dh_header_t *hdr = dh->hdr;
  if (DH_UNLIKELY(logical_idx >= hdr->size))
    return DH_ERR_BOUNDS;

  size_t phys = _dh_index(dh)[logical_idx];
  DH_PREFETCH(_dh_data(dh) + phys * hdr->elem_size);
  memcpy(out_elem, _dh_data(dh) + phys * hdr->elem_size, hdr->elem_size);
  return DH_OK;
}

DH_HOT dh_status_t dense_heap_get_ptr(const dense_heap_t *dh,
                                      size_t logical_idx, void **out_ptr) {
  if (DH_UNLIKELY(!dh || !dh->hdr || !out_ptr))
    return DH_ERR_INVAL;

  dh_header_t *hdr = dh->hdr;
  if (DH_UNLIKELY(logical_idx >= hdr->size))
    return DH_ERR_BOUNDS;

  size_t phys = _dh_index(dh)[logical_idx];
  *out_ptr = _dh_data(dh) + phys * hdr->elem_size;
  return DH_OK;
}

DH_HOT dh_status_t dense_heap_set(dense_heap_t *dh, size_t logical_idx,
                                  const void *elem) {
  if (DH_UNLIKELY(!dh || !dh->hdr || !elem))
    return DH_ERR_INVAL;

  dh_header_t *hdr = dh->hdr;
  if (DH_UNLIKELY(logical_idx >= hdr->size))
    return DH_ERR_BOUNDS;

  size_t phys = _dh_index(dh)[logical_idx];
  memcpy(_dh_data(dh) + phys * hdr->elem_size, elem, hdr->elem_size);
  return DH_OK;
}

/* -------------------------------------------------------------------------
 * dense_heap_remove_out  — core removal, O(1)
 *
 * Swap-removes element at logical_idx:
 *   1. locate physical slot of target
 *   2. locate physical slot of last element
 *   3. memcpy last → target slot in data[]
 *   4. fix up index[] and reverse[] for moved element
 *   5. shrink size
 * ---------------------------------------------------------------------- */

DH_HOT dh_status_t dense_heap_remove_out(dense_heap_t *dh, size_t logical_idx,
                                         void *out_elem) {
  if (DH_UNLIKELY(!dh || !dh->hdr))
    return DH_ERR_INVAL;

  dh_header_t *hdr = dh->hdr;
  if (DH_UNLIKELY(logical_idx >= hdr->size))
    return DH_ERR_BOUNDS;
  if (DH_UNLIKELY(hdr->size == 0))
    return DH_ERR_EMPTY;

  size_t es = hdr->elem_size;
  char *data = (char *)hdr + hdr->data_off;
  size_t *idx = (size_t *)((char *)hdr + hdr->index_off);
  size_t *rev = (size_t *)((char *)hdr + hdr->reverse_off);
  size_t last_log = hdr->size - 1;

  size_t phys_k = idx[logical_idx];
  size_t phys_last = idx[last_log];

  /* optionally return removed element to caller */
  if (out_elem)
    memcpy(out_elem, data + phys_k * es, es);

  /* only do the swap if we're not removing the last logical element */
  if (DH_LIKELY(logical_idx != last_log)) {
    /*
     * Swap-remove in physical storage:
     * move last physical element → freed physical slot
     */
    memcpy(data + phys_k * es, data + phys_last * es, es);

    /*
     * The last-logical element now lives at phys_k.
     * Update its logical-position's index entry.
     */
    idx[last_log] = phys_k;
    rev[phys_k] = last_log;

    /*
     * Plug the removed logical slot with the last logical entry,
     * then drop the last logical slot.
     */
    idx[logical_idx] = idx[last_log]; /* == phys_k, already set above */
    /* rev for phys_k already points to last_log */

    /*
     * Now remove the logical slot at logical_idx by overwriting it
     * with the entry from last_log, and shrink.
     *
     * Wait — let's be precise per the spec:
     *
     *   idx[logical_idx] = idx[size-1]   (the moved element's new phys)
     *   rev[idx[logical_idx]] = logical_idx
     *
     * Since idx[size-1] = phys_k (we just set it), and we want
     * logical_idx to now refer to what was last_log's element:
     */
    idx[logical_idx] = phys_k;
    rev[phys_k] = logical_idx;
  }

  hdr->size = last_log;
  return DH_OK;
}

DH_HOT dh_status_t dense_heap_remove(dense_heap_t *dh, size_t logical_idx) {
  return dense_heap_remove_out(dh, logical_idx, NULL);
}

/* -------------------------------------------------------------------------
 * dense_heap_pop_logical  — remove last logical element, O(1)
 * ---------------------------------------------------------------------- */

DH_HOT dh_status_t dense_heap_pop_logical(dense_heap_t *dh, void *out_elem) {
  if (DH_UNLIKELY(!dh || !dh->hdr))
    return DH_ERR_INVAL;
  if (DH_UNLIKELY(dh->hdr->size == 0))
    return DH_ERR_EMPTY;

  return dense_heap_remove_out(dh, dh->hdr->size - 1, out_elem);
}

/* -------------------------------------------------------------------------
 * dense_heap_swap_logical  — swap two logical positions, O(1)
 * ---------------------------------------------------------------------- */

DH_HOT dh_status_t dense_heap_swap_logical(dense_heap_t *dh, size_t a,
                                           size_t b) {
  if (DH_UNLIKELY(!dh || !dh->hdr))
    return DH_ERR_INVAL;

  dh_header_t *hdr = dh->hdr;
  if (DH_UNLIKELY(a >= hdr->size || b >= hdr->size))
    return DH_ERR_BOUNDS;
  if (a == b)
    return DH_OK;

  size_t *idx = (size_t *)((char *)hdr + hdr->index_off);
  size_t *rev = (size_t *)((char *)hdr + hdr->reverse_off);

  size_t pa = idx[a];
  size_t pb = idx[b];

  /* swap index entries */
  idx[a] = pb;
  idx[b] = pa;

  /* fix reverse */
  rev[pa] = b;
  rev[pb] = a;

  return DH_OK;
}
