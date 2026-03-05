#pragma once

/*
 * dense_heap.h — Dense contiguous container with logical order mapping
 *
 * Layout (single mmap region):
 *   [ header | data[capacity] | index[capacity] | reverse[capacity] ]
 *
 * - data[]    : physical element storage, always dense (no holes)
 * - index[]   : logical → physical  (index[i] gives slot in data[])
 * - reverse[] : physical → logical  (reverse[phys] gives logical index)
 *
 * All core ops are O(1). Iteration is O(n) in either logical or physical order.
 */

#include <assert.h>
#include <errno.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <sys/mman.h>

/* -------------------------------------------------------------------------
 * Tunables
 * ---------------------------------------------------------------------- */

/* Default initial capacity if none given */
#define DH_DEFAULT_CAPACITY 64u

/* Growth factor numerator/denominator  (1.5x = 3/2) */
#define DH_GROW_NUM 3u
#define DH_GROW_DEN 2u

/* Cache line size assumed for alignment */
#define DH_CACHELINE 64u

/* -------------------------------------------------------------------------
 * Compiler portability helpers
 * ---------------------------------------------------------------------- */

#if defined(__GNUC__) || defined(__clang__)
#define DH_LIKELY(x) __builtin_expect(!!(x), 1)
#define DH_UNLIKELY(x) __builtin_expect(!!(x), 0)
#define DH_INLINE __attribute__((always_inline)) static inline
#define DH_NOINLINE __attribute__((noinline))
#define DH_PURE __attribute__((pure))
#define DH_HOT __attribute__((hot))
#define DH_COLD __attribute__((cold))
#define DH_RESTRICT __restrict__
#define DH_PREFETCH(p) __builtin_prefetch((p), 0, 3)
#define DH_PREFETCH_W(p) __builtin_prefetch((p), 1, 3)
#else
#define DH_LIKELY(x) (x)
#define DH_UNLIKELY(x) (x)
#define DH_INLINE static inline
#define DH_NOINLINE
#define DH_PURE
#define DH_HOT
#define DH_COLD
#define DH_RESTRICT
#define DH_PREFETCH(p) (void)(p)
#define DH_PREFETCH_W(p) (void)(p)
#endif

/* -------------------------------------------------------------------------
 * Result / error codes
 * ---------------------------------------------------------------------- */

typedef enum {
  DH_OK = 0,
  DH_ERR_NOMEM = -1,    /* mmap / realloc failed              */
  DH_ERR_BOUNDS = -2,   /* logical index out of range         */
  DH_ERR_EMPTY = -3,    /* pop/peek on empty container        */
  DH_ERR_OVERFLOW = -4, /* capacity would exceed SIZE_MAX     */
  DH_ERR_INVAL = -5,    /* null pointer or zero element_size  */
} dh_status_t;

/* -------------------------------------------------------------------------
 * Header — lives at the very start of the mmap region
 * ---------------------------------------------------------------------- */

typedef struct {
  size_t capacity;    /* max elements the current region can hold */
  size_t size;        /* current element count                     */
  size_t elem_size;   /* sizeof one element (bytes)                */
  size_t data_off;    /* byte offset of data array from base       */
  size_t index_off;   /* byte offset of index array from base      */
  size_t reverse_off; /* byte offset of reverse array from base    */
  size_t region_size; /* total bytes in the mmap region            */
  uint32_t magic;     /* sanity check: DH_MAGIC                   */
  uint32_t _pad;
} dh_header_t;

#define DH_MAGIC 0xDEA1BEEFu

/* -------------------------------------------------------------------------
 * Handle — tiny struct held by caller; everything else lives in the mapping
 * ---------------------------------------------------------------------- */

typedef struct {
  dh_header_t *hdr; /* pointer == base of mmap region */
} dense_heap_t;

/* -------------------------------------------------------------------------
 * Internal helpers (access arrays through offsets)
 * ---------------------------------------------------------------------- */

DH_INLINE DH_PURE char *_dh_data(const dense_heap_t *dh) {
  return (char *)dh->hdr + dh->hdr->data_off;
}

DH_INLINE DH_PURE size_t *_dh_index(const dense_heap_t *dh) {
  return (size_t *)((char *)dh->hdr + dh->hdr->index_off);
}

DH_INLINE DH_PURE size_t *_dh_reverse(const dense_heap_t *dh) {
  return (size_t *)((char *)dh->hdr + dh->hdr->reverse_off);
}

/* -------------------------------------------------------------------------
 * Public API declarations
 * ---------------------------------------------------------------------- */

/* Lifecycle */
dh_status_t dense_heap_create(dense_heap_t *out, size_t elem_size,
                              size_t initial_cap);
dh_status_t dense_heap_destroy(dense_heap_t *dh);

/* Core ops — all O(1) */
dh_status_t dense_heap_push(dense_heap_t *dh, const void *elem);
dh_status_t dense_heap_pop_logical(dense_heap_t *dh,
                                   void *out_elem); /* removes last logical */
dh_status_t dense_heap_remove(dense_heap_t *dh, size_t logical_idx);
dh_status_t dense_heap_remove_out(dense_heap_t *dh, size_t logical_idx,
                                  void *out_elem);

/* Access — O(1) */
dh_status_t dense_heap_get(const dense_heap_t *dh, size_t logical_idx,
                           void *out_elem);
dh_status_t dense_heap_get_ptr(const dense_heap_t *dh, size_t logical_idx,
                               void **out_ptr);
dh_status_t dense_heap_set(dense_heap_t *dh, size_t logical_idx,
                           const void *elem);

/* Swap logical positions — O(1) */
dh_status_t dense_heap_swap_logical(dense_heap_t *dh, size_t a, size_t b);

/* Capacity management */
dh_status_t dense_heap_reserve(dense_heap_t *dh, size_t new_cap);

/* Queries */
DH_INLINE DH_PURE size_t dense_heap_size(const dense_heap_t *dh) {
  return dh->hdr->size;
}
DH_INLINE DH_PURE size_t dense_heap_capacity(const dense_heap_t *dh) {
  return dh->hdr->capacity;
}
DH_INLINE DH_PURE bool dense_heap_empty(const dense_heap_t *dh) {
  return dh->hdr->size == 0;
}

/* Iteration helpers */

/*
 * Logical iteration macro — respects insertion order.
 *   elem_ptr: (ElemType*) pointer set each iteration
 *   dh_ptr  : (dense_heap_t*)
 *   ElemType: concrete type
 *
 * Usage:
 *   DH_FOREACH_LOGICAL(dh, MyType, p) { printf("%d\n", p->field); }
 */
#define DH_FOREACH_LOGICAL(dh_ptr, ElemType, elem_ptr)                         \
  for (size_t _dh_i = 0, _dh_n = (dh_ptr)->hdr->size,                          \
              *_dh_idx = _dh_index(dh_ptr);                                    \
       _dh_i < _dh_n &&                                                        \
       ((elem_ptr) = (ElemType *)(_dh_data(dh_ptr) +                           \
                                  _dh_idx[_dh_i] * (dh_ptr)->hdr->elem_size),  \
       1);                                                                     \
       ++_dh_i)

/*
 * Physical iteration macro — fastest, order unspecified.
 *   elem_ptr: (ElemType*) pointer set each iteration
 *   dh_ptr  : (dense_heap_t*)
 *   ElemType: concrete type
 *
 * Usage:
 *   DH_FOREACH_PHYSICAL(dh, MyType, p) { process(p); }
 */
#define DH_FOREACH_PHYSICAL(dh_ptr, ElemType, elem_ptr)                        \
  for (size_t _dh_i = 0, _dh_n = (dh_ptr)->hdr->size;                          \
       _dh_i < _dh_n &&                                                        \
       ((elem_ptr) =                                                           \
            (ElemType *)(_dh_data(dh_ptr) + _dh_i * (dh_ptr)->hdr->elem_size), \
       1);                                                                     \
       ++_dh_i)
