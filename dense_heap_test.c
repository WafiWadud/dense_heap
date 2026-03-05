#define _POSIX_C_SOURCE 200809L
/*
 * dense_heap_test.c — comprehensive tests + mini benchmark
 */

#include "dense_heap.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* -------------------------------------------------------------------------
 * tiny test harness
 * ---------------------------------------------------------------------- */

static int _tests_run = 0;
static int _tests_failed = 0;

#define CHECK(cond, msg)                                                       \
  do {                                                                         \
    ++_tests_run;                                                              \
    if (!(cond)) {                                                             \
      ++_tests_failed;                                                         \
      fprintf(stderr, "FAIL [%s:%d] %s\n", __FILE__, __LINE__, msg);           \
    }                                                                          \
  } while (0)

#define CHECK_EQ(a, b, msg) CHECK((a) == (b), msg)

/* -------------------------------------------------------------------------
 * helpers
 * ---------------------------------------------------------------------- */

static void print_state(const dense_heap_t *dh, const char *label) {
  printf("  [%s] size=%zu  logical: ", label, dense_heap_size(dh));
  for (size_t i = 0; i < dense_heap_size(dh); ++i) {
    int v;
    dense_heap_get(dh, i, &v);
    printf("%d ", v);
  }
  printf("\n");
}

/* -------------------------------------------------------------------------
 * test_basic_push_get
 * ---------------------------------------------------------------------- */

static void test_basic_push_get(void) {
  printf("test_basic_push_get\n");
  dense_heap_t dh;
  assert(dense_heap_create(&dh, sizeof(int), 8) == DH_OK);

  for (int i = 0; i < 5; ++i)
    assert(dense_heap_push(&dh, &i) == DH_OK);

  CHECK_EQ(dense_heap_size(&dh), 5u, "size after 5 pushes");

  for (int i = 0; i < 5; ++i) {
    int v = -1;
    dense_heap_get(&dh, (size_t)i, &v);
    CHECK_EQ(v, i, "logical order preserved after push");
  }

  print_state(&dh, "after push 0..4");
  dense_heap_destroy(&dh);
}

/* -------------------------------------------------------------------------
 * test_remove_middle
 * ---------------------------------------------------------------------- */

static void test_remove_middle(void) {
  printf("test_remove_middle\n");
  dense_heap_t dh;
  assert(dense_heap_create(&dh, sizeof(int), 8) == DH_OK);

  /* push 0,1,2,3,4 */
  for (int i = 0; i < 5; ++i)
    dense_heap_push(&dh, &i);

  print_state(&dh, "before remove[2]");

  /* remove logical index 2 (value 2) */
  int removed = -99;
  assert(dense_heap_remove_out(&dh, 2, &removed) == DH_OK);
  CHECK_EQ(removed, 2, "removed value == 2");
  CHECK_EQ(dense_heap_size(&dh), 4u, "size after remove");

  print_state(&dh, "after remove[2]");

  /* remaining elements should be {0,1,3,4} in logical order
   * BUT the spec says last logical element fills the removed slot,
   * so we expect {0,1,4,3} */
  int expected[] = {0, 1, 4, 3};
  for (int i = 0; i < 4; ++i) {
    int v = -1;
    dense_heap_get(&dh, (size_t)i, &v);
    CHECK_EQ(v, expected[i], "value after swap-remove");
  }

  dense_heap_destroy(&dh);
}

/* -------------------------------------------------------------------------
 * test_pop_logical
 * ---------------------------------------------------------------------- */

static void test_pop_logical(void) {
  printf("test_pop_logical\n");
  dense_heap_t dh;
  assert(dense_heap_create(&dh, sizeof(int), 8) == DH_OK);

  for (int i = 0; i < 4; ++i)
    dense_heap_push(&dh, &i);

  int v;
  assert(dense_heap_pop_logical(&dh, &v) == DH_OK);
  CHECK_EQ(v, 3, "popped last logical == 3");
  CHECK_EQ(dense_heap_size(&dh), 3u, "size after pop");

  dense_heap_destroy(&dh);
}

/* -------------------------------------------------------------------------
 * test_grow
 * ---------------------------------------------------------------------- */

static void test_grow(void) {
  printf("test_grow\n");
  dense_heap_t dh;
  assert(dense_heap_create(&dh, sizeof(int), 4) == DH_OK);

  /* push 16 elements, forcing multiple grows */
  for (int i = 0; i < 16; ++i)
    assert(dense_heap_push(&dh, &i) == DH_OK);

  CHECK_EQ(dense_heap_size(&dh), 16u, "size after 16 pushes");

  for (int i = 0; i < 16; ++i) {
    int v = -1;
    dense_heap_get(&dh, (size_t)i, &v);
    CHECK_EQ(v, i, "value preserved across grow");
  }

  dense_heap_destroy(&dh);
}

/* -------------------------------------------------------------------------
 * test_swap_logical
 * ---------------------------------------------------------------------- */

static void test_swap_logical(void) {
  printf("test_swap_logical\n");
  dense_heap_t dh;
  assert(dense_heap_create(&dh, sizeof(int), 8) == DH_OK);

  for (int i = 0; i < 5; ++i)
    dense_heap_push(&dh, &i);

  print_state(&dh, "before swap(1,3)");
  assert(dense_heap_swap_logical(&dh, 1, 3) == DH_OK);
  print_state(&dh, "after  swap(1,3)");

  int a, b;
  dense_heap_get(&dh, 1, &a);
  dense_heap_get(&dh, 3, &b);
  CHECK_EQ(a, 3, "logical[1] == 3 after swap");
  CHECK_EQ(b, 1, "logical[3] == 1 after swap");

  dense_heap_destroy(&dh);
}

/* -------------------------------------------------------------------------
 * test_get_ptr
 * ---------------------------------------------------------------------- */

static void test_get_ptr(void) {
  printf("test_get_ptr\n");
  dense_heap_t dh;
  assert(dense_heap_create(&dh, sizeof(int), 8) == DH_OK);

  int val = 42;
  dense_heap_push(&dh, &val);

  void *ptr = NULL;
  assert(dense_heap_get_ptr(&dh, 0, &ptr) == DH_OK);
  CHECK(ptr != NULL, "get_ptr returns non-null");
  CHECK_EQ(*(int *)ptr, 42, "get_ptr value correct");

  /* mutate through pointer */
  *(int *)ptr = 99;
  int v2;
  dense_heap_get(&dh, 0, &v2);
  CHECK_EQ(v2, 99, "mutation through pointer reflected");

  dense_heap_destroy(&dh);
}

/* -------------------------------------------------------------------------
 * test_error_codes
 * ---------------------------------------------------------------------- */

static void test_error_codes(void) {
  printf("test_error_codes\n");
  dense_heap_t dh;
  assert(dense_heap_create(&dh, sizeof(int), 4) == DH_OK);

  int dummy;
  CHECK_EQ(dense_heap_get(&dh, 0, &dummy), DH_ERR_BOUNDS,
           "get on empty → BOUNDS");
  CHECK_EQ(dense_heap_pop_logical(&dh, &dummy), DH_ERR_EMPTY,
           "pop on empty → EMPTY");
  CHECK_EQ(dense_heap_remove(&dh, 0), DH_ERR_BOUNDS,
           "remove on empty → BOUNDS");

  int v = 7;
  dense_heap_push(&dh, &v);
  CHECK_EQ(dense_heap_get(&dh, 1, &dummy), DH_ERR_BOUNDS,
           "get out-of-bounds → BOUNDS");

  dense_heap_destroy(&dh);
}

/* -------------------------------------------------------------------------
 * test_physical_iteration_macro
 * ---------------------------------------------------------------------- */

static void test_physical_iteration(void) {
  printf("test_physical_iteration\n");
  dense_heap_t dh;
  assert(dense_heap_create(&dh, sizeof(int), 8) == DH_OK);

  int sum_expected = 0;
  for (int i = 1; i <= 5; ++i) {
    dense_heap_push(&dh, &i);
    sum_expected += i;
  }

  int sum = 0;
  int *p;
  DH_FOREACH_PHYSICAL(&dh, int, p) { sum += *p; }
  CHECK_EQ(sum, sum_expected, "physical iteration sum correct");

  dense_heap_destroy(&dh);
}

/* -------------------------------------------------------------------------
 * test_logical_iteration_macro
 * ---------------------------------------------------------------------- */

static void test_logical_iteration(void) {
  printf("test_logical_iteration\n");
  dense_heap_t dh;
  assert(dense_heap_create(&dh, sizeof(int), 8) == DH_OK);

  for (int i = 0; i < 5; ++i)
    dense_heap_push(&dh, &i);

  /* swap so logical order is different from push order */
  dense_heap_swap_logical(&dh, 0, 4); /* {4,1,2,3,0} */

  int *p;
  int count = 0;
  DH_FOREACH_LOGICAL(&dh, int, p) {
    (void)p;
    ++count;
  }
  CHECK_EQ(count, 5, "logical iteration visits all 5 elements");

  dense_heap_destroy(&dh);
}

/* -------------------------------------------------------------------------
 * test_large_struct
 * ---------------------------------------------------------------------- */

typedef struct {
  char buf[256];
  int id;
} BigThing;

static void test_large_struct(void) {
  printf("test_large_struct\n");
  dense_heap_t dh;
  assert(dense_heap_create(&dh, sizeof(BigThing), 16) == DH_OK);

  for (int i = 0; i < 10; ++i) {
    BigThing bt;
    snprintf(bt.buf, sizeof(bt.buf), "item_%d", i);
    bt.id = i;
    dense_heap_push(&dh, &bt);
  }

  BigThing out;
  dense_heap_get(&dh, 5, &out);
  CHECK_EQ(out.id, 5, "large struct get by logical index");

  dense_heap_remove(&dh, 5);
  CHECK_EQ(dense_heap_size(&dh), 9u, "size after remove large struct");

  dense_heap_destroy(&dh);
}

/* -------------------------------------------------------------------------
 * benchmark: push + physical iterate + remove-all
 * ---------------------------------------------------------------------- */

#define BENCH_N 1000000

static double _now_sec(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void benchmark(void) {
  printf("\n--- benchmark (N=%d) ---\n", BENCH_N);
  dense_heap_t dh;
  assert(dense_heap_create(&dh, sizeof(int), (size_t)BENCH_N) == DH_OK);

  double t0 = _now_sec();
  for (int i = 0; i < BENCH_N; ++i)
    dense_heap_push(&dh, &i);
  double push_ms = (_now_sec() - t0) * 1000.0;

  /* physical iteration */
  volatile long long sum = 0;
  t0 = _now_sec();
  int *p;
  DH_FOREACH_PHYSICAL(&dh, int, p) { sum += *p; }
  double phys_ms = (_now_sec() - t0) * 1000.0;

  /* logical iteration */
  volatile long long sum2 = 0;
  t0 = _now_sec();
  int *q;
  DH_FOREACH_LOGICAL(&dh, int, q) { sum2 += *q; }
  double log_ms = (_now_sec() - t0) * 1000.0;

  /* remove all from front (worst-case logical remove) */
  t0 = _now_sec();
  while (!dense_heap_empty(&dh))
    dense_heap_remove(&dh, 0);
  double rem_ms = (_now_sec() - t0) * 1000.0;

  printf("  push %dM elems   : %.2f ms  (%.1f M/s)\n", BENCH_N / 1000000,
         push_ms, BENCH_N / push_ms / 1000.0);
  printf("  physical iterate : %.2f ms  (sum=%lld)\n", phys_ms, sum);
  printf("  logical  iterate : %.2f ms  (sum=%lld)\n", log_ms, sum2);
  printf("  remove-all front : %.2f ms\n", rem_ms);

  dense_heap_destroy(&dh);
}

/* -------------------------------------------------------------------------
 * main
 * ---------------------------------------------------------------------- */

int main(void) {
  printf("=== dense_heap tests ===\n\n");

  test_basic_push_get();
  test_remove_middle();
  test_pop_logical();
  test_grow();
  test_swap_logical();
  test_get_ptr();
  test_error_codes();
  test_physical_iteration();
  test_logical_iteration();
  test_large_struct();

  printf("\n%d/%d tests passed\n", _tests_run - _tests_failed, _tests_run);

  if (_tests_failed)
    printf("*** %d FAILURES ***\n", _tests_failed);
  else
    printf("All tests OK.\n");

  benchmark();

  return _tests_failed ? 1 : 0;
}
