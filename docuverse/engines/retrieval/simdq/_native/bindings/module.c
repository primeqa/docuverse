// _simdq_native — CPython C-API bindings for simdq kernels.
//
// Exposes:
//   compute_scales(Y: float32[N, D]) -> float32[N]
//   pack_b1(Y: float32[N, D], scales: float32[N]) -> uint8[D * ceil(N/8)]
//   pack_b2(Y: float32[N, D], scales: float32[N]) -> uint8[D * ceil(N/4)]
//   pack_b4(Y: float32[N, D], scales: float32[N]) -> uint8[D * ceil(N/2)]
//   scan_b1(codes: uint8, N: int, D: int, q: float32[D], K: int, num_threads: int)
//                                -> (scores: float32[K], indices: int64[K])
//   scan_b2(...)
//   scan_b4(...)
//   pack_hamming(Y: float32[N, D]) -> uint8[N * D/8]  (AoS layout, D/8 bytes per code)
//   scan_hamming(codes: uint8, N: int, D: int, q: uint8[D/8], K: int, num_threads: int)
//                                -> (distances: int64[K], indices: int64[K])
//
// D is a runtime parameter taken from each input array's shape (compute_scales,
// pack_b{1,2,4}, pack_hamming) or passed as an explicit argument (scan_b{1,2,4},
// scan_hamming). Supported D values: {192, 384, 512, 768, 1024, 1536}.
// All numpy arrays must be C-contiguous and dtype as listed; the binding raises
// ValueError otherwise. The scan functions release the GIL around the kernel call
// and honor num_threads via omp_set_num_threads.

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// Pull in all four kernel headers in this single TU. The Plan 2 T1 macro
// renames make this safe.
#include "simdq_pack.h"
#include "simdq_kernels_hamming_topk.h"
#include "simdq_kernels_asym_b1.h"
#include "simdq_kernels_asym_b2.h"
#include "simdq_kernels_asym_b4.h"
#include "fagin_ta.h"

/*
 * Validate that d is one of the supported dimensionalities.
 * Returns 0 on success, -1 on failure (with PyExc_ValueError set).
 */
static int simdq_check_d(Py_ssize_t d) {
    static const Py_ssize_t supported[] = {192, 384, 512, 768, 1024, 1536};
    for (size_t i = 0; i < sizeof(supported)/sizeof(supported[0]); i++)
        if (supported[i] == d) return 0;
    PyErr_Format(PyExc_ValueError,
                 "d must be one of {192, 384, 512, 768, 1024, 1536}; got %zd", d);
    return -1;
}

/*
 * Helper: extract a contiguous, typed buffer view from a Python object.
 * Returns 0 on success, -1 on failure (with a Python exception set).
 * The caller MUST call PyBuffer_Release on the view when done.
 */
static int get_buffer(PyObject *obj, Py_buffer *view, char itemkind, int writable) {
    int flags = PyBUF_C_CONTIGUOUS | PyBUF_FORMAT;
    if (writable) flags |= PyBUF_WRITABLE;
    if (PyObject_GetBuffer(obj, view, flags) != 0) return -1;
    if (!view->format) {
        PyErr_SetString(PyExc_TypeError, "buffer has no format");
        PyBuffer_Release(view);
        return -1;
    }
    char fmt = view->format[0];
    // Accept both '<' / '=' / '@' prefixes.
    if (fmt == '<' || fmt == '>' || fmt == '=' || fmt == '@') fmt = view->format[1];
    if (fmt != itemkind) {
        PyErr_Format(PyExc_TypeError,
                     "expected dtype with format '%c', got '%s'",
                     itemkind, view->format);
        PyBuffer_Release(view);
        return -1;
    }
    return 0;
}

/*
 * Allocate a new bytes object of the given size, return a writable buffer
 * view into it via *out_data, and return the bytes object (not yet
 * refcounted for return — caller decides). On failure returns NULL with
 * exception set.
 */
static PyObject *new_bytes_buffer(Py_ssize_t nbytes, char **out_data) {
    PyObject *b = PyBytes_FromStringAndSize(NULL, nbytes);
    if (!b) return NULL;
    *out_data = PyBytes_AS_STRING(b);
    memset(*out_data, 0, (size_t)nbytes);
    return b;
}

// ---------- compute_scales ----------

static PyObject *py_compute_scales(PyObject *self, PyObject *args) {
    PyObject *y_obj;
    if (!PyArg_ParseTuple(args, "O", &y_obj)) return NULL;

    Py_buffer y_view;
    if (get_buffer(y_obj, &y_view, 'f', 0) != 0) return NULL;
    if (y_view.ndim != 2) {
        PyErr_Format(PyExc_ValueError,
                     "Y must be 2-D, got ndim=%d", y_view.ndim);
        PyBuffer_Release(&y_view);
        return NULL;
    }
    Py_ssize_t d = y_view.shape[1];
    if (simdq_check_d(d) != 0) { PyBuffer_Release(&y_view); return NULL; }

    Py_ssize_t N = y_view.shape[0];
    char *out_data;
    PyObject *out = new_bytes_buffer(N * (Py_ssize_t)sizeof(float), &out_data);
    if (!out) { PyBuffer_Release(&y_view); return NULL; }
    float *scales = (float *)out_data;
    const float *Y = (const float *)y_view.buf;

    Py_BEGIN_ALLOW_THREADS
    const float inv_sqrt_d = 1.0f / sqrtf((float)d);
    #pragma omp parallel for
    for (Py_ssize_t i = 0; i < N; i++) {
        const float *yi = Y + (size_t)i * d;
        float s = 0.0f;
        for (Py_ssize_t w = 0; w < d; w++) s += yi[w] * yi[w];
        s = sqrtf(s) * inv_sqrt_d;
        scales[i] = (s == 0.0f) ? 1.0f : s;
    }
    Py_END_ALLOW_THREADS

    PyBuffer_Release(&y_view);
    return out;
}

// ---------- pack helpers ----------

static PyObject *do_pack(PyObject *args, int b) {
    PyObject *y_obj, *s_obj;
    if (!PyArg_ParseTuple(args, "OO", &y_obj, &s_obj)) return NULL;
    Py_buffer y_view, s_view;
    if (get_buffer(y_obj, &y_view, 'f', 0) != 0) return NULL;
    if (get_buffer(s_obj, &s_view, 'f', 0) != 0) {
        PyBuffer_Release(&y_view); return NULL;
    }
    if (y_view.ndim != 2) {
        PyErr_Format(PyExc_ValueError, "Y must be 2-D");
        goto fail;
    }
    Py_ssize_t d = y_view.shape[1];
    if (simdq_check_d(d) != 0) goto fail;

    Py_ssize_t N = y_view.shape[0];
    if (s_view.ndim != 1 || s_view.shape[0] != N) {
        PyErr_Format(PyExc_ValueError,
                     "scales must have shape (%zd,)", N);
        goto fail;
    }
    Py_ssize_t row_bytes;
    if      (b == 1) row_bytes = (N + 7) / 8;
    else if (b == 2) row_bytes = (N + 3) / 4;
    else             row_bytes = (N + 1) / 2;
    Py_ssize_t total = d * row_bytes;

    char *out_data;
    PyObject *out = new_bytes_buffer(total, &out_data);
    if (!out) goto fail;

    Py_BEGIN_ALLOW_THREADS
    if (b == 1)      simdq_pack_b1((const float *)y_view.buf, (size_t)N, (size_t)d,
                                   (const float *)s_view.buf, (uint8_t *)out_data);
    else if (b == 2) simdq_pack_b2((const float *)y_view.buf, (size_t)N, (size_t)d,
                                   (const float *)s_view.buf, (uint8_t *)out_data);
    else             simdq_pack_b4((const float *)y_view.buf, (size_t)N, (size_t)d,
                                   (const float *)s_view.buf, (uint8_t *)out_data);
    Py_END_ALLOW_THREADS

    PyBuffer_Release(&y_view);
    PyBuffer_Release(&s_view);
    return out;

fail:
    PyBuffer_Release(&y_view);
    PyBuffer_Release(&s_view);
    return NULL;
}

static PyObject *py_pack_b1(PyObject *self, PyObject *args) { return do_pack(args, 1); }
static PyObject *py_pack_b2(PyObject *self, PyObject *args) { return do_pack(args, 2); }
static PyObject *py_pack_b4(PyObject *self, PyObject *args) { return do_pack(args, 4); }

// ---------- scan helpers ----------

static PyObject *do_scan(PyObject *args, int b) {
    PyObject *codes_obj, *q_obj;
    Py_ssize_t N, d, K, num_threads;
    if (!PyArg_ParseTuple(args, "OnnOnn",
                          &codes_obj, &N, &d, &q_obj, &K, &num_threads)) return NULL;
    if (K <= 0 || K > 256) {
        PyErr_SetString(PyExc_ValueError, "K must be in [1, 256]");
        return NULL;
    }
    if (simdq_check_d(d) != 0) return NULL;

    Py_buffer codes_view, q_view;
    if (get_buffer(codes_obj, &codes_view, 'B', 0) != 0) return NULL;
    if (get_buffer(q_obj, &q_view, 'f', 0) != 0) {
        PyBuffer_Release(&codes_view); return NULL;
    }
    if (q_view.ndim != 1 || q_view.shape[0] != d) {
        PyErr_Format(PyExc_ValueError, "q must have shape (%zd,)", d);
        PyBuffer_Release(&codes_view); PyBuffer_Release(&q_view); return NULL;
    }
    Py_ssize_t row_bytes;
    if      (b == 1) row_bytes = (N + 7) / 8;
    else if (b == 2) row_bytes = (N + 3) / 4;
    else             row_bytes = (N + 1) / 2;
    Py_ssize_t expect = d * row_bytes;
    if (codes_view.shape[0] < expect) {
        PyErr_Format(PyExc_ValueError,
                     "codes buffer too small: got %zd bytes, need >= %zd "
                     "for N=%zd b=%d D=%zd",
                     codes_view.shape[0], expect, N, b, d);
        PyBuffer_Release(&codes_view); PyBuffer_Release(&q_view); return NULL;
    }

    char *scores_data, *idx_data;
    PyObject *scores_bytes = new_bytes_buffer(K * (Py_ssize_t)sizeof(float), &scores_data);
    if (!scores_bytes) goto fail2;
    PyObject *idx_bytes = new_bytes_buffer(K * (Py_ssize_t)sizeof(int64_t), &idx_data);
    if (!idx_bytes) { Py_DECREF(scores_bytes); goto fail2; }
    float *scores = (float *)scores_data;
    int64_t *idxs = (int64_t *)idx_data;

    int saved_threads = omp_get_max_threads();
    if (num_threads > 0) omp_set_num_threads((int)num_threads);

    Py_BEGIN_ALLOW_THREADS
    if      (b == 1) scan_asym_b1_topk_parallel(
                          (const uint8_t *)codes_view.buf, (size_t)N, (size_t)d,
                          (const float *)q_view.buf, (int)K, scores, idxs);
    else if (b == 2) scan_asym_b2_topk_parallel(
                          (const uint8_t *)codes_view.buf, (size_t)N, (size_t)d,
                          (const float *)q_view.buf, (int)K, scores, idxs);
    else             scan_asym_b4_topk_parallel(
                          (const uint8_t *)codes_view.buf, (size_t)N, (size_t)d,
                          (const float *)q_view.buf, (int)K, scores, idxs);
    Py_END_ALLOW_THREADS

    if (num_threads > 0) omp_set_num_threads(saved_threads);

    PyBuffer_Release(&codes_view);
    PyBuffer_Release(&q_view);
    return Py_BuildValue("(NN)", scores_bytes, idx_bytes);

fail2:
    PyBuffer_Release(&codes_view);
    PyBuffer_Release(&q_view);
    return NULL;
}

static PyObject *py_scan_b1(PyObject *self, PyObject *args) { return do_scan(args, 1); }
static PyObject *py_scan_b2(PyObject *self, PyObject *args) { return do_scan(args, 2); }
static PyObject *py_scan_b4(PyObject *self, PyObject *args) { return do_scan(args, 4); }

// ---------- pack_hamming ----------

static PyObject *py_pack_hamming(PyObject *self, PyObject *args) {
    PyObject *y_obj;
    if (!PyArg_ParseTuple(args, "O", &y_obj)) return NULL;
    Py_buffer y_view;
    if (get_buffer(y_obj, &y_view, 'f', 0) != 0) return NULL;
    if (y_view.ndim != 2) {
        PyErr_Format(PyExc_ValueError,
                     "Y must be 2-D, got ndim=%d", y_view.ndim);
        PyBuffer_Release(&y_view);
        return NULL;
    }
    Py_ssize_t d = y_view.shape[1];
    if (simdq_check_d(d) != 0) { PyBuffer_Release(&y_view); return NULL; }
    if ((d % 64) != 0) {
        PyErr_Format(PyExc_ValueError,
                     "pack_hamming: d must be a multiple of 64; got %zd", d);
        PyBuffer_Release(&y_view); return NULL;
    }
    Py_ssize_t N = y_view.shape[0];
    Py_ssize_t words = d / 64;

    // Output: AoS layout, words uint64 per code, N codes total.
    char *out_data;
    PyObject *out = new_bytes_buffer(N * words * (Py_ssize_t)sizeof(uint64_t), &out_data);
    if (!out) { PyBuffer_Release(&y_view); return NULL; }
    uint64_t *db = (uint64_t *)out_data;
    const float *Y = (const float *)y_view.buf;

    Py_BEGIN_ALLOW_THREADS
    #pragma omp parallel for
    for (Py_ssize_t i = 0; i < N; i++) {
        const float *yi = Y + (size_t)i * d;
        for (Py_ssize_t w = 0; w < words; w++) {
            uint64_t bits = 0;
            for (int b = 0; b < 64; b++) {
                if (yi[w * 64 + b] >= 0.0f) bits |= ((uint64_t)1 << b);
            }
            db[i * words + w] = bits;
        }
    }
    Py_END_ALLOW_THREADS

    PyBuffer_Release(&y_view);
    return out;
}

// ---------- scan_hamming ----------

static PyObject *py_scan_hamming(PyObject *self, PyObject *args) {
    PyObject *codes_obj, *q_obj;
    Py_ssize_t N, D, K, num_threads;
    if (!PyArg_ParseTuple(args, "OnnOnn",
                          &codes_obj, &N, &D, &q_obj, &K, &num_threads)) return NULL;
    if (K <= 0 || K > 256) {
        PyErr_SetString(PyExc_ValueError, "K must be in [1, 256]"); return NULL;
    }
    if (simdq_check_d(D) != 0) return NULL;
    if ((D % 64) != 0) {
        PyErr_Format(PyExc_ValueError,
                     "scan_hamming: D must be a multiple of 64; got %zd", D);
        return NULL;
    }
    Py_ssize_t words = D / 64;

    Py_buffer codes_view, q_view;
    if (get_buffer(codes_obj, &codes_view, 'B', 0) != 0) return NULL;
    if (get_buffer(q_obj, &q_view, 'B', 0) != 0) {
        PyBuffer_Release(&codes_view); return NULL;
    }
    Py_ssize_t expect_codes = N * words * (Py_ssize_t)sizeof(uint64_t);
    Py_ssize_t expect_q     = words * (Py_ssize_t)sizeof(uint64_t);
    if (codes_view.shape[0] < expect_codes) {
        PyErr_Format(PyExc_ValueError,
                     "codes buffer too small: %zd < %zd", codes_view.shape[0], expect_codes);
        goto fail3;
    }
    if (q_view.shape[0] < expect_q) {
        PyErr_Format(PyExc_ValueError,
                     "q buffer too small: %zd < %zd", q_view.shape[0], expect_q);
        goto fail3;
    }

    // SoA-fill: dbT[w * N + i] = db[i * words + w].
    // We do the AoS->SoA conversion manually here since fill_soa_parallel
    // generates random data internally rather than copying from a source.
    uint64_t *dbT = (uint64_t *)aligned_alloc(64, (size_t)N * words * sizeof(uint64_t));
    if (!dbT) { PyErr_NoMemory(); goto fail3; }
    const uint64_t *db = (const uint64_t *)codes_view.buf;
    for (Py_ssize_t w = 0; w < words; w++)
        for (Py_ssize_t i = 0; i < N; i++)
            dbT[(size_t)w * (size_t)N + (size_t)i] = db[(size_t)i * (size_t)words + (size_t)w];

    char *scores_data, *idx_data;
    PyObject *scores_bytes = new_bytes_buffer(K * (Py_ssize_t)sizeof(int64_t), &scores_data);
    if (!scores_bytes) { free(dbT); goto fail3; }
    PyObject *idx_bytes = new_bytes_buffer(K * (Py_ssize_t)sizeof(int64_t), &idx_data);
    if (!idx_bytes) { Py_DECREF(scores_bytes); free(dbT); goto fail3; }
    int64_t *scores = (int64_t *)scores_data;
    int64_t *idxs   = (int64_t *)idx_data;

    int saved_threads = omp_get_max_threads();
    if (num_threads > 0) omp_set_num_threads((int)num_threads);

    Py_BEGIN_ALLOW_THREADS
    scan_hamming_topk_parallel(dbT, (size_t)N, (size_t)words,
                               (const uint64_t *)q_view.buf, (int)K,
                               scores, idxs);
    Py_END_ALLOW_THREADS

    if (num_threads > 0) omp_set_num_threads(saved_threads);
    free(dbT);

    PyBuffer_Release(&codes_view); PyBuffer_Release(&q_view);
    return Py_BuildValue("(NN)", scores_bytes, idx_bytes);

fail3:
    PyBuffer_Release(&codes_view); PyBuffer_Release(&q_view);
    return NULL;
}

// ---------- scan_hamming_soa (no per-query transpose) ----------
//
// Codes are already stored SoA (dbT[w*N + i]), so this skips the AoS->SoA
// transpose+alloc that scan_hamming does on every call. Scans the index range
// [i0, i1) (word stride is the full N); the IVF path passes one cluster's range.
static PyObject *py_scan_hamming_soa(PyObject *self, PyObject *args) {
    PyObject *codes_obj, *q_obj;
    Py_ssize_t N, D, K, num_threads, i0, i1;
    if (!PyArg_ParseTuple(args, "OnnOnnnn",
                          &codes_obj, &N, &D, &q_obj, &K, &num_threads,
                          &i0, &i1)) return NULL;
    if (K <= 0 || K > 256) {
        PyErr_SetString(PyExc_ValueError, "K must be in [1, 256]"); return NULL;
    }
    if (simdq_check_d(D) != 0) return NULL;
    if ((D % 64) != 0) {
        PyErr_Format(PyExc_ValueError,
                     "scan_hamming_soa: D must be a multiple of 64; got %zd", D);
        return NULL;
    }
    if (i0 < 0 || i1 > N || i0 > i1) {
        PyErr_Format(PyExc_ValueError,
                     "scan_hamming_soa: bad range [%zd, %zd) for N=%zd", i0, i1, N);
        return NULL;
    }
    Py_ssize_t words = D / 64;

    Py_buffer codes_view, q_view;
    if (get_buffer(codes_obj, &codes_view, 'B', 0) != 0) return NULL;
    if (get_buffer(q_obj, &q_view, 'B', 0) != 0) {
        PyBuffer_Release(&codes_view); return NULL;
    }
    Py_ssize_t expect_codes = N * words * (Py_ssize_t)sizeof(uint64_t);
    Py_ssize_t expect_q     = words * (Py_ssize_t)sizeof(uint64_t);
    if (codes_view.shape[0] < expect_codes) {
        PyErr_Format(PyExc_ValueError,
                     "codes buffer too small: %zd < %zd", codes_view.shape[0], expect_codes);
        goto failsoa;
    }
    if (q_view.shape[0] < expect_q) {
        PyErr_Format(PyExc_ValueError,
                     "q buffer too small: %zd < %zd", q_view.shape[0], expect_q);
        goto failsoa;
    }
    if (K > (i1 - i0)) K = (i1 - i0);   // can't return more than the range holds
    if (K <= 0) {                       // empty range -> empty result
        PyBuffer_Release(&codes_view); PyBuffer_Release(&q_view);
        char *e1, *e2;
        PyObject *b1 = new_bytes_buffer(0, &e1);
        PyObject *b2 = new_bytes_buffer(0, &e2);
        if (!b1 || !b2) { Py_XDECREF(b1); Py_XDECREF(b2); return NULL; }
        return Py_BuildValue("(NN)", b1, b2);
    }

    char *scores_data, *idx_data;
    PyObject *scores_bytes = new_bytes_buffer(K * (Py_ssize_t)sizeof(int64_t), &scores_data);
    if (!scores_bytes) goto failsoa;
    PyObject *idx_bytes = new_bytes_buffer(K * (Py_ssize_t)sizeof(int64_t), &idx_data);
    if (!idx_bytes) { Py_DECREF(scores_bytes); goto failsoa; }

    int saved_threads = omp_get_max_threads();
    if (num_threads > 0) omp_set_num_threads((int)num_threads);

    Py_BEGIN_ALLOW_THREADS
    scan_hamming_topk_parallel_range((const uint64_t *)codes_view.buf, (size_t)N,
                                     (size_t)words, (size_t)i0, (size_t)i1,
                                     (const uint64_t *)q_view.buf, (int)K,
                                     (int64_t *)scores_data, (int64_t *)idx_data);
    Py_END_ALLOW_THREADS

    if (num_threads > 0) omp_set_num_threads(saved_threads);

    PyBuffer_Release(&codes_view); PyBuffer_Release(&q_view);
    return Py_BuildValue("(NN)", scores_bytes, idx_bytes);

failsoa:
    PyBuffer_Release(&codes_view); PyBuffer_Release(&q_view);
    return NULL;
}

// ---------- fagin threshold algorithm ----------

static PyObject *py_fagin_search(PyObject *self, PyObject *args) {
    PyObject *y_obj, *order_obj, *vals_obj, *q_obj;
    Py_ssize_t K, batch, max_depth, num_threads, schedule = 0;
    float epsilon;
    double r2 = 1.0;   // ball radius^2 for norm-aware schedules (default unit)
    if (!PyArg_ParseTuple(args, "OOOOnnfnn|nd",
                          &y_obj, &order_obj, &vals_obj, &q_obj,
                          &K, &batch, &epsilon, &max_depth, &num_threads,
                          &schedule, &r2))
        return NULL;
    if (K < 1)     { PyErr_SetString(PyExc_ValueError, "K must be >= 1");     return NULL; }
    if (batch < 1) { PyErr_SetString(PyExc_ValueError, "batch must be >= 1"); return NULL; }
    if (schedule < FAGIN_SCHEDULE_LOCKSTEP || schedule > FAGIN_SCHEDULE_STEEPEST_NORM) {
        PyErr_SetString(PyExc_ValueError,
                        "schedule must be 0 (lockstep), 1 (steepest), "
                        "2 (lockstep_norm/GTA) or 3 (steepest_norm/GTASD)");
        return NULL;
    }

    Py_buffer y_view, order_view, vals_view, q_view;
    if (get_buffer(y_obj, &y_view, 'f', 0) != 0) return NULL;
    if (get_buffer(order_obj, &order_view, 'i', 0) != 0) {
        PyBuffer_Release(&y_view); return NULL;
    }
    if (get_buffer(vals_obj, &vals_view, 'f', 0) != 0) {
        PyBuffer_Release(&y_view); PyBuffer_Release(&order_view); return NULL;
    }
    if (get_buffer(q_obj, &q_view, 'f', 0) != 0) {
        PyBuffer_Release(&y_view); PyBuffer_Release(&order_view);
        PyBuffer_Release(&vals_view); return NULL;
    }

    PyObject *scores_bytes = NULL, *idx_bytes = NULL;

    if (y_view.ndim != 2) {
        PyErr_SetString(PyExc_ValueError, "Y must be 2-D (N, D)");
        goto failta;
    }
    {
        Py_ssize_t N = y_view.shape[0], D = y_view.shape[1];
        if (order_view.ndim != 2 || order_view.shape[0] != D
                                 || order_view.shape[1] != N) {
            PyErr_Format(PyExc_ValueError, "order must have shape (%zd, %zd)", D, N);
            goto failta;
        }
        if (vals_view.ndim != 2 || vals_view.shape[0] != D
                                || vals_view.shape[1] != N) {
            PyErr_Format(PyExc_ValueError, "vals must have shape (%zd, %zd)", D, N);
            goto failta;
        }
        if (q_view.ndim != 1 || q_view.shape[0] != D) {
            PyErr_Format(PyExc_ValueError, "q must have shape (%zd,)", D);
            goto failta;
        }

        char *scores_data, *idx_data;
        scores_bytes = new_bytes_buffer(K * (Py_ssize_t)sizeof(float), &scores_data);
        if (!scores_bytes) goto failta;
        idx_bytes = new_bytes_buffer(K * (Py_ssize_t)sizeof(int64_t), &idx_data);
        if (!idx_bytes) goto failta;

        fagin_stats_t st;
        int saved_threads = omp_get_max_threads();
        if (num_threads > 0) omp_set_num_threads((int)num_threads);

        Py_BEGIN_ALLOW_THREADS
        fagin_ta_search((const float *)y_view.buf,
                        (const int32_t *)order_view.buf,
                        (const float *)vals_view.buf,
                        (int64_t)N, (int64_t)D,
                        (const float *)q_view.buf,
                        (int)K, (int64_t)batch, epsilon, (int64_t)max_depth,
                        (int)schedule, r2,
                        (float *)scores_data, (int64_t *)idx_data, &st);
        Py_END_ALLOW_THREADS

        if (num_threads > 0) omp_set_num_threads(saved_threads);

        PyBuffer_Release(&y_view); PyBuffer_Release(&order_view);
        PyBuffer_Release(&vals_view); PyBuffer_Release(&q_view);

        PyObject *stats = Py_BuildValue(
            "{s:L,s:L,s:L,s:L,s:i}",
            "depth", (long long)st.depth,
            "sorted_accesses", (long long)st.sorted_accesses,
            "random_accesses", (long long)st.random_accesses,
            "rounds", (long long)st.rounds,
            "exhausted", st.exhausted);
        if (!stats) { Py_DECREF(scores_bytes); Py_DECREF(idx_bytes); return NULL; }
        return Py_BuildValue("(NNN)", scores_bytes, idx_bytes, stats);
    }

failta:
    Py_XDECREF(scores_bytes);
    Py_XDECREF(idx_bytes);
    PyBuffer_Release(&y_view); PyBuffer_Release(&order_view);
    PyBuffer_Release(&vals_view); PyBuffer_Release(&q_view);
    return NULL;
}

// ---------- module table ----------

static PyMethodDef SimdqMethods[] = {
    {"compute_scales", py_compute_scales, METH_VARARGS,
     "compute_scales(Y) -> scales (fp32, ||y||/sqrt(D) per row)"},
    {"pack_b1", py_pack_b1, METH_VARARGS, "pack_b1(Y, scales) -> codes (uint8 bytes)"},
    {"pack_b2", py_pack_b2, METH_VARARGS, "pack_b2(Y, scales) -> codes (uint8 bytes)"},
    {"pack_b4", py_pack_b4, METH_VARARGS, "pack_b4(Y, scales) -> codes (uint8 bytes)"},
    {"scan_b1", py_scan_b1, METH_VARARGS,
     "scan_b1(codes, N, D, q, K, num_threads) -> (scores, indices) bytes"},
    {"scan_b2", py_scan_b2, METH_VARARGS,
     "scan_b2(codes, N, D, q, K, num_threads) -> (scores, indices) bytes"},
    {"scan_b4", py_scan_b4, METH_VARARGS,
     "scan_b4(codes, N, D, q, K, num_threads) -> (scores, indices) bytes"},
    {"pack_hamming", py_pack_hamming, METH_VARARGS,
     "pack_hamming(Y) -> codes (uint8 bytes, AoS layout, D*N/8 bytes total)"},
    {"scan_hamming", py_scan_hamming, METH_VARARGS,
     "scan_hamming(codes, N, D, q, K, num_threads) -> (distances, indices) bytes; "
     "AoS codes, transposes to SoA per call. distances int64 (smaller = better)."},
    {"scan_hamming_soa", py_scan_hamming_soa, METH_VARARGS,
     "scan_hamming_soa(codes, N, D, q, K, num_threads, i0, i1) -> (distances, indices) "
     "bytes; codes already SoA (dbT[w*N+i]), no transpose; scans range [i0, i1)."},
    {"fagin_search", py_fagin_search, METH_VARARGS,
     "fagin_search(Y, order, vals, q, K, batch, epsilon, max_depth,"
     " num_threads[, schedule[, r2]])"
     " -> (scores, indices, stats); exact top-K inner product via Fagin's"
     " Threshold Algorithm. Y fp32 (N,D); order int32 (D,N) descending argsort"
     " per dim; vals fp32 (D,N) sorted values; epsilon = additive halting"
     " slack (0 = exact); max_depth 0 = unlimited (steepest: per-dim cap);"
     " schedule 0 = lockstep round-robin (default), 1 = steepest-descent"
     " (advance the dim with the largest marginal threshold drop), 2/3 = the"
     " norm-aware (GTA/GTASD) variants of 0/1 using the water-filling halting"
     " bound with ||x|| <= sqrt(r2); r2 = max squared row norm (default 1.0,"
     " i.e. unit-norm corpus). Exact for any r2 >= the true max; tightest at"
     " the true value."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef simdq_module = {
    PyModuleDef_HEAD_INIT, "_simdq_native",
    "CPython binding for simdq SIMD-quantized scan kernels.",
    -1, SimdqMethods,
};

PyMODINIT_FUNC PyInit__simdq_native(void) {
    PyObject *m = PyModule_Create(&simdq_module);
    if (!m) return NULL;
    PyObject *supported = Py_BuildValue("(iiiiii)", 192, 384, 512, 768, 1024, 1536);
    if (!supported) { Py_DECREF(m); return NULL; }
    if (PyModule_AddObject(m, "SUPPORTED_D", supported) < 0) {
        Py_DECREF(supported); Py_DECREF(m); return NULL;
    }
    return m;
}
