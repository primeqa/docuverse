// _simdq_native — CPython C-API bindings for simdq kernels.
//
// Exposes:
//   compute_scales(Y: float32[N, D]) -> float32[N]
//   pack_b1(Y: float32[N, D], scales: float32[N]) -> uint8[D * ceil(N/8)]
//   pack_b2(Y: float32[N, D], scales: float32[N]) -> uint8[D * ceil(N/4)]
//   pack_b4(Y: float32[N, D], scales: float32[N]) -> uint8[D * ceil(N/2)]
//   scan_b1(codes: uint8, N: int, q: float32[D], K: int, num_threads: int)
//                                -> (scores: float32[K], indices: int64[K])
//   scan_b2(...)
//   scan_b4(...)
//
// D is fixed at 768 (the Plan 1 / Plan 2 kernel template). All numpy arrays
// must be C-contiguous and dtype as listed; the binding raises ValueError
// otherwise. The scan functions release the GIL around the kernel call and
// honor num_threads via omp_set_num_threads.

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

#define SIMDQ_D 768

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
    if (y_view.ndim != 2 || y_view.shape[1] != SIMDQ_D) {
        PyErr_Format(PyExc_ValueError,
                     "Y must have shape (N, %d), got ndim=%d shape[1]=%zd",
                     SIMDQ_D, y_view.ndim, y_view.ndim >= 2 ? y_view.shape[1] : -1);
        PyBuffer_Release(&y_view);
        return NULL;
    }
    Py_ssize_t N = y_view.shape[0];
    char *out_data;
    PyObject *out = new_bytes_buffer(N * (Py_ssize_t)sizeof(float), &out_data);
    if (!out) { PyBuffer_Release(&y_view); return NULL; }
    float *scales = (float *)out_data;
    const float *Y = (const float *)y_view.buf;

    Py_BEGIN_ALLOW_THREADS
    const float inv_sqrt_d = 1.0f / sqrtf((float)SIMDQ_D);
    #pragma omp parallel for
    for (Py_ssize_t i = 0; i < N; i++) {
        const float *yi = Y + (size_t)i * SIMDQ_D;
        float s = 0.0f;
        for (int w = 0; w < SIMDQ_D; w++) s += yi[w] * yi[w];
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
    if (y_view.ndim != 2 || y_view.shape[1] != SIMDQ_D) {
        PyErr_Format(PyExc_ValueError,
                     "Y must have shape (N, %d)", SIMDQ_D);
        goto fail;
    }
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
    Py_ssize_t total = (Py_ssize_t)SIMDQ_D * row_bytes;

    char *out_data;
    PyObject *out = new_bytes_buffer(total, &out_data);
    if (!out) goto fail;

    Py_BEGIN_ALLOW_THREADS
    if (b == 1)      simdq_pack_b1((const float *)y_view.buf, (size_t)N, SIMDQ_D,
                                   (const float *)s_view.buf, (uint8_t *)out_data);
    else if (b == 2) simdq_pack_b2((const float *)y_view.buf, (size_t)N, SIMDQ_D,
                                   (const float *)s_view.buf, (uint8_t *)out_data);
    else             simdq_pack_b4((const float *)y_view.buf, (size_t)N, SIMDQ_D,
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
    Py_ssize_t N, K, num_threads;
    if (!PyArg_ParseTuple(args, "OnOnn",
                          &codes_obj, &N, &q_obj, &K, &num_threads)) return NULL;
    if (K <= 0 || K > 256) {
        PyErr_SetString(PyExc_ValueError, "K must be in [1, 256]");
        return NULL;
    }
    Py_buffer codes_view, q_view;
    if (get_buffer(codes_obj, &codes_view, 'B', 0) != 0) return NULL;
    if (get_buffer(q_obj, &q_view, 'f', 0) != 0) {
        PyBuffer_Release(&codes_view); return NULL;
    }
    if (q_view.ndim != 1 || q_view.shape[0] != SIMDQ_D) {
        PyErr_Format(PyExc_ValueError, "q must have shape (%d,)", SIMDQ_D);
        PyBuffer_Release(&codes_view); PyBuffer_Release(&q_view); return NULL;
    }
    Py_ssize_t row_bytes;
    if      (b == 1) row_bytes = (N + 7) / 8;
    else if (b == 2) row_bytes = (N + 3) / 4;
    else             row_bytes = (N + 1) / 2;
    Py_ssize_t expect = (Py_ssize_t)SIMDQ_D * row_bytes;
    if (codes_view.shape[0] < expect) {
        PyErr_Format(PyExc_ValueError,
                     "codes buffer too small: got %zd bytes, need >= %zd "
                     "for N=%zd b=%d D=%d",
                     codes_view.shape[0], expect, N, b, SIMDQ_D);
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
    if      (b == 1) scan_asym_b1_d768_topk_parallel(
                          (const uint8_t *)codes_view.buf, (size_t)N,
                          (const float *)q_view.buf, (int)K, scores, idxs);
    else if (b == 2) scan_asym_b2_d768_topk_parallel(
                          (const uint8_t *)codes_view.buf, (size_t)N,
                          (const float *)q_view.buf, (int)K, scores, idxs);
    else             scan_asym_b4_d768_topk_parallel(
                          (const uint8_t *)codes_view.buf, (size_t)N,
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

// ---------- module table ----------

static PyMethodDef SimdqMethods[] = {
    {"compute_scales", py_compute_scales, METH_VARARGS,
     "compute_scales(Y) -> scales (fp32, ||y||/sqrt(D) per row)"},
    {"pack_b1", py_pack_b1, METH_VARARGS, "pack_b1(Y, scales) -> codes (uint8 bytes)"},
    {"pack_b2", py_pack_b2, METH_VARARGS, "pack_b2(Y, scales) -> codes (uint8 bytes)"},
    {"pack_b4", py_pack_b4, METH_VARARGS, "pack_b4(Y, scales) -> codes (uint8 bytes)"},
    {"scan_b1", py_scan_b1, METH_VARARGS,
     "scan_b1(codes, N, q, K, num_threads) -> (scores, indices) bytes"},
    {"scan_b2", py_scan_b2, METH_VARARGS,
     "scan_b2(codes, N, q, K, num_threads) -> (scores, indices) bytes"},
    {"scan_b4", py_scan_b4, METH_VARARGS,
     "scan_b4(codes, N, q, K, num_threads) -> (scores, indices) bytes"},
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
    if (PyModule_AddIntConstant(m, "D", SIMDQ_D) < 0) { Py_DECREF(m); return NULL; }
    return m;
}
