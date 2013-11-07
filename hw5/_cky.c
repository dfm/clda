#include <Python.h>
#include <numpy/arrayobject.h>

struct module_state {
    PyObject *error;
};

#if PY_MAJOR_VERSION >= 3
#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))
#else
#define GETSTATE(m) (&_state)
static struct module_state _state;
#endif

#define PARSE_ARRAY(o) (PyArrayObject*) PyArray_FROM_OTF(o, NPY_DOUBLE, \
        NPY_INOUT_ARRAY)

typedef struct sparse_struct {

    int n, nnz, *inds;
    double *values;

} sparse;

sparse * sparse_init (int n, int nnz)
{
    sparse *self = malloc(sizeof(sparse));
    self->n = n;
    self->nnz = nnz;
    self->inds = malloc(nnz*sizeof(int));
    self->values = malloc(nnz*sizeof(double));
    return self;
}

void sparse_free (sparse *self)
{
    free(self->inds);
    free(self->values);
    free(self);
}

void update_unaries (int n, int ntags, int start, int end,
                     sparse **unaries, double *score, PyObject **back,
                     double theta, int prune)
{
    double value, prob, p, tmp, max_score = -INFINITY;
    int i, child, parent, added = 1, ind = (start*n+end)*ntags, ip;
    PyObject *list;
    while (added) {
        added = 0;
        for (child = 0; child < ntags; ++child) {
            value = score[ind+child];
            if (value <= 0.0) {
                for (i = 0; i < unaries[child]->nnz; ++i) {
                    parent = unaries[child]->inds[i];
                    prob = unaries[child]->values[i];
                    if (prob <= 0.0) {
                        ip = ind + parent;
                        p = prob + value;
                        tmp = score[ip];
                        if (tmp > 0 || p > tmp) {
                            added = 1;
                            score[ip] = p;
                            if (p > max_score) max_score = p;

                            list = PyList_New(1);
                            PyList_SetItem(list, 0, Py_BuildValue("iiii", start, end, child, 0));
                            Py_DECREF(back[ip]);
                            back[ip] = list;
                        }
                    }
                }
            }
        }
    }

    // Pruning.
    if (prune)
        for (parent = 0; parent < ntags; ++parent)
            if (score[ind+parent] < max_score - theta)
                score[ind+parent] = 1.0;
}

static PyObject
*cky_decode (PyObject *self, PyObject *args)
{
    double theta;
    int i, j, k, ind, n, nnz, ntags;
    PyObject *score_obj, *back_obj,
             *unary_inds_obj, *unary_values_obj,
             *binary_inds_obj, *binary_values_obj;
    if (!PyArg_ParseTuple(args, "iiOOOOOOd", &n, &ntags, &score_obj, &back_obj,
                          &unary_inds_obj, &unary_values_obj,
                          &binary_inds_obj, &binary_values_obj, &theta))
        return NULL;

    PyArrayObject *score_array = PARSE_ARRAY(score_obj),
                  *back_array = (PyArrayObject*) PyArray_FROM_OTF(back_obj, NPY_OBJECT, NPY_INOUT_ARRAY);
    double *score = PyArray_DATA(score_array);
    PyObject **back = PyArray_DATA(back_array);

    // Build unary and binary structures.
    PyObject *tmp_inds, *tmp_values, *tmp_inds2, *tmp_values2;
    sparse **unaries = malloc(ntags * sizeof(sparse*));
    for (i = 0; i < ntags; ++i) {
        tmp_inds = PyList_GetItem(unary_inds_obj, i);
        tmp_values = PyList_GetItem(unary_values_obj, i);
        nnz = PyList_Size(tmp_inds);
        unaries[i] = sparse_init (ntags, nnz);
        for (j = 0; j < nnz; ++j) {
            unaries[i]->inds[j] = (int)PyLong_AsLong(PyList_GetItem(tmp_inds, j));
            unaries[i]->values[j] = PyFloat_AS_DOUBLE(PyList_GetItem(tmp_values, j));
        }
    }

    sparse **binaries = malloc(ntags * ntags * sizeof(sparse*));
    for (i = 0; i < ntags; ++i) {
        tmp_inds = PyList_GetItem(binary_inds_obj, i);
        tmp_values = PyList_GetItem(binary_values_obj, i);
        for (k = 0; k < ntags; ++k) {
            tmp_inds2 = PyList_GetItem(tmp_inds, k);
            tmp_values2 = PyList_GetItem(tmp_values, k);

            nnz = PyList_Size(tmp_inds2);

            ind = i*ntags+k;
            binaries[ind] = sparse_init (ntags, nnz);
            for (j = 0; j < nnz; ++j) {
                binaries[ind]->inds[j] = (int)PyLong_AsLong(PyList_GetItem(tmp_inds2, j));
                binaries[ind]->values[j] = PyFloat_AS_DOUBLE(PyList_GetItem(tmp_values2, j));
            }
        }
    }

    for (i = 0; i < n; ++i)
        update_unaries(n, ntags, i, 0, unaries, score, back, theta, 0);

    PyObject *list;
    double prob, lp, rp, p, tmp, *r, *l;
    int span, begin, end, split, parent, lchild, rchild;
    for (span = 0; span < n; ++span) {
        for (begin = 0; begin < n-span-1; ++begin) {
            end = span + 1;
            ind = (begin*n+end)*ntags;
            for (split = 0; split < span+1; ++split) {
                l = &(score[(begin*n+split)*ntags]);
                r = &(score[((begin+split+1)*n+(span-split))*ntags]);
                for (lchild = 0; lchild < ntags; ++lchild) {
                    lp = l[lchild];
                    if (lp <= 0.0) {
                        for (rchild = 0; rchild < ntags; ++rchild) {
                            rp = r[rchild];
                            if (rp <= 0.0) {
                                j = lchild*ntags+rchild;
                                for (i = 0; i < binaries[j]->nnz; ++i) {
                                    parent = binaries[j]->inds[i];
                                    prob = binaries[j]->values[i];
                                    if (prob <= 0.0) {
                                        p = lp + rp + prob;
                                        tmp = score[ind+parent];
                                        if (tmp > 0 || p > tmp) {
                                            score[ind+parent] = p;

                                            list = PyList_New(2);
                                            PyList_SetItem(list, 0, Py_BuildValue("iiii", begin, split, lchild, 0));
                                            PyList_SetItem(list, 1, Py_BuildValue("iiii", begin+split+1, span-split, rchild, 0));
                                            Py_DECREF(back[ind+parent]);
                                            back[ind+parent] = list;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            update_unaries(n, ntags, begin, end, unaries, score, back, theta, 1);
        }
    }

    for (i = 0; i < ntags; ++i)
        sparse_free (unaries[i]);
    for (i = 0; i < ntags*ntags; ++i)
        sparse_free (binaries[i]);

    Py_DECREF(score_array);
    Py_DECREF(back_array);

    Py_INCREF(Py_None);
    return Py_None;
}

static PyMethodDef cky_methods[] = {
    {"decode",
     (PyCFunction) cky_decode,
     METH_VARARGS,
     ""},
    {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3

static int cky_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int cky_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_cky",
    NULL,
    sizeof(struct module_state),
    cky_methods,
    NULL,
    cky_traverse,
    cky_clear,
    NULL
};

#define INITERROR return NULL

PyObject *PyInit__cky(void)
#else
#define INITERROR return

void init_cky(void)
#endif
{
#if PY_MAJOR_VERSION >= 3
    PyObject *module = PyModule_Create(&moduledef);
#else
    PyObject *module = Py_InitModule("_cky", cky_methods);
#endif

    if (module == NULL)
        INITERROR;
    struct module_state *st = GETSTATE(module);

    st->error = PyErr_NewException("_cky.Error", NULL, NULL);
    if (st->error == NULL) {
        Py_DECREF(module);
        INITERROR;
    }

    import_array();

#if PY_MAJOR_VERSION >= 3
    return module;
#endif
}
