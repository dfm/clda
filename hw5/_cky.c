#include <Python.h>
#include <structmember.h>
#include <numpy/arrayobject.h>

#define PARSE_ARRAY(o) (PyArrayObject*) PyArray_FROM_OTF(o, NPY_DOUBLE, \
        NPY_INOUT_ARRAY)

typedef struct unary_struct {

    int n, *parents, *children;
    double *value;

} unary_rule;

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

typedef struct {
    PyObject_HEAD
    int ntags;
    sparse **unaries;
    sparse **binaries;
} _cky;

static void _cky_dealloc(_cky *self)
{
    int i;
    for (i = 0; i < self->ntags; ++i)
        sparse_free (self->unaries[i]);
    for (i = 0; i < self->ntags*self->ntags; ++i)
        sparse_free (self->binaries[i]);

    self->ob_type->tp_free((PyObject*)self);
}

static PyObject *_cky_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    _cky *self;
    self = (_cky*)type->tp_alloc(type, 0);
    self->unaries = NULL;
    self->binaries = NULL;
    return (PyObject*)self;
}

static int _cky_init(_cky *self, PyObject *args, PyObject *kwds)
{
    int i, j, k, ind, nnz, ntags;
    PyObject *unary_inds_obj, *unary_values_obj,
             *binary_inds_obj, *binary_values_obj;
    if (!PyArg_ParseTuple(args, "iOOOO", &ntags,
                          &unary_inds_obj, &unary_values_obj,
                          &binary_inds_obj, &binary_values_obj))
        return -1;

    self->ntags = ntags;

    // Build unary and binary structures.
    PyObject *tmp_inds, *tmp_values, *tmp_inds2, *tmp_values2;
    self->unaries = malloc(ntags * sizeof(sparse*));
    for (i = 0; i < ntags; ++i) {
        tmp_inds = PyList_GetItem(unary_inds_obj, i);
        tmp_values = PyList_GetItem(unary_values_obj, i);
        nnz = PyList_Size(tmp_inds);
        self->unaries[i] = sparse_init (ntags, nnz);
        for (j = 0; j < nnz; ++j) {
            self->unaries[i]->inds[j] = (int)PyLong_AsLong(PyList_GetItem(tmp_inds, j));
            self->unaries[i]->values[j] = PyFloat_AS_DOUBLE(PyList_GetItem(tmp_values, j));
        }
    }

    self->binaries = malloc(ntags * ntags * sizeof(sparse*));
    for (i = 0; i < ntags; ++i) {
        tmp_inds = PyList_GetItem(binary_inds_obj, i);
        tmp_values = PyList_GetItem(binary_values_obj, i);
        for (k = 0; k < ntags; ++k) {
            tmp_inds2 = PyList_GetItem(tmp_inds, k);
            tmp_values2 = PyList_GetItem(tmp_values, k);

            nnz = PyList_Size(tmp_inds2);

            ind = i*ntags+k;
            self->binaries[ind] = sparse_init (ntags, nnz);
            for (j = 0; j < nnz; ++j) {
                self->binaries[ind]->inds[j] = (int)PyLong_AsLong(PyList_GetItem(tmp_inds2, j));
                self->binaries[ind]->values[j] = PyFloat_AS_DOUBLE(PyList_GetItem(tmp_values2, j));
            }
        }
    }

    return 0;
}

static PyMemberDef _cky_members[] = {{NULL}};

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
*cky_decode (_cky *self, PyObject *args)
{
    double theta;
    int i, j, ind, n, ntags = self->ntags;
    PyObject *score_obj, *back_obj;
    if (!PyArg_ParseTuple(args, "iOOd", &n, &score_obj, &back_obj, &theta))
        return NULL;

    PyArrayObject *score_array = PARSE_ARRAY(score_obj),
                  *back_array = (PyArrayObject*) PyArray_FROM_OTF(back_obj, NPY_OBJECT, NPY_INOUT_ARRAY);
    double *score = PyArray_DATA(score_array);
    PyObject **back = PyArray_DATA(back_array);

    for (i = 0; i < n; ++i)
        update_unaries(n, ntags, i, 0, self->unaries, score, back, theta, 0);

    PyObject *list;
    double prob, lp, rp, p, tmp, *r, *l;
    int span, begin, end, split, parent, lchild, rchild;
    sparse **binaries = self->binaries;
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
            update_unaries(n, ntags, begin, end, self->unaries, score, back, theta, 1);
        }
    }

    Py_DECREF(score_array);
    Py_DECREF(back_array);

    Py_INCREF(Py_None);
    return Py_None;
}

static PyMethodDef _cky_methods[] = {
    {"decode",
     (PyCFunction) cky_decode,
     METH_VARARGS,
     ""},
    {NULL, NULL, 0, NULL}
};

static PyTypeObject _cky_type = {
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
    "_cky._cky",         /*tp_name*/
    sizeof(_cky),           /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)_cky_dealloc, /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    0,                         /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
    "",                   /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    _cky_methods,               /* tp_methods */
    _cky_members,               /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)_cky_init,        /* tp_init */
    0,                         /* tp_alloc */
    _cky_new,                   /* tp_new */
};

static char module_doc[] = "";
static PyMethodDef module_methods[] = {{NULL}};
void init_cky(void)
{
    PyObject *m;

    if (PyType_Ready(&_cky_type) < 0)
        return;

    m = Py_InitModule3("_cky", module_methods, module_doc);
    if (m == NULL)
        return;

    Py_INCREF(&_cky_type);
    PyModule_AddObject(m, "_cky", (PyObject *)&_cky_type);

    import_array();
}
