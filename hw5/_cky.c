#include <Python.h>
#include <structmember.h>
#include <numpy/arrayobject.h>

#define PARSE_ARRAY(o) (PyArrayObject*) PyArray_FROM_OTF(o, NPY_DOUBLE, \
        NPY_INOUT_ARRAY)

typedef struct unary_struct {

    int parent, child;
    double value;

} unary_rule;

typedef struct binary_struct {

    int parent, lchild, rchild;
    double value;

} binary_rule;

unary_rule * unary_init (int parent, int child, double value)
{
    unary_rule *self = malloc(sizeof(unary_rule));
    self->parent = parent;
    self->child = child;
    self->value = value;
    return self;
}

binary_rule * binary_init (int parent, int lchild, int rchild, double value)
{
    binary_rule *self = malloc(sizeof(binary_rule));
    self->parent = parent;
    self->lchild = lchild;
    self->rchild = rchild;
    self->value = value;
    return self;
}

typedef struct {
    PyObject_HEAD
    int ntags, nbinaries, nunaries;
    unary_rule **unaries;
    binary_rule **binaries;
} _cky;

static void _cky_dealloc(_cky *self)
{
    int i;
    for (i = 0; i < self->nunaries; ++i)
        free (self->unaries[i]);
    for (i = 0; i < self->nbinaries; ++i)
        free (self->binaries[i]);

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
    int i, ntags;
    PyObject *unaries_obj, *binaries_obj;
    if (!PyArg_ParseTuple(args, "iOO", &ntags, &unaries_obj, &binaries_obj))
        return -1;

    self->ntags = ntags;

    self->nunaries = PyList_Size(unaries_obj);
    self->unaries = malloc(self->nunaries * sizeof(unary_rule*));
    self->nbinaries = PyList_Size(binaries_obj);
    self->binaries = malloc(self->nbinaries * sizeof(binary_rule*));

    PyObject *el;
    for (i = 0; i < self->nunaries; ++i) {
        el = PyList_GetItem(unaries_obj, i);
        self->unaries[i] = unary_init(
            (int)PyLong_AsLong(PyList_GetItem(el, 0)),
            (int)PyLong_AsLong(PyList_GetItem(el, 1)),
            PyFloat_AS_DOUBLE(PyList_GetItem(el, 2))
        );
    }

    for (i = 0; i < self->nbinaries; ++i) {
        el = PyList_GetItem(binaries_obj, i);
        self->binaries[i] = binary_init(
            (int)PyLong_AsLong(PyList_GetItem(el, 0)),
            (int)PyLong_AsLong(PyList_GetItem(el, 1)),
            (int)PyLong_AsLong(PyList_GetItem(el, 2)),
            PyFloat_AS_DOUBLE(PyList_GetItem(el, 3))
        );
    }

    return 0;
}

static PyMemberDef _cky_members[] = {{NULL}};

void update_unaries (int n, int ntags, int start, int end, int nunaries,
                     unary_rule **unaries, double *score, PyObject **back,
                     double theta, int prune)
{
    double value, prob, p, tmp, max_score = -INFINITY;
    int i, child, parent, added = 1, ind = (start*n+end)*ntags, ip;
    PyObject *list;
    while (added) {
        added = 0;

        for (i = 0; i < nunaries; ++i) {
            parent = unaries[i]->parent;
            child = unaries[i]->child;
            value = score[ind+child];
            if (value <= 0.0) {
                prob = unaries[i]->value;
                ip = ind + parent;
                p = prob + value;
                tmp = score[ip];
                if (tmp > 0 || p > tmp) {
                    added = 1;
                    score[ip] = p;
                    list = PyList_New(1);
                    PyList_SetItem(list, 0, Py_BuildValue("iiii", start, end, child, 0));
                    Py_DECREF(back[ip]);
                    back[ip] = list;
                }
            }
        }
    }

    // Pruning.
    if (prune) {
        for (parent = 0; parent < ntags; ++parent) {
            p = score[ind+parent];
            if (p > max_score) max_score = p;
        }

        for (parent = 0; parent < ntags; ++parent)
            if (score[ind+parent] < max_score - theta)
                score[ind+parent] = 1.0;
    }
}

static PyObject
*cky_decode (_cky *self, PyObject *args)
{
    double theta;
    int i, ind, n, ntags = self->ntags;
    PyObject *score_obj, *back_obj;
    if (!PyArg_ParseTuple(args, "iOOd", &n, &score_obj, &back_obj, &theta))
        return NULL;

    PyArrayObject *score_array = PARSE_ARRAY(score_obj),
                  *back_array = (PyArrayObject*) PyArray_FROM_OTF(back_obj, NPY_OBJECT, NPY_INOUT_ARRAY);
    double *score = PyArray_DATA(score_array);
    PyObject **back = PyArray_DATA(back_array);

    for (i = 0; i < n; ++i)
        update_unaries(n, ntags, i, 0, self->nunaries, self->unaries, score, back, theta, 0);

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
                for (i = 0; i < self->nbinaries; ++i) {
                    parent = self->binaries[i]->parent;
                    lchild = self->binaries[i]->lchild;
                    rchild = self->binaries[i]->rchild;
                    lp = l[lchild];
                    rp = r[rchild];
                    if (lp <= 0.0 && rp <= 0.0) {
                        prob = self->binaries[i]->value;
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
            update_unaries(n, ntags, begin, end, self->nunaries, self->unaries, score, back, theta, 1);
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
