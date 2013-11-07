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

void update_unaries (int n, int ntags, int start, int end,
                     double *unaries, double *score, PyObject **back)
{
    double value, prob, p, tmp, *u;
    int child, parent, added = 1, ind = (start*n+end)*ntags, ip;
    PyObject *list;
    while (added) {
        added = 0;
        for (child = 0; child < ntags; ++child) {
            value = score[ind+child];
            if (value <= 0.0) {
                u = &(unaries[child*ntags]);
                for (parent = 0; parent < ntags; ++parent) {
                    ip = ind + parent;
                    prob = u[parent];
                    if (prob <= 0.0) {
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
        }
    }
}

static PyObject
*cky_decode (PyObject *self, PyObject *args)
{
    int i, n, ntags;
    PyObject *score_obj, *back_obj, *unaries_obj, *binaries_obj;
    if (!PyArg_ParseTuple(args, "iiOOOO", &n, &ntags, &score_obj, &back_obj,
                          &unaries_obj, &binaries_obj))
        return NULL;

    PyArrayObject *unaries_array = PARSE_ARRAY(unaries_obj),
                  *binaries_array = PARSE_ARRAY(binaries_obj),
                  *score_array = PARSE_ARRAY(score_obj),
                  *back_array = (PyArrayObject*) PyArray_FROM_OTF(back_obj, NPY_OBJECT, NPY_INOUT_ARRAY);
    double *unaries = PyArray_DATA(unaries_array),
           *binaries = PyArray_DATA(binaries_array),
           *score = PyArray_DATA(score_array);
    PyObject **back = PyArray_DATA(back_array);

    printf("initial unaries\n");
    for (i = 0; i < n; ++i)
        update_unaries(n, ntags, i, 0, unaries, score, back);

    PyObject *list;
    double prob, lp, rp, p, tmp, *r, *l, *b;
    int span, begin, end, split, parent, lchild, rchild, ind;
    for (span = 0; span < n; ++span) {
        printf("span = %d\n", span);
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
                                b = &(binaries[(lchild*ntags+rchild)*ntags]);
                                for (parent = 0; parent < ntags; ++parent) {
                                    prob = b[parent];
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
            update_unaries(n, ntags, begin, end, unaries, score, back);
        }
    }

    Py_DECREF(unaries_array);
    Py_DECREF(binaries_array);
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
