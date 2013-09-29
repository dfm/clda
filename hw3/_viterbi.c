#include <Python.h>

struct module_state {
    PyObject *error;
};

#if PY_MAJOR_VERSION >= 3
#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))
#else
#define GETSTATE(m) (&_state)
static struct module_state _state;
#endif

typedef struct state {

    int active, ptag, tag, ind;
    PyObject *word;
    double delta;
    void *psi;

} State;

State *init_state (int ntags, int ind, int ptag, int tag, PyObject *word)
{
    State *self = malloc(sizeof(State));
    self->active = 0;
    self->psi = NULL;
    self->ptag = ptag;
    self->tag = tag;
    self->ind = ind;
    self->word = word;
    return self;
}

void free_state (State *self)
{
    free(self);
}

typedef struct trellis {

    int nwords, ntags;
    State **states;

} Trellis;

State *get_state (Trellis *trellis, int ind, int ppt, int pt)
{
    return trellis->states[(ind*trellis->ntags + ppt) * trellis->ntags + pt];
}

Trellis *init_trellis (int nwords, PyObject **words, int ntags)
{
    Trellis *self = malloc(sizeof(Trellis));
    self->nwords = nwords;
    self->ntags = ntags;
    self->states = malloc(nwords * ntags * ntags * sizeof(State*));

    // Initialize the memory.
    int i, j, k;
    for (i = 0; i < nwords; ++i)
        for (j = 0; j < ntags; ++j)
            for (k = 0; k < ntags; ++k)
                self->states[(i*ntags+j)*ntags+k] =
                    init_state (ntags, i, j, k, words[i]);

    // Set up the initial state.
    State *initial = get_state(self, 0, 0, 0);
    initial->delta = 0.0;
    initial->active = 1;

    return self;
}

void free_trellis (Trellis *self)
{
    int i;
    for (i = 0; i < self->nwords * self->ntags * self->ntags; ++i)
        free_state(self->states[i]);
    free(self->states);
    free(self);
}

void evaluate_state (Trellis *trellis, State *state, PyObject *scorer)
{
    int k, ntags = trellis->ntags;
    PyObject *scores = PyObject_CallFunction(scorer, "iiO",
                                             state->ptag, state->tag,
                                             state->word);

    if (scores == NULL) {
        Py_XDECREF(scores);
        return;
    }

    // Loop over the computed scores and update the deltas.
    for (k = 0; k < ntags; ++k) {
        PyObject *score_obj = PyList_GetItem(scores, k);
        if (score_obj == NULL) {
            Py_DECREF(scores);
            return;
        }

        if (score_obj != Py_None) {
            double score = PyFloat_AsDouble(score_obj) + state->delta;
            State *next_state = get_state(trellis, state->ind+1,
                                          state->tag, k);

            if (!next_state->active || score > next_state->delta) {
                next_state->active = 1;
                next_state->delta = score;
                next_state->psi = state;
            }
        }
    }

    Py_DECREF(scores);
}

static PyObject
*viterbi_viterbi (PyObject *self, PyObject *args)
{
    int i, j, ntags;
    PyObject *wordlist, *scorer;
    if (!PyArg_ParseTuple(args, "iOO", &ntags, &wordlist, &scorer))
        return NULL;

    // Dimensions.
    int word, nwords = PyList_Size(wordlist);

    // Extract the pointers to the words.
    PyObject **words = malloc(nwords * sizeof(PyObject*));
    for (i = 0; i < nwords; ++i)
        words[i] = PyList_GetItem(wordlist, i);

    Trellis *trellis = init_trellis (nwords, words, ntags);

    // Loop over words and every possible state combination.
    for (word = 0; word < nwords-1; ++word)
        for (i = 0; i < ntags; ++i)
            for (j = 0; j < ntags; ++j) {
                // Get the current state and if it's active compute the scores.
                State *state = get_state(trellis, word, i, j);
                if (state->active) evaluate_state (trellis, state, scorer);
            }

    // Find the final state.
    State *final_state;
    int cont = 1;
    for (i = 0; i < ntags && cont; ++i) {
        for (j = 0; j < ntags && cont; ++j) {
            final_state = get_state(trellis, nwords-1, i, j);
            if (final_state->active) cont = 0;
        }
    }

    PyObject *result = PyList_New(nwords - 1);
    if (result == NULL) {
        free_trellis(trellis);
        free(words);
        return NULL;
    }

    while (final_state->psi != NULL) {
        PyList_SetItem(result, final_state->ind - 1,
                       PyLong_FromLong(final_state->ptag));
        final_state = final_state->psi;
    }

    free_trellis(trellis);
    free(words);

    return result;
}

static PyMethodDef viterbi_methods[] = {
    {"viterbi",
     (PyCFunction) viterbi_viterbi,
     METH_VARARGS,
     ""},
    {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3

static int viterbi_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int viterbi_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_viterbi",
    NULL,
    sizeof(struct module_state),
    viterbi_methods,
    NULL,
    viterbi_traverse,
    viterbi_clear,
    NULL
};

#define INITERROR return NULL

PyObject *PyInit__viterbi(void)
#else
#define INITERROR return

void init_viterbi(void)
#endif
{
#if PY_MAJOR_VERSION >= 3
    PyObject *module = PyModule_Create(&moduledef);
#else
    PyObject *module = Py_InitModule("_viterbi", viterbi_methods);
#endif

    if (module == NULL)
        INITERROR;
    struct module_state *st = GETSTATE(module);

    st->error = PyErr_NewException("_viterbi.Error", NULL, NULL);
    if (st->error == NULL) {
        Py_DECREF(module);
        INITERROR;
    }

#if PY_MAJOR_VERSION >= 3
    return module;
#endif
}
