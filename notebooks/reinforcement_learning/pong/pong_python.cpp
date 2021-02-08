#define PY_SSIZE_T_CLEAN
#include "game.hpp"
#include <Python.h>


////////////////////////////////////////////////////////////////////////////////
PyMethodDef method_table[] = {
  {NULL, NULL, 0, NULL}
};

PyModuleDef py_module = {
  PyModuleDef_HEAD_INIT,
  "pong",
  "Module docstring",
  -1,
  method_table,
  NULL, // Optional slot definitions
  NULL, // Optional traversal function
  NULL, // Optional clear function
  NULL  // Optional module deallocation function
};

PyMODINIT_FUNC
PyInit_pong(void) {
  return PyModule_Create(&py_module);
}
