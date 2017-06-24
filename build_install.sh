#!/bin/bash
source activate testenv
py.test --verbose --cov
