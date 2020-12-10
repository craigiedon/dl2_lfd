# Elaborating on Learned Demonstrations with Temporal Logic Specifications

The larger (somewhat messier) supporting code-base for experiments run in the RSS-2020 paper: [Elaborating on Learned Demonstrations with Temporal Logic Specifications](https://roboticsconference.org/2020/program/papers/4.html)

Note: For just the specific code relating to differentiable metrics for Linear Temporal Logic, see the submodule [LTL-Diff](https://github.com/craigiedon/ltl_diff)

## Setup

This project contains the submodule "LTL_Diff", which constains the language for converting temporal logic statements into differentiable costs. To ensure this submodule is downloaded and updated correctly, make sure you run ```git clone``` with the following flag:

    git clone --recurse-submodules

Or, if you forget to do this, run the following two commands after cloning the repository:

    git submodule init
    git submodule update
