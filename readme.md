# nonlinear_solid

A research-oriented solid mechanics codebase for **finite-strain constitutive modeling** and **data-driven constitutive learning**.

This repository combines two complementary directions:

## 1. Classical constitutive modeling (`fvm_isotropic_hardening/`)
- finite-strain elastoplasticity  
- isotropic hardening  
- modular material-state updates  
- reusable stress / tangent interfaces for numerical solvers  

## 2. Constitutive Artificial Neural Networks (CANN) (`CANN/`)
- data-driven constitutive law learning  
- invariant-based representation of strain and stress  
- neural-network-based strain-energy or stress surrogate models  
- extensible architecture for anisotropy and feature-aware learning  

---

The overall goal is to build a **clean, research-grade, and extensible framework** for nonlinear solid mechanics, suitable for:

- reproducing literature results  
- testing new constitutive hypotheses  
- developing hybrid physics–data-driven models  
- preparing PhD-level research work  

---

# Research Motivation

Constitutive modeling is one of the central problems in computational solid mechanics.

Classical analytical models are:
- physically interpretable  
- computationally efficient  
- but difficult to calibrate for complex materials  

Data-driven constitutive models are:
- flexible and expressive  
- capable of learning nonlinear behavior from data  
- but require physical constraints and careful design  

---

This repository is built around a hybrid philosophy:

- Use **physics-based constitutive laws** where reliable  
- Use **CANN-based learning** where constitutive structure is unknown  
- Maintain a **shared engineering interface** between both  

This is particularly relevant for:

- data-driven materials design  
- metamaterials modeling  
- inverse constitutive identification  
- physics–machine learning hybrid workflows  

---

# Repository Structure

```text
nonlinear_solid/
├── fvm_isotropic_hardening/
├── CANN/
├── examples/
├── scripts/
├── configs/
├── data/
├── tests/
├── docs/
├── requirements.txt
├── README.md
└── LICENSE
````

---

# Main Components

## fvm_isotropic_hardening

This module implements a **finite-strain elastoplastic constitutive model with isotropic hardening**.

### Responsibilities

* Kinematics and tensor operations
* Internal variable evolution
* Yield function formulation
* Return mapping algorithm
* Hardening law implementation
* Consistent tangent computation
* Material parameter handling
* Interfaces for FEM / numerical solvers

👉 This is the physics-based baseline of the repository.

---

## CANN

This module implements a **Constitutive Artificial Neural Network (CANN)**.

It learns constitutive behavior from data while preserving mechanics structure.

### Responsibilities

* Stress–strain dataset preprocessing
* Invariant-based feature construction
* Neural constitutive model
* Training / validation pipelines
* Stress / energy prediction
* Optional thermodynamic constraints
* Visualization and diagnostics

👉 Complements classical constitutive modeling.

---

# Theoretical Background

## Classical solid mechanics

* Deformation gradient **F**
* Right Cauchy-Green tensor **C**
* Second Piola-Kirchhoff stress **S**
* Internal variables
* Consistent tangent operator

---

## CANN-based learning

* Mechanics-informed features
* Reduced-dimensional constitutive learning
* Data-driven surrogate modeling
* Interpretable neural architecture
* Extendable to complex materials

---

# Design Principles

* Separation of concerns
* Reusable interfaces
* Config-driven experiments
* Reproducibility
* Extensibility
* Readability

---

# Typical Workflow

## Classical constitutive model

1. Define material parameters
2. Compute strain measures
3. Update stress and internal variables
4. Compute tangent
5. Use in FEM solver

---

## CANN workflow

1. Prepare dataset
2. Build invariant features
3. Train neural constitutive model
4. Validate predictions
5. Compare with classical model
6. Deploy in simulation

---

# Installation

## Create environment

```bash
python -m venv .venv
source .venv/bin/activate
```

## Install dependencies

```bash
pip install -r requirements.txt
```

---


---

# Repository Content

## Classical module

* elastoplastic constitutive law
* isotropic hardening
* stress update algorithm

## CANN module

* neural constitutive learning
* invariant-based architecture
* data-driven stress prediction

## Utilities

* preprocessing
* visualization
* experiments

---

# Research Value

This repository is a **research platform**, not just code.

It enables exploration of:

* nonlinear constitutive law design
* data-driven surrogate modeling
* physics–ML hybrid constitutive frameworks
* integration into numerical solvers

---

# Future Extensions

* hyperelastic + plastic coupling
* thermodynamic consistency
* energy-based neural networks
* FEM coupling (FEniCS / Abaqus)
* inverse material design
* uncertainty quantification
* metamaterial modeling

---


# License

This project follows the repository license (if available).

