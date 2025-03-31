
# Structural Health Monitoring (SHM) Using Advanced Deep Learning Techniques

This repository presents a comprehensive exploration of Structural Health Monitoring (SHM) methodologies through state-of-the-art deep learning methods and rigorous numerical modeling. Originating from extensive research, the code and analyses presented here correspond to Chapters 3, 4, and 5 of my thesis, covering a wide range of structural types—from simple framed structures and bridge trusses to experimental validation of numerical integrity.

## Quick Links

*   [Introduction](#introduction)
*   [Case Studies](#case-studies)
    *   [Chapter 3: Simple Framed Structure](#chapter-3-simple-framed-structure)
    *   [Chapter 4: Bridge Truss Analysis](#chapter-4-bridge-truss-analysis)
    *   [Chapter 5: Experimental Validation](#chapter-5-experimental-validation)
*   [Key Results](#key-results)
*   [Conclusion & Future Directions](#conclusion--future-directions)

## Introduction

Structural Health Monitoring (`SHM`) is crucial for ensuring the reliability and longevity of civil engineering structures. This repository employs state-of-the-art deep learning and numerical modeling techniques to detect, quantify, and analyze structural damage. The project bridges advanced computational methods with practical civil engineering applications, offering robust solutions suitable for both academic research and industrial implementation.

## Case Studies

### Chapter 3: Simple Framed Structure

*   **Linear Dynamic Analysis:** Implemented numerical methods to solve the equations of motion for multi-degree-of-freedom (`MDOF`) framed structures, simplifying to single-degree-of-freedom (`SDOF`) for efficient analysis.
*   **Model Development & Optimization:**
    *   Transitioned computational models from Maple to Python, significantly enhancing performance through `NumPy`, `SymPy`, and multi-threading (achieving **~60x speed improvement**).
    *   Constructed extensive datasets (**~250k combinations**) representing structural conditions including incremental damage in Young's Modulus (`E`) and cross-sectional area (`A`).
*   **Deep Learning Application:** Leveraged and significantly improved upon a custom `TabTransformer` architecture by integrating advanced mechanisms such as Residual Networks, Sparse Attention, Mixture-of-Experts (`MoE`), and Squeeze-and-Excite Networks (`SENet`).
*   **Model Performance:** Achieved a final `MAPE` of **4.4%** for individual predictions, enhanced further through stacking ensemble methods down to **1.17%**.

### Chapter 4: Bridge Truss Analysis

*   **Bridge Structural Modeling:**
    *   Detailed exploration of bridge loads per Eurocode standards (`EN 1991-2`), focusing on Warren, Pratt, and Howe trusses.
    *   Geometric design optimized within Eurocode recommended constraints, ensuring realistic force distribution and computational feasibility.
*   **Dataset Creation:**
    *   Generated structured datasets including geometric, mechanical, and modal parameters, optimized for deep learning inputs.
*   **Deep Learning Approach:**
    *   Employed advanced data preprocessing and hyperparameter tuning, significantly improving baseline performance from **23.36% `MAPE`** to approximately **1.94%** through innovative custom loss functions.
*   **Predictive Robustness:** Validated predictive performance on extensive unseen datasets, maintaining accuracy and demonstrating resilience to varying structural damage scenarios.

### Chapter 5: Experimental Validation

*   **Experimental Setup:**
    *   Conducted a thorough comparison of theoretical predictions against experimental data using `LVDTs` and photogrammetry for a simply supported steel beam.
    *   Evaluated elastic deflections under incremental loading to validate numerical modeling accuracy.
*   **Validation Results:**
    *   Verified the numerical integrity and accuracy of displacement predictions (within **~3% error** for typical loading conditions).
    *   Identified limitations in photogrammetric measurements, highlighting precision trade-offs in practical monitoring applications.

## Key Results

| Model Type                  | Structure        | MAPE (%)   | MdAPE (%)  | Notes
| :-------------------------- | :--------------- | :--------- | :--------- |  :------------------------------- |
| Baseline Transformer        | Framed Structure | 24-30%     | -          | Initial unoptimized model        |
| Optimized Transformer       | Framed Structure | **4.4%**   | **1.93%**  | Advanced architecture & tuning   |
| Stacking Ensemble           | Framed Structure | **1.17%**  | **1.12%**  | Best accuracy achieved           |
| Baseline Transformer        | Bridge Truss     | **23.36%** | **13.35%** | Initial setup, suboptimal        |
| Custom Loss & Optimization | Bridge Truss     | **1.94%**  | **1.07%**  | Optimal results with adaptive loss combinations |

*(MAPE: Mean Absolute Percentage Error, MdAPE: Median Absolute Percentage Error)*

## Conclusion & Future Directions

This repository underscores the potential of combining **high-fidelity numerical modeling** with **cutting-edge deep learning** to enhance SHM tasks across civil engineering structures. Key contributions include:

-   **Accelerated Computation**: Achieving large-scale dataset generation and real-time predictive capabilities.
    
-   **High-Accuracy Damage Detection**: Demonstrated robust performance under diverse damage scenarios, validated by experimental data.
    
-   **Ensemble Methods**: Showcased the value of combining multiple models to improve predictive reliability.
    

**Ongoing and future research** may explore:

-   Nonlinear and plastic deformation modeling (enabling more realistic damage progression analysis).
    
-   Transfer learning for rapid adaptation across multiple structures.
    
-   Real-time SHM pipelines with sensor streaming for industrial-scale deployment.

**Future research directions**:

*   Extension to nonlinear structural behaviors.
*   Integration of transfer learning methods to improve adaptability across various structural types.
*   Real-time deployment considerations for industrial-scale `SHM`.

## Acknowledgments

Grateful acknowledgment is extended to:

-   **Google Colab**: for computational resources critical to generating the large amount of datasets and for training the deep learning models.
    
-   **Kaggle**: for the primary training of Bridge Truss models and supplementary GPU support.
    

Please cite this repository or contact the authors for inquiries regarding extended data access, collaboration, or related publications.