
# Structural Health Monitoring Using Deep Learning Regression

This repository delves into the critical domain of  **Structural Health Monitoring (SHM)**, aimed at quantifying structural damage through the integration of deep learning and regression techniques. The coding aspects of my thesis are presented here.

## Quick Links

- [Introduction](#introduction)
- [Results](#results)
- [Conclusion](#conclusion)

## Introduction
1.  **2D Frame Construction**: Initially, I developed a 2D frame—an elemental representation of structural components. Initially, I implemented the code in  **Maple**, leveraging its symbolic capabilities. However, due to optimization limitations, I transitioned to  **Python**. This transition allowed for the utilization of Python’s robust optimization libraries, significantly enhancing computational efficiency.
    
2.  **Optimization Challenges**: The optimization phase presented intriguing challenges, particularly concerning computational time. Initially, each run demanded approximately **4 minutes**, rendering it impractical for large-scale analyses. My objective was clear—to enhance efficiency. Through meticulous optimization efforts, I achieved a remarkable  **100-fold speedup**. Nonetheless, further improvements, such as the integration of **Cython** or other low-level languages, remain on the horizon. It is noteworthy that only specific portions strictly involving numerical computations have been successfully reworked in Cython, as previously tested.
    
3.  **Selecting the Right Deep Learning Model**: With foundational work laid, I delved into the realm of deep learning. The selection of an appropriate model for structured data prediction was pivotal. After meticulous evaluation, I augmented an existing architecture, tailoring it to our specific SHM context. The architecture chosen drew inspiration from MLP and transformers. However, it became evident that the chosen architecture lacked the desired complexity. Consequently, I iteratively refined the model, incorporating insights from other prevalent architectures to achieve state-of-the-art complexity and optimal performance.
    
4.  **Model Inputs and Domain Knowledge**: Selecting model inputs required a blend of domain expertise and intuition. I steered clear of blind trial and error, drawing upon my understanding of structural mechanics. The identification of pertinent features proved instrumental in optimizing the model’s performance.
    
5.  **Domain Knowledge in Machine/Deep Learning**: Profound comprehension of machine learning and deep learning intricacies was paramount. I grappled with questions like:
    
    -   How should I order the data?
    -   What normalization techniques are suitable?
    -   Should I introduce categorical variables?
    -   Is a deeper/wider model worth it? (speed/accuracy trade-off)
    
    These decisions significantly influenced the model’s accuracy and robustness.

## Results

The journey culminated in impressive results. Initially, our predictions deviated by  **23%**  from ground truth. However, through architectural refinements, we slashed this to approximately  **14%**. Further fine-tuning, including optimizing loss functions and extending training, yielded a remarkable final error rate of just  **4.4%**.

Additionally, training time averaged around 15 minutes for 15k epochs on a T4 GPU, graciously provided by Google.

Finally, Empirical testing regarding the depth and width of the network was conducted, revealing minimal gains in accuracy, if any.

## Conclusion

Our exploration of deep learning models for SHM underscores their viability. With an error rate of 4.4% (based on  **80k combinations**), we’ve demonstrated their potential. Even when considering the full dataset (based on  **200k combinations**), the error remained commendably low at  **6.6%**. 

Furthermore, it is crucial to highlight the substantial enhancement in performance in the final findings. By leveraging common techniques such as **Ensemble**, led to astonishingly low errors—around  **1.2%**—across both scenarios.

In conclusion, these results bridge the gap between theory and application, demonstrating promising prospects for integrating such methodologies into more reliable and cost-effective monitoring practices.
