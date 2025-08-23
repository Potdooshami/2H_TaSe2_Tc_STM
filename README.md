[If you click here]([url](https://github.com/Potdooshami/2H_TaSe2_Tc_STM/blob/main/README().md))
# Problem definition
<img width="832" height="514" alt="image" src="https://github.com/user-attachments/assets/9b1748fa-52b8-4d91-99c8-862e9bdcac23" />

3x3 CDW exits(9-degenerate ground state possible)  
9 possible degenerate ground state construct DWN  
I want to segment images by phase-coherence 
<img width="690" height="665" alt="image" src="https://github.com/user-attachments/assets/de0bd37b-ae40-48c2-941f-39347adc1c39" />  
I want to achieve it automatically

# Read STM data as python
`.sxm` data load by `Gwyddion` and saved into `.gwy` format after few preprocessing(correction and leveling.  
python library `gwyfile` can be installed by `pip`
# Preprocessing (remove background)
Additional preprocessing can't be done with `Gwyddion`.  
here, my data contains a mechanical fluctation effect, I removed it by gaussian substraction
<img width="1794" height="691" alt="image" src="https://github.com/user-attachments/assets/b761fd9e-b5c2-4edf-a6ec-545a69874755" />

# Find lattice periodicity(FFT peak find)
It is easy to find the FFT peaks in cleaned data.
$k_1$ , $k_2$ are enough for algorithm, but considering triangular symmetry and cross-check, I'll define $k_3$( = $-k_1-k_2$) too  

<img width="303" height="300" alt="image" src="https://github.com/user-attachments/assets/f53da73d-8463-4eae-bd09-5433267f8558" />   

Below figure is real space visualization of $k_1^{CDW},k_2^{CDW},k_3^{CDW}$.

<img width="846" height="836" alt="image" src="https://github.com/user-attachments/assets/a643ac90-e10b-47b7-9542-0526ef66f823" />

# Visualize phase-amplitude(Lawler-Fujita method)
The Lawler-Fujita method is a drift-correction and distortion-removal algorithm for STM images, using local lattice fitting to achieve sub-pixel accuracy in real-space analysis.

<img width="1854" height="1258" alt="image" src="https://github.com/user-attachments/assets/accf7fa2-aaec-4832-ba66-d3a82fca1f47" />

# Remove extrinsic displcement
## Creep motion
Saptial distrotion and dirift effect can be compensated by the fact that material's intrinsic lattice should be a single phase.
<img width="813" height="890" alt="image" src="https://github.com/user-attachments/assets/105f072d-7b20-4140-8fc8-c17376c92d62" />

## Remove tip change effect
Tip change effect can be easily compensate by sigmoid fitting.
<img width="848" height="498" alt="image" src="https://github.com/user-attachments/assets/ea431414-9674-4502-97e9-e203f2e19eca" />
# Final results
<img width="1236" height="407" alt="image" src="https://github.com/user-attachments/assets/f83b95c2-fb2c-457f-8d76-d43958a7b1df" />
