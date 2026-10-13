# Chemistry ML Dissertation and UV-Vis.ipynb
This project was designed to predict absorption and emission properties of prospective L.E.D. chemicals, primarily using a random forest model to learn
these spectra directly from molecular structures alone, though with other modules to refine these estimates using minimal physics calculations (specifically
the sTDA method for excited energy calculation). The primary result of spectra predictions vs their true values is below, and the pdf of the full report
is attached in this repo:
<img width="726" height="535" alt="Screenshot 2026-10-13 at 19 44 19" src="https://github.com/user-attachments/assets/35b0431a-fa58-4fac-b4f0-f9000fd711eb" />


# Excited Energy Predictor.ipynb
This model takes smile strings of small organic molecules and tokenises them using the bert-base-case tokeniser, which vectorises each symbol into the smile string into a 768-dimensional vector.
These vectors are then passed through 40 x [Attention + MLP] layers, with a final MLP head, to predict the 1st excited state energy of the molecule.
It was trained on 20,000 molecule smile strings and their energies via the Kaggle data set ‘qm8’, and sent to a HPC (runpod.io RTX5090 GPU).
It is validated on 1787 molecules as follows, with a MAE of +/-0.384 eV:
<img width="884" height="884" alt="image" src="https://github.com/user-attachments/assets/1adb0ddd-36b1-4d2f-884f-79d0d3fc5250" />


# Atomic Interaction Energy Modeller.ipynb
Minimal neural network which learns inter-orbital potentials between common organic atoms, by separately learning Kinetic (T), Nuclear Potential
(V), Electron Coulomb Repulsion (J) and Electron Exchange Energy (K) by inter-atomic distance.
Could be used to provide rapid estimates to molecular energy, though doesn't account for four-fold-orbital J and K interactions.
<img width="886" height="590" alt="Inter-atomic Energy" src="https://github.com/user-attachments/assets/d73607db-8696-4280-883c-6f8864905385" />


# Additional
There are further weather-modelling projects, using image-processing with ConvLSTM and Random Forest models, which I cannot show the code nor results for
because I am bound by non-disclosure agreements.
