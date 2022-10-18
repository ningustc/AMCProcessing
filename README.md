# AMC-Processing
Processing electron mciroscopy images of amorphous carbon monolayers
1. The python codes are used to preprocessing expermental ADF-STEM images. These processings includes denoise, carbon ring detection, and generation of patches including these carbon rings, voronoi meshing. 
2. The C++ codes are used to fit the ADF-STEM patches using the GMM model. The Gaussian parameters of each carbon columns are iteratively refined using the Levenberg-Marquardt algorithm, and GPU acceleration based on NVIDIA CUDA is adopted to enhance the performance of iterative optimization.

The Levenberg-Marquardt library is compiled into a C++ library files (LMFit.lib, LMFit.dll).
