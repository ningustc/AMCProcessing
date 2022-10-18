#ifndef CUDA_FIT_H
#define CUDA_FIT_H
#include "gaussian_error.cuh"
#include "psf_error.cuh"
#include "gaussian_error_periodic.cuh"
#include "psf_position_error.cuh"
#include "dense_jacobian.cuh"
#include "lmfit.cuh"
namespace GMMProc 
{
	class GPUFit
	{
	public:
		void geneGaussianBlobs(std::vector<float>& img, const int &row, const int &col,
			const std::vector<float>& params, bool enable_general, bool same_sigma)
		{
			std::size_t param_num = params.size();
			std::size_t point_num = row*col;
			float* param_ptr, *frame_ptr;
			cudaMalloc((void**)&param_ptr, sizeof(float)*param_num);
			cudaMalloc((void**)&frame_ptr, sizeof(float)*point_num);
			cudaMemcpy(param_ptr, params.data(), sizeof(float)*param_num, cudaMemcpyHostToDevice);
			GaussianError<float> Gaussian_gene;
			Gaussian_gene.setMode(enable_general, same_sigma);
			Gaussian_gene.initial(row, col, param_num);
			Gaussian_gene.geneBlobs(param_ptr, frame_ptr);
			img.resize(point_num);
			cudaMemcpy(img.data(), frame_ptr, sizeof(float)*point_num, cudaMemcpyDeviceToHost);
		}
		void genPGaussianBlobs(std::vector<float>& img, const int &row, const int &col,
			const std::vector<float>& params, const std::vector<int>& unitcell_idx,
			const std::vector<float> &basis_vec, bool enable_general)
		{
			std::size_t param_num = params.size();
			std::size_t point_num = row*col;
			float* param_ptr, *frame_ptr;
			cudaMalloc((void**)&param_ptr, sizeof(float)*param_num);
			cudaMalloc((void**)&frame_ptr, sizeof(float)*point_num);
			cudaMemcpy(param_ptr, params.data(), sizeof(float)*param_num, cudaMemcpyHostToDevice);
			GaussianErrorPeriodic<float> Gaussian_gene;
			Gaussian_gene.setUCellInfo(unitcell_idx, basis_vec, enable_general);
			Gaussian_gene.initial(row, col, param_num);
			Gaussian_gene.geneBlobs(param_ptr, frame_ptr);
			img.resize(point_num);
			cudaMemcpy(img.data(), frame_ptr, sizeof(float)*point_num, cudaMemcpyDeviceToHost);
		}
		float fitGaussianParams(const std::vector<float>& img, const int &row, const int &col,
			std::vector<float>& params, bool enable_general, bool same_sigma, float threshold_val = 1e-5f,
			bool show_log = true)
		{
			std::size_t param_num = params.size();
			//begin the iterations.
			GaussianError<float> Gaussian_gene;
			JacobianDSGenerator<float> jacobianGene;
			Gaussian_gene.setMode(enable_general, same_sigma);
			Gaussian_gene.initial(row, col, param_num);
			filterOption<float> options_i;
			options_i.show_progress = show_log;
			options_i.Tolerance = threshold_val;
			jacobianGene.initial(&Gaussian_gene);
			LMFitter<float> fitters(options_i, params, &jacobianGene, img);
			fitters.Minimize();
			fitters.exportParams(params);
			return fitters.exportError();
		}
		float fitPGaussianParams(const std::vector<float>& img, const int &row, const int &col,
			std::vector<float>& params, const std::vector<int>& unitcell_idx,
			const std::vector<float> &basis_vec, bool enable_general, float threshold_val = 1e-5f,
			bool show_log = true)
		{
			std::size_t param_num = params.size();
			//begin the iterations.
			GaussianErrorPeriodic<float> Gaussian_gene;
			JacobianDSGenerator<float> jacobianGene;
			Gaussian_gene.setUCellInfo(unitcell_idx, basis_vec, enable_general);
			Gaussian_gene.initial(row, col, param_num);
			filterOption<float> options_i;
			options_i.show_progress = show_log;
			options_i.Tolerance = threshold_val;
			jacobianGene.initial(&Gaussian_gene);
			LMFitter<float> fitters(options_i, params, &jacobianGene, img);
			fitters.Minimize();
			fitters.exportParams(params);
			return fitters.exportError();
		}
		/************************Solve PSF************************/
		void genePSFFrame(std::vector<float> & img, const int &row, const int &col, const std::vector<float>& psf,
			const std::vector<float> &pts_pos, const std::vector<int> &pts_type_i, const std::vector<float> &type_ratio_i,
			const float &scale_val_i, const int &psf_dim_i)
		{
			std::size_t param_num = psf.size();
			std::size_t point_num = row*col;
			float* param_ptr, *frame_ptr;
			cudaMalloc((void**)&param_ptr, sizeof(float)*param_num);
			cudaMalloc((void**)&frame_ptr, sizeof(float)*point_num);
			cudaMemcpy(param_ptr, psf.data(), sizeof(float)*param_num, cudaMemcpyHostToDevice);
			PSFError<float> PSF_gene;
			PSF_gene.initial(row, col, param_num);
			//initialization and then set the PSF Info
			PSF_gene.setPSFInfo(pts_pos, pts_type_i, type_ratio_i, scale_val_i, psf_dim_i);
			PSF_gene.geneBlobs(param_ptr, frame_ptr);
			img.resize(point_num);
			cudaMemcpy(img.data(), frame_ptr, sizeof(float)*point_num, cudaMemcpyDeviceToHost);
		}
		float fitPSF2D(const std::vector<float> & img, const int &row, const int &col, std::vector<float>& psf, 
			const std::vector<float> &pts_pos, const std::vector<int> &pts_type_i, const std::vector<float> &type_ratio_i,
			const float &scale_val_i, const int &psf_dim_i, float threshold_val = 1e-5f, bool show_log = true)
		{
			std::size_t param_num = psf.size();
			//begin the iterations.
			PSFError<float> PSF_gene;
			JacobianDSGenerator<float> jacobianGene;
			PSF_gene.initial(row, col, param_num);
			PSF_gene.setPSFInfo(pts_pos, pts_type_i, type_ratio_i, scale_val_i, psf_dim_i);
			filterOption<float> options_i;
			options_i.show_progress = show_log;
			options_i.Tolerance = threshold_val;
			jacobianGene.initial(&PSF_gene);
			LMFitter<float> fitters(options_i, psf, &jacobianGene, img);
			fitters.Minimize();
			fitters.exportParams(psf);
			return fitters.exportError();
		}
		/************************Fit Pos According PSF************************/
		void genePSFPosFrame(std::vector<float>& img, const int &row, const int &col, const std::vector<float>& params,
			const std::vector<int> &pts_type_i, const std::vector<float> & psf_val_i, float scal_val_i, const int &psf_dim_i)
		{
			std::size_t param_num = params.size();
			std::size_t point_num = row*col;
			float* param_ptr, *frame_ptr;
			cudaMalloc((void**)&param_ptr, sizeof(float)*param_num);
			cudaMalloc((void**)&frame_ptr, sizeof(float)*point_num);
			cudaMemcpy(param_ptr, params.data(), sizeof(float)*param_num, cudaMemcpyHostToDevice);
			PSFPositionError<float> PSF_gene;
			PSF_gene.initial(row, col, param_num);
			PSF_gene.setPSF(psf_val_i, pts_type_i, psf_dim_i, scal_val_i);
			PSF_gene.geneBlobs(param_ptr, frame_ptr);
			img.resize(point_num);
			cudaMemcpy(img.data(), frame_ptr, sizeof(float)*point_num, cudaMemcpyDeviceToHost);
		}
		float fitPSF2DPos(const std::vector<float> & img, const int &row, const int &col, std::vector<float>& params,
			const std::vector<float> &psf_val, const std::vector<int> &pts_type_i, float scal_val_i, const int &psf_dim_i,
			float threshold_val = 1e-5f, bool show_log = true)
		{
			std::size_t param_num = params.size();
			//begin the iterations.
			PSFPositionError <float> PSF_gene;
			JacobianDSGenerator<float> jacobianGene;
			PSF_gene.initial(row, col, param_num);
			PSF_gene.setPSF(psf_val, pts_type_i, psf_dim_i, scal_val_i);
			filterOption<float> options_i;
			options_i.show_progress = show_log;
			options_i.Tolerance = threshold_val;
			jacobianGene.initial(&PSF_gene);
			LMFitter<float> fitters(options_i, params, &jacobianGene, img);
			fitters.Minimize();
			fitters.exportParams(params);
			return fitters.exportError();
		}
	};
}
#endif