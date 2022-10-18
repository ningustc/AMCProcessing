#include "cufit.cuh"
#include "emio.hpp"
#include "gmm_fit.cuh"
using namespace GMMProc;
void NormMat(cv::Mat &img_o)
{
	double min_val, max_val;
	cv::minMaxLoc(img_o, &min_val, &max_val);
	img_o = (img_o - min_val) / (max_val - min_val);
}
void vector2MatNorm(cv::Mat &img_o, const std::vector<float>&vector_data, const int &img_row, const int&img_col)
{
	img_o = cv::Mat(img_row, img_col, CV_32FC1);
	memcpy(img_o.data, vector_data.data(), sizeof(float)*img_row*img_col);
	NormMat(img_o);
}
void detAllCols()
{
	int row, col, frame_num;
	EMIOFunctions file_reader;
	std::vector<float> image, pts_pos;
	GMMParams gmmparams;
	gmmparams.symm = false;
	gmmparams.batch_size = 0;
	gmmparams.same_sigma = true;
	gmmparams.enable_blur = false;
	gmmparams.thresh_ratio = 0.5f;
	gmmparams.blob_sigma_val = 5.0f;
	gmmparams.dist_threshold = 10.0f;
	gmmparams.converge_threshold = 1e-5f;
	gmmparams.intensity_threshold = 0.3f;
	for (auto i=0; i<= 255; i++)
	{
		std::cout << i << std::endl;
		std::string subfix = std::to_string(i) + ".npy";
		file_reader.readNumpyArray<float>("C:/Users/sning/Desktop/Mengyao2/col_pos" + subfix, pts_pos, row, col, frame_num);
		if (pts_pos.size()==0) continue;
		file_reader.readNumpyArray<float>("C:/Users/sning/Desktop/Mengyao2/region" + subfix, image, row, col, frame_num);
		//normalize the image start to GMM
		cv::Mat norm_image;
		vector2MatNorm(norm_image, image, row, col);
		//input the data into GMM engine to determine the positions.
		GMMEngine gmmfit(norm_image, gmmparams);
		gmmfit.importGuess(pts_pos);
		gmmfit.geneAtomCols();
		//gmmfit.exportColumns(pts_pos);
		file_reader.writeNumpyArray("C:/Users/sning/Desktop/Mengyao2/col_det" + subfix, pts_pos, pts_pos.size() / 2, 2);
	}
}
void detMACCols()
{
	int row, col, frame_num;
	EMIOFunctions file_reader;
	GMMParams gmmparams;
	gmmparams.symm = false;
	gmmparams.batch_size = 0;
	gmmparams.same_sigma = true;
	gmmparams.enable_blur = false;
	gmmparams.thresh_ratio = 0.3f;
	gmmparams.blob_sigma_val = 6.0f;
	gmmparams.dist_threshold = 12.0f;
	gmmparams.converge_threshold = 1e-5f;
	gmmparams.intensity_threshold = 0.2f;
	std::vector<float> image, pts_pos, pts_intensity, pts_sigma;
	for (auto i = 0; i <= 1601; i++)
	{
		std::cout << i << std::endl;
		std::string subfix = std::to_string(i) + ".npy";
		file_reader.readNumpyArray<float>("D:/GMM500/carbon_pos" + subfix, pts_pos, row, col, frame_num);
		if (pts_pos.size() == 0) continue;
		file_reader.readNumpyArray<float>("D:/GMM500/region" + subfix, image, row, col, frame_num);
		//normalize the image start to GMM
		cv::Mat norm_image;
		vector2MatNorm(norm_image, image, row, col);
		//input the data into GMM engine to determine the positions.
		GMMEngine gmmfit(norm_image, gmmparams);
		gmmfit.importGuess(pts_pos);
		gmmfit.geneAtomCols();
		gmmfit.exportColumns(pts_pos, pts_sigma, pts_intensity);
		file_reader.writeNumpyArray("D:/GMM500/subpos" + subfix, pts_pos, pts_pos.size() / 2, 2);
		file_reader.writeNumpyArray("D:/GMM500/col_sigma" + subfix, pts_sigma, pts_sigma.size(), 1);
		file_reader.writeNumpyArray("D:/GMM500/col_intensity" + subfix, pts_intensity, pts_intensity.size(), 1);
	}
}
void GaussianBlobGene()
{
	int pts_num = 2;
	int row = 64, col = 64;
	std::vector<float> parameters(3 * pts_num + 2, 0.0f);
	parameters[0] = col / 2.0f;
	parameters[1] = row / 2.0f - 3.0f;
	parameters[2] = 1.0f;
	parameters[3] = col / 2.0f;
	parameters[4] = row / 2.0f + 3.0f;
	parameters[5] = 1.0f;
	//sigma value and background value.
	parameters[6] = 4.0f;
	parameters[7] = 0.0f;
	GPUFit fitdemo;
	std::vector<float> image;
	fitdemo.geneGaussianBlobs(image, row, col, parameters, false, true);
	EMIOFunctions file_reader;
	file_reader.writeNumpyArray("D:/GMM/simpleD6.npy", image,row,col);
}
void GMMFit()
{
	GMMParams gmmparams;
	gmmparams.symm = true;
	gmmparams.batch_size = 1;
	gmmparams.same_sigma = true;
	gmmparams.enable_blur = false;
	gmmparams.thresh_ratio = 0.3f;
	gmmparams.blob_sigma_val = 4.0f;
	gmmparams.dist_threshold = 1.0f;
	gmmparams.converge_threshold = 1e-5f;
	gmmparams.intensity_threshold = 0.01f;
	//read the 2D frame
	int row, col, frame_num;
	std::vector<float> image;
	EMIOFunctions file_reader;
	file_reader.readNumpyArray<float>("D:/GMM/simpleD6.npy", image, row, col, frame_num);
	//normalize the image start to GMM
	cv::Mat norm_image;
	vector2MatNorm(norm_image, image, row, col);
	//input the data into GMM engine to determine the positions.
	GMMEngine gmmfit(norm_image, gmmparams);
	gmmfit.geneAtomCols();
}
int main()
{
	detAllColsS();
	return 0;
}
