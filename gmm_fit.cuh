#ifndef GAUSSIAN_BLOB_FITTING_H
#define GAUSSIAN_BLOB_FITTING_H
#include "cufit.cuh"
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
namespace GMMProc
{
	template <class T>
	struct AtomCol
	{
		float x_pos, y_pos;
		T intensity, sigma;
		AtomCol() : x_pos(T(0)), y_pos(T(0)), intensity(T(0)), sigma(T(0)){}
		AtomCol(const T& x_pos_i, const T& y_pos_i,const T& sigma_i, const T& intensity_i) :
			x_pos(x_pos_i), y_pos(y_pos_i), intensity(intensity_i), sigma(sigma_i)  {}
		T getWeight()
		{
			return intensity * sigma;
		}
	};
	struct GMMParams
	{
		float thresh_ratio;												//ratio between the new blob and fitted blob during filtering.
		int batch_size, min_num, max_iter;								//batch size when adding new blobs, minimum column number, maximum iteration number
		bool enable_blur, same_sigma, symm;								//enable the blur of residual frame, share the same sigma, enable symmetry
		float blur_sigma_val, blob_sigma_val;							//sigma values for blurring and initialization of the new blob.
		float intensity_threshold, converge_threshold, dist_threshold;	//Threshold values including intensity, error, and distance
		GMMParams() :symm(false), enable_blur(false), same_sigma(false),
			blur_sigma_val(1.0f), blob_sigma_val(1.0f), batch_size(4),
			intensity_threshold(1e-2f), converge_threshold(1e-4f), 
			dist_threshold(10.0f), thresh_ratio(0.2f), min_num(0),
			max_iter(30) {}
	};
	void putMarker(const cv::Mat &img, const std::vector<cv::Point> &col_pos,
		const std::string &win_name = "marker")
	{
		cv::Mat outputImage;
		cvtColor(img, outputImage, CV_GRAY2RGB);
		for (auto ipos = 0; ipos < col_pos.size(); ipos++)
		{
			cv::circle(outputImage, col_pos[ipos], 3, cv::Scalar(0, 0, 255), -1);
		}
		cv::imshow(win_name, outputImage);
		cv::waitKey(0);
	}
	float getSTD(const std::vector<float> &data_i)
	{
		float sum = std::accumulate(std::begin(data_i), std::end(data_i), 0.0);
		float m = sum / data_i.size();
		float accum = 0.0;
		std::for_each(std::begin(data_i), std::end(data_i), [&](const float d) 
		{
			accum += (d - m) * (d - m);
		});
		return sqrt(accum / (data_i.size() - 1));
	}
	//Detect coarse atom columns. 
	class GMMEngine
	{
	public:
		GMMEngine()
		{
			_setDefaultValues();
		}
		GMMEngine(const cv::Mat &img_i, const GMMParams& input_params_i)
		{
			_setDefaultValues();
			updateParams(input_params_i);
			updateFrame(img_i);
		}
		void updateParams(const GMMParams& input_params_i)
		{
			m_param_ptr = &input_params_i;
			m_sigma = m_param_ptr->blob_sigma_val;
		}
		void updateFrame(const cv::Mat &img_i)
		{
			m_columns.clear();
			m_row = img_i.rows;
			m_col = img_i.cols;
			m_frame_size = m_row*m_col;
			img_i.copyTo(m_image);
			img_i.copyTo(m_residual);
			img_i.copyTo(m_gaussian);
			m_frame.resize(m_frame_size);
			memcpy(m_frame.data(), m_image.data, sizeof(float)*m_frame_size);
			m_background = *std::min_element(m_frame.begin(), m_frame.end());
			m_threshold = m_param_ptr->intensity_threshold*getSTD(m_frame);
		}
		void importGuess(const std::vector<float> &pts_pos_i)
		{
			for (auto icol =0; icol< pts_pos_i.size()/2; icol++)
			{
				int y_pos = int(pts_pos_i[icol * 2]);
				int x_pos = int(pts_pos_i[icol * 2 + 1]);
				m_columns.push_back(AtomCol<float>(x_pos, y_pos, m_sigma, m_frame[x_pos+y_pos*m_col]-m_background));
			}
			_updateThreshold(); //reset the intensity threshold
			_singleCycle();		//refine the atom columns and update the residual frame
			_updateThreshold(); //update the threshold value again.
		}
		//before the fitting, the minimum number of the parameter must be initialized.
		void importColumns(const std::vector<float> &pts_pos_i, const std::vector<float> &sigma_i, 
			const std::vector<float> &intensity_i)
		{
			if (m_param_ptr->min_num == 0)
			{
				std::cout << "please initialize the atom column number in the parameters." << std::endl;
				std::exit(0);
			}
			if ((intensity_i.size()!= sigma_i.size()) || (pts_pos_i.size()/2 != sigma_i.size()))
			{
				std::cout << "The input dimension does not match." << std::endl;
				std::exit(0);
			}
			m_sigma = std::accumulate(sigma_i.begin(), sigma_i.end(), 0.0f) / sigma_i.size();
			//import the known atom columns.
			for (auto icol = 0; icol < pts_pos_i.size() / 2; icol++)
			{
				int y_pos = int(pts_pos_i[icol * 2]);
				int x_pos = int(pts_pos_i[icol * 2 + 1]);
				m_columns.push_back(AtomCol<float>(x_pos, y_pos, sigma_i[icol], intensity_i[icol]));
			}
			_updateThreshold(); //reset the intensity threshold
			_singleCycle();		//refine the atom columns and update the residual frame
			_updateThreshold(); //update the threshold value again.
		}
		void geneAtomCols()
		{
			//stop when the size of the columns does not change.
			int iter_num = 0;
			float curr_error = 0.0f;
			float prev_error = -1.0f;
			while (std::fabs(curr_error-prev_error)>m_param_ptr->converge_threshold&&
				iter_num<m_param_ptr->max_iter)
			{
				prev_error = curr_error;
				_singleCycle();
				curr_error = m_error;
				iter_num++;
			}
		}
		void exportColumns(std::vector<float> &pts_pos_o, std::vector<float>& sigma_o, std::vector<float>& intensity_o)
		{
			sigma_o.clear();
			pts_pos_o.clear();
			intensity_o.clear();
			for (auto icol = 0; icol < m_columns.size(); icol++)
			{
				sigma_o.push_back(m_columns[icol].sigma);
				pts_pos_o.push_back(m_columns[icol].x_pos);
				pts_pos_o.push_back(m_columns[icol].y_pos);
				intensity_o.push_back(m_columns[icol].intensity);
			}
		}
		void showPeaks(bool filter)
		{
			std::vector<cv::Point> pixel_posititions;
			for (auto icol = 0; icol < m_columns.size(); icol++)
			{
				float x_pos = m_columns[icol].x_pos;
				float y_pos = m_columns[icol].y_pos;
				if (filter&& !_insideBound(x_pos, y_pos, m_param_ptr->blob_sigma_val)) continue;
				pixel_posititions.push_back(cv::Point(std::round(x_pos), std::round(y_pos)));
			}
			putMarker(m_image, pixel_posititions, "Origin");
		}
		void showResidual()
		{
			cv::imshow("residual", m_residual);
			cv::waitKey(0);
		}
	protected:
		void _singleCycle()
		{
			_addColumns();		//compute extra peaks on the reminded frame.
			_refineColumns();	//refine the positions.
			if (_filterColumns()) _refineColumns();
			m_residual = m_image - m_gaussian;
			//showPeaks(false);
		}
		void _setDefaultValues()
		{
			m_row = 0;
			m_col = 0;
			m_theta = 0.0f;
			m_ratio = 1.0f;
			m_frame_size = 0;
		}
		void _refineColumns()
		{
			std::vector<float> temp_params;
			std::size_t col_num = m_columns.size();
			if (m_param_ptr->symm)
			{
				if (m_param_ptr->same_sigma)
				{
					temp_params.resize(3 * col_num + 2);
					temp_params[col_num * 3] = m_sigma;
					temp_params[col_num * 3 + 1] = m_background;
				}
				else
				{
					temp_params.resize(4 * col_num + 1);
					temp_params[col_num * 4] = m_background;
				}
			}
			else
			{
				if (m_param_ptr->same_sigma)
				{
					temp_params.resize(3 * col_num + 4);
					temp_params[col_num * 3] = m_sigma;
					temp_params[col_num * 3 + 1] = m_ratio;
					temp_params[col_num * 3 + 2] = m_theta;
					temp_params[col_num * 3 + 3] = m_background;
				}
				else
				{
					temp_params.resize(4 * col_num + 3);
					temp_params[col_num * 4] = m_ratio;
					temp_params[col_num * 4 + 1] = m_theta;
					temp_params[col_num * 4 + 2] = m_background;
				}
			}
			_convertFromColumns(temp_params);
			m_error = m_fitting.fitGaussianParams(m_frame, m_row, m_col, temp_params, !m_param_ptr->symm, 
				m_param_ptr->same_sigma, m_param_ptr->converge_threshold / 10.0f, false);
			_convert2Columns(temp_params, col_num);
			if (m_param_ptr->symm)
			{
				if (m_param_ptr->same_sigma)
				{
					m_sigma = temp_params[col_num * 3];
					m_background = temp_params[col_num * 3 + 1];
				}
				else
					m_background = temp_params[col_num * 4];
			}
			else
			{
				if (m_param_ptr->same_sigma)
				{
					m_sigma = temp_params[col_num * 3];
					m_ratio = temp_params[col_num * 3 + 1];
					m_theta = temp_params[col_num * 3 + 2];
					m_background = temp_params[col_num * 3 + 3];
				}
				else
				{
					m_ratio = temp_params[col_num * 4];
					m_theta = temp_params[col_num * 4 + 1];
					m_background = temp_params[col_num * 4 + 2];
				}
			}
			std::vector<float> gaussian_frame;
			m_fitting.geneGaussianBlobs(gaussian_frame, m_row, m_col, temp_params, !m_param_ptr->symm, m_param_ptr->same_sigma);
			memcpy(m_gaussian.data, gaussian_frame.data(), sizeof(float)*m_frame_size);
			std::cout << "current error: " << m_error << std::endl;
		}
		void _addColumns()
		{
			//detect new Gaussian peaks.
			cv::Mat blured_frame;
			std::vector<AtomCol<float>> gpeak_temp;
			if (!m_param_ptr->enable_blur) m_residual.copyTo(blured_frame);
			else cv::GaussianBlur(m_residual, blured_frame, cv::Size(0, 0), m_param_ptr->blur_sigma_val);
			int dialate_size = std::ceil(m_sigma);
			cv::Mat dilated(m_row, m_col, blured_frame.type());
			cv::Mat element = cv::getStructuringElement(2, cv::Size(dialate_size, dialate_size));
			cv::dilate(blured_frame, dilated, element);
			for (auto i = 0; i < m_row; i++)
			{
				const float* src1 = (const float*)(blured_frame.data + blured_frame.step[0] * i);
				const float* src2 = (const float*)(dilated.data + dilated.step[0] * i);
				for (auto j = 0; j < m_col; j++)
				{
					if (src1[j] == src2[j]) 
						gpeak_temp.push_back(AtomCol<float>(j, i, m_sigma, src1[j]));
				}
			}
			//sort the new Gaussian peaks by their intensity
			auto sort_by_intensity = [=](AtomCol<float> a, AtomCol<float> b)->bool
			{
				return a.intensity > b.intensity;
			};
			std::sort(gpeak_temp.begin(), gpeak_temp.end(), sort_by_intensity);
			//add the new columns by batch
			int effect_num = 0;
			int batch_num = m_param_ptr->batch_size < 1 ? gpeak_temp.size() : m_param_ptr->batch_size;
			for (auto icol = 0; icol < gpeak_temp.size(); icol++)
			{
				float x_pos = gpeak_temp[icol].x_pos;
				float y_pos = gpeak_temp[icol].y_pos;
				//we have a lower demanded for the new positions.
				if ((!_close2Columns(x_pos, y_pos)) &&
					(gpeak_temp[icol].intensity > m_threshold*m_param_ptr->thresh_ratio))
				{
					effect_num++;
					m_columns.push_back(gpeak_temp[icol]);
				}
				if (effect_num == batch_num) break;
			}
		}
		void _updateThreshold()
		{
			float avg_intensity = 0.0f;
			for (auto icol = 0; icol < m_columns.size(); icol++)
			{
				avg_intensity += m_columns[icol].intensity;
			}
			m_threshold = avg_intensity*m_param_ptr->intensity_threshold / m_columns.size();
		}
		bool _insideBound(float x_pos, float y_pos, float bound)
		{
			if (x_pos > bound&&x_pos < (m_col - bound) &&
				y_pos > bound&&y_pos < (m_row - bound))
			{
				return true;
			}
			else return false;
		}
		bool _close2Columns(float x_pos, float y_pos)
		{
			for (auto icol = 0; icol < m_columns.size(); icol++)
			{
				float distance_val = std::pow(x_pos - m_columns[icol].x_pos, 2);
				distance_val += std::pow(y_pos - m_columns[icol].y_pos, 2);
				distance_val = std::sqrt(distance_val);
				if (distance_val < m_param_ptr->dist_threshold) return true;
			}
			return false;
		}
		bool _filterColumns()
		{
			std::size_t prev_len = m_columns.size();
			std::vector<AtomCol<float>> gpeak_temp;
			//keep the specified atom columns 
			gpeak_temp.assign(m_columns.begin() + m_param_ptr->min_num, m_columns.end());
			//sort the remaining columns by intensities
			auto sort_by_intensity = [=](AtomCol<float> a, AtomCol<float> b)->bool
			{
				return a.getWeight() > b.getWeight();
			};
			std::sort(gpeak_temp.begin(), gpeak_temp.end(), sort_by_intensity);
			//remove the too close point, and remove the columns lower than the intensity threshold.
			m_columns.clear();
			for (auto icol = 0; icol < gpeak_temp.size(); icol++)
			{
				float x_pos = gpeak_temp[icol].x_pos;
				float y_pos = gpeak_temp[icol].y_pos;
				if ((!_close2Columns(x_pos, y_pos)) && gpeak_temp[icol].intensity > m_threshold)
				{
					m_columns.push_back(gpeak_temp[icol]);
				}
			}
			return prev_len != m_columns.size();
		}
		void _convert2Columns(const std::vector<float> &temp_params, int col_num)
		{
			m_columns.clear();
			for (auto icol = 0; icol < col_num; icol++)
			{
				if (m_param_ptr->same_sigma)
					m_columns.push_back(AtomCol<float>(temp_params[icol * 3], temp_params[icol * 3 + 1], m_sigma, temp_params[icol * 3 + 2]));
				else
					m_columns.push_back(AtomCol<float>(temp_params[icol * 4], temp_params[icol * 4 + 1], temp_params[icol * 4 + 2], temp_params[icol * 4 + 3]));
			}
		}
		void _convertFromColumns(std::vector<float> &temp_params)
		{
			std::size_t peak_num = m_columns.size();
			for (auto icol = 0; icol < peak_num; icol++)
			{
				if (m_param_ptr->same_sigma)
				{
					temp_params[icol * 3] = m_columns[icol].x_pos;
					temp_params[icol * 3 + 1] = m_columns[icol].y_pos;
					temp_params[icol * 3 + 2] = m_columns[icol].intensity;
				}
				else
				{
					temp_params[icol * 4] = m_columns[icol].x_pos;
					temp_params[icol * 4 + 1] = m_columns[icol].y_pos;
					temp_params[icol * 4 + 2] = m_columns[icol].sigma;
					temp_params[icol * 4 + 3] = m_columns[icol].intensity;
				}
			}
		}
		int m_row, m_col;
		GPUFit m_fitting;
		std::size_t m_frame_size;
		std::vector<float> m_frame;
		const GMMParams* m_param_ptr;
		std::vector<AtomCol<float>> m_columns;
		cv::Mat m_image, m_residual, m_gaussian;
		float m_theta, m_ratio, m_background, m_sigma, m_error, m_threshold;
	};
}
#endif