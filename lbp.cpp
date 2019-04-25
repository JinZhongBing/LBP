/*
   name:Jzb
   所有图像用unsigned char  既领域只能为8
   圆形灰度等级不变模式LBP     pattren=256
   旋转不变模式LBP ri          pattren=256
   等价模式LBP u2              pattren=59
   没有旋转等价模式LBP ri u2
*/

/*
   lbp图片为8位
   histogram是CV_32FC1
*/




#include"lbp.h"

//原始方形lbp
void olbp_(InputArray _src, OutputArray _dst) {
	// get matrices
	Mat src = _src.getMat();
	// allocate memory for result
	_dst.create(src.rows - 2, src.cols - 2, CV_8UC1);
	Mat dst = _dst.getMat();
	// zero the result matrix
	dst.setTo(0);
	// calculate patterns
	for (int i = 1; i<src.rows - 1; i++) {
		for (int j = 1; j<src.cols - 1; j++) {
			unsigned char center = src.at<unsigned char>(i, j);
			unsigned char code = 0;
			code |= (src.at<unsigned char>(i - 1, j - 1) >= center) << 7;
			code |= (src.at<unsigned char>(i - 1, j) >= center) << 6;
			code |= (src.at<unsigned char>(i - 1, j + 1) >= center) << 5;
			code |= (src.at<unsigned char>(i, j + 1) >= center) << 4;
			code |= (src.at<unsigned char>(i + 1, j + 1) >= center) << 3;
			code |= (src.at<unsigned char>(i + 1, j) >= center) << 2;
			code |= (src.at<unsigned char>(i + 1, j - 1) >= center) << 1;
			code |= (src.at<unsigned char>(i, j - 1) >= center) << 0;
			dst.at<unsigned char>(i - 1, j - 1) = code;
		}
	}
}

//------------------------------------------------------------------------------
// Calculate the lbp feature mapping
//------------------------------------------------------------------------------

//圆形LBP特征计算，这种方法适于理解，但在效率上存在问题，声明时默认neighbors=8
void getCircularLBPFeature(InputArray _src, OutputArray _dst, int radius, int neighbors)
{
	//get matrices
	Mat src = _src.getMat();
	// allocate memory for result
	_dst.create(src.rows - 2 * radius, src.cols - 2 * radius, CV_8UC1);
	Mat dst = _dst.getMat();
	// zero
	dst.setTo(0);
	for (int n = 0; n<neighbors; n++) {
		// sample points
		float x = static_cast<float>(radius * cos(2.0*CV_PI*n / static_cast<float>(neighbors)));
		float y = static_cast<float>(-radius * sin(2.0*CV_PI*n / static_cast<float>(neighbors)));
		// relative indices
		int fx = static_cast<int>(floor(x));
		int fy = static_cast<int>(floor(y));
		int cx = static_cast<int>(ceil(x));
		int cy = static_cast<int>(ceil(y));
		// fractional part
		float ty = y - fy;
		float tx = x - fx;
		// set interpolation weights
		float w1 = (1 - tx) * (1 - ty);
		float w2 = tx  * (1 - ty);
		float w3 = (1 - tx) *      ty;
		float w4 = tx  *      ty;
		// iterate through your data
		for (int i = radius; i < src.rows - radius; i++) {
			for (int j = radius; j < src.cols - radius; j++) {
				// calculate interpolated value
				float t = static_cast<float>(w1*src.at<unsigned char>(i + fy, j + fx) + w2*src.at<unsigned char>(i + fy, j + cx) + w3*src.at<unsigned char>(i + cy, j + fx) + w4*src.at<unsigned char>(i + cy, j + cx));
				// floating point precision, so check some machine-dependent epsilon
				dst.at<unsigned char>(i - radius, j - radius) += ((t > src.at<unsigned char>(i, j)) || (std::abs(t - src.at<unsigned char>(i, j)) < std::numeric_limits<float>::epsilon())) << n;
			}
		}
	}
}

//旋转不变圆形LBP特征计算，声明时默认neighbors=8
void getRotationInvariantLBPFeature(InputArray _src, OutputArray _dst, int radius, int neighbors)
{
	//get matrices
	Mat src = _src.getMat();
	// allocate memory for result
	_dst.create(src.rows - 2 * radius, src.cols - 2 * radius, CV_8UC1);
	Mat dst = _dst.getMat();
	// zero
	dst.setTo(0);
	for (int n = 0; n<neighbors; n++) {
		// sample points
		float x = static_cast<float>(radius * cos(2.0*CV_PI*n / static_cast<float>(neighbors)));
		float y = static_cast<float>(-radius * sin(2.0*CV_PI*n / static_cast<float>(neighbors)));
		// relative indices
		int fx = static_cast<int>(floor(x));
		int fy = static_cast<int>(floor(y));
		int cx = static_cast<int>(ceil(x));
		int cy = static_cast<int>(ceil(y));
		// fractional part
		float ty = y - fy;
		float tx = x - fx;
		// set interpolation weights
		float w1 = (1 - tx) * (1 - ty);
		float w2 = tx  * (1 - ty);
		float w3 = (1 - tx) *      ty;
		float w4 = tx  *      ty;
		// iterate through your data
		for (int i = radius; i < src.rows - radius; i++) {
			for (int j = radius; j < src.cols - radius; j++) {
				// calculate interpolated value
				float t = static_cast<float>(w1*src.at<unsigned char>(i + fy, j + fx) + w2*src.at<unsigned char>(i + fy, j + cx) + w3*src.at<unsigned char>(i + cy, j + fx) + w4*src.at<unsigned char>(i + cy, j + cx));
				// floating point precision, so check some machine-dependent epsilon
				dst.at<unsigned char>(i - radius, j - radius) += ((t > src.at<unsigned char>(i, j)) || (std::abs(t - src.at<unsigned char>(i, j)) < std::numeric_limits<float>::epsilon())) << n;
			}
		}
	}
	//进行旋转不变处理
	for (int i = 0; i<dst.rows; i++)
	{
		for (int j = 0; j<dst.cols; j++)
		{
			unsigned char currentValue = dst.at<uchar>(i, j);
			unsigned char minValue = currentValue;
			for (int k = 1; k<neighbors; k++)		//循环左移
			{
				unsigned char temp = (currentValue >> (neighbors - k)) | (currentValue << k);
				if (temp < minValue)
				{
					minValue = temp;
				}
			}
			dst.at<uchar>(i, j) = minValue;
		}
	}
}

//等价模式LBP特征计算
void getUniformPatternLBPFeature(InputArray _src, OutputArray _dst, int radius, int neighbors)
{
	//get matrices
	Mat src = _src.getMat();
	// allocate memory for result
	_dst.create(src.rows - 2 * radius, src.cols - 2 * radius, CV_8UC1);
	Mat dst = _dst.getMat();
	// zero
	dst.setTo(0);
	for (int n = 0; n<neighbors; n++) {
		// sample points
		float x = static_cast<float>(radius * cos(2.0*CV_PI*n / static_cast<float>(neighbors)));
		float y = static_cast<float>(-radius * sin(2.0*CV_PI*n / static_cast<float>(neighbors)));
		// relative indices
		int fx = static_cast<int>(floor(x));
		int fy = static_cast<int>(floor(y));
		int cx = static_cast<int>(ceil(x));
		int cy = static_cast<int>(ceil(y));
		// fractional part
		float ty = y - fy;
		float tx = x - fx;
		// set interpolation weights
		float w1 = (1 - tx) * (1 - ty);
		float w2 = tx  * (1 - ty);
		float w3 = (1 - tx) *      ty;
		float w4 = tx  *      ty;
		// iterate through your data
		for (int i = radius; i < src.rows - radius; i++) {
			for (int j = radius; j < src.cols - radius; j++) {
				// calculate interpolated value
				float t = static_cast<float>(w1*src.at<unsigned char>(i + fy, j + fx) + w2*src.at<unsigned char>(i + fy, j + cx) + w3*src.at<unsigned char>(i + cy, j + fx) + w4*src.at<unsigned char>(i + cy, j + cx));
				// floating point precision, so check some machine-dependent epsilon
				dst.at<unsigned char>(i - radius, j - radius) += ((t > src.at<unsigned char>(i, j)) || (std::abs(t - src.at<unsigned char>(i, j)) < std::numeric_limits<float>::epsilon())) << n;
			}
		}
	}
	//进行等价模式处理
	uchar temp = 1;
	uchar table[256] = { 0 };
	for (int i = 0; i<256; i++)
	{
		if (getHopTimes(i)<3)
		{
			table[i] = temp;
			temp++;
		}
	}
	for (int i = 0; i < dst.rows; i++)
	{
		for (int j = 0; j < dst.cols; j++)
		{
			dst.at<unsigned char>(i, j) = table[dst.at<unsigned char>(i, j)];
		}
	}
}
//计算跳变次数
int getHopTimes(int n)
{
	int count = 0;
	bitset<8> binaryCode = n;
	for (int i = 0; i<8; i++)
	{
		if (binaryCode[i] != binaryCode[(i + 1) % 8])
		{
			count++;
		}
	}
	return count;
}



//------------------------------------------------------------------------------
// Calculate the lbp histogram feature
//------------------------------------------------------------------------------

Mat histc_(const Mat& src, int minVal , int maxVal , bool normed )
{
	Mat result;
	// Establish the number of bins.
	int histSize = maxVal - minVal + 1;
	// Set the ranges.
	float range[] = { static_cast<float>(minVal), static_cast<float>(maxVal + 1) };
	const float* histRange = { range };
	// calc histogram
	calcHist(&src, 1, 0, Mat(), result, 1, &histSize, &histRange, true, false);
	// normalize
	if (normed) {
		result /= (int)src.total();
	}
	return result.reshape(1, 1);
}
//计算一张单独图片的histgram
Mat histc(InputArray _src, int minVal, int maxVal, bool normed)
{
	Mat src = _src.getMat();
	switch (src.type()) {
	case CV_8SC1:
		return histc_(Mat_<float>(src), minVal, maxVal, normed);
		break;
	case CV_8UC1:
		return histc_(src, minVal, maxVal, normed);
		break;
	case CV_16SC1:
		return histc_(Mat_<float>(src), minVal, maxVal, normed);
		break;
	case CV_16UC1:
		return histc_(src, minVal, maxVal, normed);
		break;
	case CV_32SC1:
		return histc_(Mat_<float>(src), minVal, maxVal, normed);
		break;
	case CV_32FC1:
		return histc_(src, minVal, maxVal, normed);
		break;
	}
	CV_Error(Error::StsUnmatchedFormats, "This type is not implemented yet.");
}

//计算一张图片经过gradx * grady 分块后的lbphistc
Mat spatial_histogram(InputArray _src, int numPatterns,
	int grid_x, int grid_y, bool normed)
{
	Mat src = _src.getMat();
	// calculate LBP patch size
	int width = src.cols / grid_x;
	int height = src.rows / grid_y;
	// allocate memory for the spatial histogram
	Mat result = Mat::zeros(grid_x * grid_y, numPatterns, CV_32FC1);
	// return matrix with zeros if no data was given
	if (src.empty())
		return result.reshape(1, 1);
	// initial result_row
	int resultRowIdx = 0;
	// iterate through grid
	for (int i = 0; i < grid_y; i++) {
		for (int j = 0; j < grid_x; j++) {
			Mat src_cell = Mat(src, Range(i*height, (i + 1)*height), Range(j*width, (j + 1)*width));
			Mat cell_hist = histc(src_cell, 0, (numPatterns - 1), normed);
			// copy to the result matrix
			Mat result_row = result.row(resultRowIdx);
			cell_hist.reshape(1, 1).convertTo(result_row, CV_32FC1);
			// increase row count in result matrix
			resultRowIdx++;
		}
	}
	// return result as reshaped feature vector
	return result.reshape(1, 1);
}

//------------------------------------------------------------------------------
// Calculate the lbp Verification 
//------------------------------------------------------------------------------
double chiSquare(InputArray _H1, InputArray _H2, int method)
	{
		Mat H1 = _H1.getMat(), H2 = _H2.getMat();
		const Mat* arrays[] = { &H1, &H2, 0 };
		Mat planes[2];
		NAryMatIterator it(arrays, planes);
		double result = 0;
		int j;

		double s1 = 0, s2 = 0, s11 = 0, s12 = 0, s22 = 0;

		CV_Assert(it.planes[0].isContinuous() && it.planes[1].isContinuous());

		for (size_t i = 0; i < it.nplanes; i++, ++it)
		{
			const uchar* h1 = it.planes[0].ptr<unsigned char>();
			const uchar* h2 = it.planes[1].ptr<unsigned char>();
			const int len = it.planes[0].rows*it.planes[0].cols*H1.channels();
			j = 0;

			if ((method == CV_COMP_CHISQR) || (method == CV_COMP_CHISQR_ALT))
			{
				for (; j < len; j++)
				{
					double a = h1[j] - h2[j];
					double b = (method == CV_COMP_CHISQR) ? h1[j] : h1[j] + h2[j];
					if (fabs(b) > 0)
						result += a*a / b;
				}
			}
		}
		return result;
	}
