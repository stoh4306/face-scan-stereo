#include "face_stereo.h"

#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>
#include <map>
#include <thread>
#include <mutex>

#include "calibr.h"
#include "reconst.h"
#include "mesh.h"
#include "image.h"
#include "numer_math.h"

void FaceStereo::set_correspondences_from_disparity(int level, int baseImage)
{
	cv::Mat leftImg = pyramid_[0][level];

	const int w = leftImg.cols;
	const int h = leftImg.rows;

	// Always, x1 : left image, x2 : right image
	x1_.reserve(w*h);
	x2_.reserve(w*h);
	corr_pid_.reserve(w*h);

	x1_.resize(0);
	x2_.resize(0);
	corr_pid_.resize(0);

	double k = imgScale_[level];

	int bmin[2], bmax[2];
	bmin[0] = (int)(region_min_[1] * k), bmax[0] = (int)(region_max_[1] * k); //bmin[0] = 345 * k, bmax[0] = 2033 * k;
	bmin[1] = (int)(region_min_[0] * k), bmax[1] = (int)(region_max_[0] * k);//bmin[1] = 423 * k, bmax[1] = 2307 * k;
	int dmin = (int)(minDisparity_ * k);
	int dmax = (int)(maxDisparity_ * k);

	double* disp;
	if (baseImage == 0) // left image based
		disp = disp_l_[level].data();
	else if (baseImage == 1)// right image based
		disp = disp_r_[level].data();
	else
	{
		std::cout << "ERROR, unknown base image index" << std::endl;
		return;
	}

	int pid = 0;
	if (baseImage == 0)
	{
		for (int i = 0; i < h; i += 1)
		{
			for (int j = 0; j < w; j += 1)
			{
				//if (i % 2 == 1 && j % 2 == 1)
				{
					const double d = disp[pid];

					//if (level==0 && i == 1115 && j == 1684)
					//{
					//    std::cout << "pid=(" << j << ", " << i << ") : d=" << d << "\n"
					//        << "bmin : (" << bmin[1] << ", " << bmin[0] << ")" << "\n"
					//        << "bmax : (" << bmax[1] << ", " << bmax[0] << ")" << "\n" 
					//        << "dmin = " << dmin << ", dmax=" << dmax << std::endl;
					//}

					if (i > bmin[0] && i < bmax[0] && j > bmin[1] && j < bmax[1]
						&& d > dmin && d < dmax)
					{

						x1_.push_back(Eigen::Vector2d(j, i)); // pid = j*w+i <- left image pixel
						x2_.push_back(Eigen::Vector2d(j - d, i));

						corr_pid_.push_back(Eigen::Vector2i(j, i));

						/*if ((i == 1023) && (j > 1020 && j < 1070))
						{
						std::cout << x1[pid].transpose() << "->" << x2[pid].transpose() << "\n";
						}*/


					}
				}

				pid++;
			}
		}
	}
	else if (baseImage == 1)
	{
		for (int i = 0; i < h; i += 1)
		{
			for (int j = 0; j < w; j += 1)
			{
				//if (i % 2 == 1 && j % 2 == 1)
				{
					const double d = disp[pid]; // pid = j*w+i <- pixel index of th

					if (i > bmin[0] && i < bmax[0] && j > bmin[1] && j < bmax[1]
						&& d > dmin && d < dmax)
					{
						//========================================================================
						// NOTE: Disparity should be read from the pixel index of the right image
						// since the base image is the right image.
						//========================================================================
						x1_.push_back(Eigen::Vector2d(j + d, i));
						x2_.push_back(Eigen::Vector2d(j, i));  // pid = j*w+i <- right image pixel

						corr_pid_.push_back(Eigen::Vector2i(j, i));

						/*if ((i == 1023) && (j > 1020 && j < 1070))
						{
						std::cout << x1[pid].transpose() << "->" << x2[pid].transpose() << "\n";
						}*/


					}
				}

				pid++;
			}
		}
	}

	std::cout << "# of correspondences=" << x1_.size() << std::endl;
}

void FaceStereo::convert_image_pts_normalized_coord(std::vector<Eigen::Vector2d>& x, Eigen::Matrix3d K,
	std::vector<Eigen::Vector3d>& nx)
{
	int numPts = (int)x.size();

	nx.resize(numPts);

	Eigen::Matrix3d invK = K.inverse();

	for (int i = 0; i < numPts; ++i)
	{
		Eigen::Vector3d tx(x[i](0), x[i](1), 1.0);

		nx[i] = invK * tx;
	}
}

void FaceStereo::export_disparity_map_image(const char* filename, int w, int h, double* disparity)
{
	cv::Mat dispImg(h, w, CV_8UC4);
	float* dispImgData = (float*)dispImg.data;
	std::fill(dispImgData, dispImgData + w * h, 0.0f);

	for (int i = 0; i < w*h; ++i)
	{
		if (disparity[i] > 0.0 && disparity[i] > 0.0 && disparity[i] < w)
			dispImgData[i] = (float)disparity[i];
	}

	cv::imwrite(filename, dispImg);
}

void FaceStereo::export_depth_map_image(const char* filename, int numPts, double* p, int w, int h, int* pt_pixel_id)
{
	cv::Mat depthImg(h, w, CV_16UC1);
	unsigned short* depthData = (unsigned short*)depthImg.data;

	std::fill(depthData, depthData + w * h, (unsigned short)0);

	for (int i = 0; i < numPts; ++i)
	{
		const int ti = pt_pixel_id[2 * i + 0];
		const int tj = pt_pixel_id[2 * i + 1];

		double z = p[3 * i + 2];
		if (z < 0.0) z = -z;

		depthData[tj*w + ti] = (unsigned short)(z * 1000.0f);
	}

	if (!cv::imwrite(filename, depthImg))
	{
		std::cout << "Warning, can't export the depthmap image : " << filename << std::endl;
	}
}

void FaceStereo::write_depth_map_mesh(const char* filename, int numPts, double* p, int w, int h, int* pt_pixel_id)
{
	std::vector<double> depthData(w*h, 0.0);
	std::map<int, int> pixel_to_ptid;

	for (int i = 0; i < numPts; ++i)
	{
		const int ti = pt_pixel_id[2 * i + 0];
		const int tj = pt_pixel_id[2 * i + 1];
		depthData[tj*w + ti] = p[3 * i + 2];
		pixel_to_ptid.insert(std::make_pair(tj*w + ti, i));
	}

	mg::TriMesh_f mesh;
	mesh.triangle_.reserve(w*h * 2);

	int mask[4];
	mask[0] = 0, mask[1] = 1;
	mask[2] = w, mask[3] = w + 1;

	double d[4];

	double eps = 1.0e-6;

	for (int j = 1; j < h - 1; ++j)
	{
		for (int i = 1; i < w - 1; ++i)
		{
			int pid = j * w + i;

			d[0] = depthData[pid + mask[0]];
			d[1] = depthData[pid + mask[1]];
			d[2] = depthData[pid + mask[2]];
			d[3] = depthData[pid + mask[3]];

			// triangle 1 : (0, 1, 2)
			if (d[0] > eps && d[1] > eps && d[2] > eps)
			{
				int v[3];
				v[0] = pixel_to_ptid[pid + mask[0]];
				v[1] = pixel_to_ptid[pid + mask[1]];
				v[2] = pixel_to_ptid[pid + mask[2]];

				mesh.triangle_.push_back(mg::Vector3i(v));
			}

			// triangle 2 : (1, 2, 3)
			if (d[1] > eps && d[2] > eps && d[3] > eps)
			{
				int v[3];
				v[0] = pixel_to_ptid[pid + mask[1]];
				v[1] = pixel_to_ptid[pid + mask[2]];
				v[2] = pixel_to_ptid[pid + mask[3]];

				mesh.triangle_.push_back(mg::Vector3i(v));
			}
		}
	}

	// Set mesh vertex
	mesh.vertex_.resize(numPts);
	for (int i = 0; i < numPts; ++i)
	{
		mg::Vector3d tp(p + 3 * i);
		mesh.vertex_[i] = mg::Vector3f((float)tp.x, (float)tp.y, (float)tp.z);
	}

	mesh.writeMeshInOBJ(filename, false, false);
}

void FaceStereo::reconstruction(int level, int baseImage, std::string folder)
{
	folder_ = folder;
	std::cout << "============================" << "\n"
		<< "**** 3D Reconstruction" << "\n"
		<< "============================" << std::endl;

	// Set correspondences from the disparity map
	// In any case, x1 : left, x2 : right
	set_correspondences_from_disparity(level, baseImage);

	// Correspondence points in terms of normalized coordinates
	std::vector< Eigen::Vector3d > np, nq;
	Eigen::Matrix3d K1 = K_l_[level];
	Eigen::Matrix3d K2 = K_r_[level];
	//std::cout << "K1=" << K1 << std::endl;
	//std::cout << "K2=" << K2 << std::endl;

	convert_image_pts_normalized_coord(x1_, K1, np);
	convert_image_pts_normalized_coord(x2_, K2, nq);

	// Projection matrix for normalized coordinates
	// From the current rectification, we have P1 = [I | 0] and P2 = [I | -t], 
	// where t=center of the right camera.
	Eigen::MatrixXd P1(3, 4), P2(3, 4);
	P1.block<3, 3>(0, 0).setIdentity();
	P1.block<3, 1>(0, 3).setZero();

	double b = (campose_[0].block<3, 1>(0, 3) - campose_[1].block<3, 1>(0, 3)).norm();
	//std::cout << "baseline=" << b << std::endl;
	Eigen::Vector3d t(b, 0, 0);
	P2.block<3, 3>(0, 0).setIdentity();
	P2.block<3, 1>(0, 3) = -t;

	//std::cout << "- P1(normalized) : " << "\n" << P1 << std::endl;
	//std::cout << "- P2(normalized) : " << "\n" << P2 << std::endl;
	auto npSize = np.size();
	std::vector<Eigen::Vector3d> X(npSize);
	//#pragma omp parallel for

	for (int i = 0; i < npSize; ++i)
	{
		linear_triangulation(np[i].data(), nq[i].data(), P1.data(), P2.data(), X[i].data());

		// Apply a transform = diag[1, -1, -1, 1] to obtain the result with +y as the up-direction
		X[i](1) *= -1.0;
		X[i](2) *= -1.0;
	}

	// Set texture
	// NOTE : Texture is obtained from the base image. 
	struct BGR { unsigned char b, g, r; };

	cv::Mat * baseImg = &(pyramid_[0][level]);
	Eigen::Vector2d* x = x1_.data();
	if (baseImage == 1)
	{
		baseImg = &(pyramid_[1][level]);
		x = x2_.data();
	}

	BGR* imgData = (BGR*)(baseImg->data);

	std::vector<BGR> pt_col(x1_.size());
	const int w = baseImg->cols;
	const int h = baseImg->rows;

	for (int i = 0; i < x1_.size(); ++i)
	{
		const int px = (int)x[i](0);
		const int py = (int)x[i](1);
		pt_col[i] = imgData[w*py + px];
	}

	//cv::imshow("base image", *baseImg);
	//cv::waitKey(0);

	// Export the reconstruction
	//std::string filename = std::string("reconst_h") + std::to_string(level) 
	//    + "_"+std::to_string(baseImage)+".ply";
	//write_point_cloud_in_ply(filename.c_str(), X.size(), 
	//    X.data()->data(), (unsigned char*)pt_col.data());
	std::ostringstream filename;
	if (output_format_ == 0) // pcd
	{
		//filename << folder_ << "reconst_h" << level << "_" << std::setw(paddings_) << std::setfill('0') << currentFrame_ << ".pcd";
		filename << folder_ << prefix_ << level << "_" << std::setw(paddings_) << std::setfill('0') << currentFrame_ << ".pcd";
		write_point_cloud_in_pcd(filename.str().c_str(), (int)X.size(), X.data()->data(), (unsigned char*)pt_col.data());
	}
	else // ply
	{
		//filename << folder_ << "reconst_h" << level << "_" << std::setw(paddings_) << std::setfill('0') << currentFrame_ << ".ply";
		filename << folder_ << prefix_ << level << "_" << std::setw(paddings_) << std::setfill('0') << currentFrame_ << ".ply";
		write_point_cloud_in_ply(filename.str().c_str(), (int)X.size(), X.data()->data(), (unsigned char*)pt_col.data());
	}


	// Export the depth map mesh for the point cloud
	/*std::string filename2 = std::string("depth_mesh_h") + std::to_string(level)
		+ "_" + std::to_string(baseImage) + ".obj";
	std::cout << X.size() << ", " << corr_pid_.size() << std::endl;
	write_depth_map_mesh(filename2.c_str(), X.size(), X.data()->data(), width_[level], height_[level], corr_pid_.data()->data());//*/

	// Export the depth map image
	std::ostringstream filename2;
	filename2 << folder_ << "depth_h" << level << "_" << std::setw(paddings_) << std::setfill('0') << currentFrame_ << +".png";
	//std::cout << X.size() << ", " << corr_pid_.size() << std::endl;
	export_depth_map_image(filename2.str().c_str(), (int)X.size(), X.data()->data(), width_[level], height_[level], corr_pid_.data()->data());

	// Export the disparity map image in RGBA
	std::ostringstream dispFileName;
	dispFileName << folder_ << "disp32_h" << level << "_" << std::setw(paddings_) << std::setfill('0') << currentFrame_ << +".png";
	export_disparity_map_image(dispFileName.str().c_str(), width_[level], height_[level], disp_l_[level].data());
}

void FaceStereo::reconstruction_multi_thread(int level, int baseImage, std::string folder)
{

	folder_ = folder;
	std::cout << "============================" << "\n"
		<< "**** 3D Reconstruction" << "\n"
		<< "============================" << std::endl;

	// Set correspondences from the disparity map
	// In any case, x1 : left, x2 : right
	set_correspondences_from_disparity(level, baseImage);

	// Correspondence points in terms of normalized coordinates
	std::vector< Eigen::Vector3d > np, nq;
	Eigen::Matrix3d K1 = K_l_[level];
	Eigen::Matrix3d K2 = K_r_[level];

	convert_image_pts_normalized_coord(x1_, K1, np);
	convert_image_pts_normalized_coord(x2_, K2, nq);

	Eigen::MatrixXd P1(3, 4), P2(3, 4);
	P1.block<3, 3>(0, 0).setIdentity();
	P1.block<3, 1>(0, 3).setZero();

	double b = (campose_[0].block<3, 1>(0, 3) - campose_[1].block<3, 1>(0, 3)).norm();
	Eigen::Vector3d t(b, 0, 0);
	P2.block<3, 3>(0, 0).setIdentity();
	P2.block<3, 1>(0, 3) = -t;

	auto npSize = np.size();
	std::vector<Eigen::Vector3d> X(npSize);


	int threadSize = std::thread::hardware_concurrency();
	//std::cout << "Thread Size : " << threadSize << std::endl;
	std::vector< std::thread> threads(threadSize);
	std::mutex mtx;
	int syncBlock = 0;
	for (int i = 0; i < threadSize; ++i) {
		threads[i] = std::thread([&](int tid, int threadNum) {
			int start = (int)((tid / (float)threadNum)*npSize);
			int end = (int)(((tid + 1) / (float)threadNum)*npSize);
			for (int i = start; i < end; ++i)
			{
				linear_triangulation(np[i].data(), nq[i].data(), P1.data(), P2.data(), X[i].data());
				X[i](1) *= -1.0;
				X[i](2) *= -1.0;
			}
		}
		, i, threadSize);
	}
	for (int i = 0; i < threadSize; ++i) {
		threads[i].join();
	}



	// Set texture
	// NOTE : Texture is obtained from the base image. 
	struct BGR { unsigned char b, g, r; };

	cv::Mat * baseImg = &(pyramid_[0][level]);
	Eigen::Vector2d* x = x1_.data();
	if (baseImage == 1)
	{
		baseImg = &(pyramid_[1][level]);
		x = x2_.data();
	}

	BGR* imgData = (BGR*)(baseImg->data);

	std::vector<BGR> pt_col(x1_.size());
	const int w = baseImg->cols;
	const int h = baseImg->rows;

	for (int i = 0; i < x1_.size(); ++i)
	{
		const int px = (int)x[i](0);
		const int py = (int)x[i](1);
		pt_col[i] = imgData[w*py + px];
	}


	std::ostringstream filename;
	if (output_format_ == 0) // pcd
	{
		filename << folder_ << prefix_ << level << "_" << std::setw(paddings_) << std::setfill('0') << currentFrame_ << ".pcd";
		write_point_cloud_in_pcd(filename.str().c_str(), (int)X.size(), X.data()->data(), (unsigned char*)pt_col.data());
	}
	else // ply
	{
		filename << folder_ << prefix_ << level << "_" << std::setw(paddings_) << std::setfill('0') << currentFrame_ << ".ply";
		write_point_cloud_in_ply(filename.str().c_str(), (int)X.size(), X.data()->data(), (unsigned char*)pt_col.data());
	}

	// Export the depth map image
	std::ostringstream filename2;
	filename2 << folder_ << "depth_h" << level << "_" << std::setw(paddings_) << std::setfill('0') << currentFrame_ << +".png";
	export_depth_map_image(filename2.str().c_str(), (int)X.size(), X.data()->data(), width_[level], height_[level], corr_pid_.data()->data());

	// Export the disparity map image in RGBA
	std::ostringstream dispFileName;
	dispFileName << folder_ << "disp32_h" << level << "_" << std::setw(paddings_) << std::setfill('0') << currentFrame_ << +".png";
	export_disparity_map_image(dispFileName.str().c_str(), width_[level], height_[level], disp_l_[level].data());
}




void FaceStereo::reconstruction_geom_corr(int level, int baseImage)
{
	std::cout
		<< "============================" << "\n"
		<< "**** 3D Reconstruction" << "\n"
		<< "============================" << std::endl;

	// Set correspondences from the disparity map
	// In any case, x1 : left, x2 : right
	set_correspondences_from_disparity(level, baseImage);

	// Correspondences in the original images(unrectified images)
	std::cout << "H=\n" << H_rect_ << std::endl;
	const int numCorr = (int)x1_.size();
	std::vector< Eigen::Vector2d > ox1(numCorr), ox2(numCorr);
	Eigen::Matrix3d invH = H_rect_.inverse();
	for (int i = 0; i < numCorr; ++i)
	{
		ox1[i] = x1_[i];

		Eigen::Vector3d tx(x2_[i](0), x2_[i](1), 1.0);

		Eigen::Vector3d tox2 = invH * tx;
		ox2[i](0) = tox2(0) / tox2(2);
		ox2[i](1) = tox2(1) / tox2(2);
	}

	// Find the Fundamental matrix
	Eigen::Matrix3d F, tF;
	find_fundamental_matrix_normalized_linear(numCorr, ox1.data()->data(), ox2.data()->data(), F.data());

	std::cout << "F=\n" << F << std::endl;

	tF = F.transpose();

	// First order geometric correction on ox1 and ox2
	for (int i = 0; i < numCorr; ++i)
	{
		Eigen::Vector3d tx1(ox1[i](0), ox1[i](1), 1.0);
		Eigen::Vector3d tx2(ox2[i](0), ox2[i](1), 1.0);

		Eigen::Vector3d Fx1 = F * tx1;
		Eigen::Vector3d tFx2 = tF * tx2;

		double alpha = tx2.dot(F*tx1)
			/ (Fx1(0)*Fx1(0) + Fx1(1)*Fx1(1) + tFx2(0)*tFx2(0) + tFx2(1)*tFx2(1));

		ox1[i](0) = tx1(0) - alpha * tFx2(0);
		ox1[i](1) = tx1(1) - alpha * tFx2(1);

		ox2[i](0) = tx2(0) - alpha * Fx1(0);
		ox2[i](1) = tx2(1) - alpha * Fx1(1);
	}//*/

	// Camera matrix
	Eigen::Matrix3d K_l, K_r;
	double factor = pow(0.5, (double)level);
	K_l = factor * K_[0]; K_l(2, 2) = 1.0;
	K_r = factor * K_[1]; K_r(2, 2) = 1.0;

	// Correspondence points in terms of normalized coordinates
	std::vector< Eigen::Vector3d > np, nq;
	Eigen::Matrix3d K1 = K_l;//K_l_[level];
	Eigen::Matrix3d K2 = K_r;//K_r_[level];
	std::cout << "K1=" << K1 << std::endl;
	std::cout << "K2=" << K2 << std::endl;

	convert_image_pts_normalized_coord(ox1, K1, np); //(x1_, K1, np);
	convert_image_pts_normalized_coord(ox2, K2, nq); //(x2_, K2, nq);

	// Projection matrix for normalized coordinates
	// From the current rectification, we have P1 = [I | 0] and P2 = [R | t], 
	// where R=tranpose(campose), t = -R^t*center
	Eigen::MatrixXd P1(3, 4), P2(3, 4);
	P1.block<3, 3>(0, 0).setIdentity();
	P1.block<3, 1>(0, 3).setZero();

	//double b = (campose_[0].block<3, 1>(0, 3) - campose_[1].block<3, 1>(0, 3)).norm();
	//std::cout << "baseline=" << b << std::endl;
	Eigen::Matrix3d tR = campose_[1].block<3, 3>(0, 0);
	Eigen::Vector3d c = campose_[1].block<3, 1>(0, 3);

	Eigen::Vector3d t = -tR * c;
	P2.block<3, 3>(0, 0) = tR.transpose();
	P2.block<3, 1>(0, 3) = t;

	std::cout << "- P1(normalized) : " << "\n" << P1 << std::endl;
	std::cout << "- P2(normalized) : " << "\n" << P2 << std::endl;

	std::vector<Eigen::Vector3d> X(np.size());
	for (int i = 0; i < np.size(); ++i)
	{
		linear_triangulation(np[i].data(), nq[i].data(), P1.data(), P2.data(), X[i].data());
	}

	// Set texture
	// NOTE : Texture is obtained from the base image. 
	struct BGR { unsigned char b, g, r; };

	cv::Mat * baseImg = &(pyramid_[0][level]);
	Eigen::Vector2d* x = x1_.data();
	if (baseImage == 1)
	{
		baseImg = &(pyramid_[1][level]);
		x = x2_.data();
	}

	BGR* imgData = (BGR*)(baseImg->data);

	std::vector<BGR> pt_col(x1_.size());
	const int w = baseImg->cols;
	const int h = baseImg->rows;

	for (int i = 0; i < x1_.size(); ++i)
	{
		const int px = (int)x[i](0);
		const int py = (int)x[i](1);
		pt_col[i] = imgData[w*py + px];
	}

	//cv::imshow("base image", *baseImg);
	//cv::waitKey(0);

	// Export the reconstruction
	//prefix_
	//std::string filename = std::string("reconst_h") + std::to_string(level)
	std::string filename = prefix_ + std::to_string(level)
		+ "_" + std::to_string(baseImage) + ".ply";
	write_point_cloud_in_ply(filename.c_str(), (int)X.size(),
		X.data()->data(), (unsigned char*)pt_col.data());

	// Export the depth map mesh for the point cloud
	//std::string filename2 = std::string("depth_mesh_h") + std::to_string(level)
	//    + "_" + std::to_string(baseImage) + ".obj";
	//std::cout << X.size() << ", " << corr_pid_.size() << std::endl;
	//write_depth_map_mesh(filename2.c_str(), X.size(), X.data()->data(), width_[level], height_[level], corr_pid_.data()->data());
}

void FaceStereo::unnormalized_window_vector(int w, int h, double* data, int pid, Eigen::VectorXd& f)
{
	int maskIndex[9];
	maskIndex[0] = -w - 1, maskIndex[1] = -w + 0, maskIndex[2] = -w + 1;
	maskIndex[3] = -1, maskIndex[4] = 0, maskIndex[5] = 1;
	maskIndex[6] = w - 1, maskIndex[7] = w + 0, maskIndex[8] = w + 1;

	int center = pid;

	f(0) = (double)data[center + maskIndex[0]];
	f(1) = (double)data[center + maskIndex[1]];
	f(2) = (double)data[center + maskIndex[2]];

	f(3) = (double)data[center + maskIndex[3]];
	f(4) = (double)data[center + maskIndex[4]];
	f(5) = (double)data[center + maskIndex[5]];

	f(6) = (double)data[center + maskIndex[6]];
	f(7) = (double)data[center + maskIndex[7]];
	f(8) = (double)data[center + maskIndex[8]];

	Eigen::VectorXd temp(9);
	temp.setConstant(1.0);
	double mean_f = f.dot(temp) / 9;

	f = f - mean_f * temp;
	if (f.norm() > 0)
		f.normalize();
}

void FaceStereo::normalized_intensity_vector(int w, int h, unsigned char* imgData, int pid, Eigen::VectorXd& nf)
{
	//if (ws == 3)
	{
		int maskIndex[9];
		maskIndex[0] = -w - 1, maskIndex[1] = -w + 0, maskIndex[2] = -w + 1;
		maskIndex[3] = -1, maskIndex[4] = 0, maskIndex[5] = 1;
		maskIndex[6] = w - 1, maskIndex[7] = w + 0, maskIndex[8] = w + 1;

		int center = pid;

		static Eigen::VectorXd f(9);
		f(0) = (double)imgData[center + maskIndex[0]];
		f(1) = (double)imgData[center + maskIndex[1]];
		f(2) = (double)imgData[center + maskIndex[2]];

		f(3) = (double)imgData[center + maskIndex[3]];
		f(4) = (double)imgData[center + maskIndex[4]];
		f(5) = (double)imgData[center + maskIndex[5]];

		f(6) = (double)imgData[center + maskIndex[6]];
		f(7) = (double)imgData[center + maskIndex[7]];
		f(8) = (double)imgData[center + maskIndex[8]];

		static Eigen::VectorXd temp(9);
		temp.setConstant(1.0);
		double mean_f = f.dot(temp) / 9;

		nf = f - mean_f * temp;
		auto nfn = nf.norm();
		if (nfn > 0) {
			nf /= nfn;
			//nf.normalize();
		}

	}
}


//inline void FaceStereo::normalized_intensity_vector_block0(int w, int h, unsigned char * imgData, int pid, Eigen::VectorXd * ng, Eigen::VectorXd * f, Eigen::VectorXd * temp, Eigen::VectorXd* nf,double* eta)
//{
//	normalized_intensity_vector_block(w, h, imgData, pid, ng, f, temp);
//	eta[0] = 0.5*(1.0 - ncc(*nf, *ng));
//}
//
//inline void FaceStereo::normalized_intensity_vector_block1(int w, int h, unsigned char * imgData, int pid, Eigen::VectorXd * ng, Eigen::VectorXd * f, Eigen::VectorXd * temp, Eigen::VectorXd* nf, double* eta)
//{
//	normalized_intensity_vector_block(w, h, imgData, pid + 1, ng, f, temp);
//	eta[0] = 0.5*(1.0 - ncc(*nf, *ng));
//}
//
//inline void FaceStereo::normalized_intensity_vector_block2(int w, int h, unsigned char * imgData, int pid, Eigen::VectorXd * ng, Eigen::VectorXd * f, Eigen::VectorXd * temp, Eigen::VectorXd* nf, double* eta)
//{
//	normalized_intensity_vector_block(w, h, imgData, pid - 1, ng, f, temp);
//	eta[0] = 0.5*(1.0 - ncc(*nf, *ng));
//}

inline void FaceStereo::normalized_intensity_vector_block(int w, int h, unsigned char* imgData, int pid, Eigen::VectorXd* nf, Eigen::VectorXd* f, Eigen::VectorXd* temp)
{
	//if (ws == 3)
	{
		int maskIndex[9];
		maskIndex[0] = -w - 1, maskIndex[1] = -w + 0, maskIndex[2] = -w + 1;
		maskIndex[3] = -1, maskIndex[4] = 0, maskIndex[5] = 1;
		maskIndex[6] = w - 1, maskIndex[7] = w + 0, maskIndex[8] = w + 1;

		int center = pid;

		(*f)(0) = (double)imgData[center + maskIndex[0]];
		(*f)(1) = (double)imgData[center + maskIndex[1]];
		(*f)(2) = (double)imgData[center + maskIndex[2]];

		(*f)(3) = (double)imgData[center + maskIndex[3]];
		(*f)(4) = (double)imgData[center + maskIndex[4]];
		(*f)(5) = (double)imgData[center + maskIndex[5]];

		(*f)(6) = (double)imgData[center + maskIndex[6]];
		(*f)(7) = (double)imgData[center + maskIndex[7]];
		(*f)(8) = (double)imgData[center + maskIndex[8]];

		temp->setConstant(1.0);
		double mean_f = f->dot((*temp)) / 9;

		(*nf) = (*f) - mean_f * (*temp);
		auto nfn = nf->norm();
		if (nfn > 0) {
			(*nf) /= nfn;
			//nf.normalize();
		}

	}
}
double FaceStereo::ncc(const Eigen::VectorXd& nf1, const Eigen::VectorXd& nf2)
{
	return nf1.dot(nf2);
}

//void FaceStereo::extend_raw_disparity_from_the_other(int level)
//{
//	// ASSUME : The current image is from the left.
//	int w = width_[level];
//	int h = height_[level];
//
//	// For test only
//	std::fill(disp_l_[level].begin(), disp_l_[level].end(), 0.0);
//
//
//	double* curr_disp = disp_l_[level].data();
//	double* ref_disp = disp_r_[level].data();
//
//	//auto result = std::minmax_element(disp_l_[level].begin(), disp_l_[level].end());
//	//std::cout << "- Disparity range : " << *result.first << "-" << *result.second << std::endl;
//
//	// 
//	// 1: Simplest method
//	for (int j = 0; j < h; ++j)
//	{
//		for (int i = 0; i < w; ++i)
//		{
//			double rd = ref_disp[j*w + i];
//			int left_i = (int)(i + rd);
//			if (rd > 0.0 && left_i >= 0 && left_i < w && curr_disp[j*w+left_i] <= 0.0)
//			{
//				curr_disp[j*w + left_i] = rd;
//			}
//		}
//	}//*/
//
//	// 2:
//	//std::vector<double> row_disp(w);
//
//	//for (int j = 0; j < h; ++j)
//	//{
//	//	for (int i = w - 2; i >= 1; --i)
//	//	{
//	//		if (curr_disp[j*w + i] <= 0.0 && curr_disp[j*w + i + 1] > 0.0)
//	//		{
//	//			double left_disp = curr_disp[j*w + i + 1];
//	//			double right_disp = ref_disp[j * w + (int)(i - left_disp)];
//	//			if (right_disp > 0.0)
//	//			{
//	//				curr_disp[j*w + i] = right_disp;
//	//			}
//	//		}
//	//	}
//
//	//	for (int i = w-1; i >= 0; --i)
//	//	{
//	//		double rd = ref_disp[j*w + i];
//	//		int left_i = (int)(i + rd)+1;
//	//		if (rd > 0.0 && left_i >= 0 && left_i < w && curr_disp[j*w + left_i] <= 0.0)
//	//		{
//	//			curr_disp[j*w + left_i] = rd;
//	//		}
//
//	//		left_i -= 1;
//	//		if (rd > 0.0 && left_i >= 0 && left_i < w && curr_disp[j*w + left_i] <= 0.0)
//	//		{
//	//			curr_disp[j*w + left_i] = rd;
//	//		}
//	//	}
//
//	//	memcpy(row_disp.data(), &curr_disp[j*w + 0], sizeof(double)*w);
//
//	//	// For hole filling(oxo)
//	//	int count = 0;
//	//	for (int i = 1; i < w - 1; ++i)
//	//	{
//	//		if (row_disp[i] <= 0.0 && row_disp[i - 1] > 0.0 && row_disp[i + 1] > 0.0)
//	//		{
//	//			curr_disp[j*w + i] = 0.5 * (row_disp[i - 1] + row_disp[i + 1]);
//	//			count++;
//	//		}
//	//	}
//
//	//	if (j == 64)
//	//	{
//	//		std::cout << "- holo-filleing count = " << count << std::endl;
//	//	}
//
//	//}
//}

void FaceStereo::set_raw_disparity_from_the_other(int level, int baseImage)
{
	int w = width_[level];
	int h = height_[level];

	// For test only
	double* disp = disp_l_[level].data();
	double* ref_disp = disp_r_[level].data();

	if (baseImage == 1)
	{
		std::swap(disp, ref_disp);
	}

	std::fill(disp, disp + w * h, 0.0);

	if (baseImage == 0)
	{
		for (int j = 0; j < h; ++j)
		{
			for (int i = 0; i < w; ++i)
			{
				double rd = ref_disp[j*w + i];
				int base_i = (int)(i + rd);
				if (rd > 0.0 && base_i >= 0 && base_i < w && disp[j*w + base_i] <= 0.0)
				{
					disp[j*w + base_i] = rd;
				}
			}
		}
	}
	else
	{
		for (int j = 0; j < h; ++j)
		{
			for (int i = 0; i < w; ++i)
			{
				double rd = ref_disp[j*w + i];
				int base_i = (int)(i - rd);
				if (rd > 0.0 && base_i >= 0 && base_i < w && disp[j*w + base_i] <= 0.0)
				{
					disp[j*w + base_i] = rd;
				}
			}
		}
	}
}

void FaceStereo::compute_raw_disparity_block_matching(int level, int baseImage)
{
	// NOTE : There is no down-scaled images to refer. 
	// Compute disparities from block matching algorithm
	cv::Mat baseImg, secondImg;
	pyramid_[0][level].copyTo(baseImg);
	pyramid_[1][level].copyTo(secondImg);

	//if (baseImage == 1)   std::swap(baseImg, secondImg);
	if (baseImage == 1)
	{
		cv::Mat tempImg;
		baseImg.copyTo(tempImg);
		secondImg.copyTo(baseImg);
		tempImg.copyTo(secondImg);

		cv::flip(baseImg, baseImg, 1);
		cv::flip(secondImg, secondImg, 1);
	}

	int w = baseImg.cols;
	int h = baseImg.rows;

	double scale_factor = imgScale_[level];

	if (baseImage == 0)
		disp_l_[level].resize(w*h);
	else
		disp_r_[level].resize(w*h);

	double* disp = disp_l_[level].data();
	if (baseImage == 1)   disp = disp_r_[level].data();

	int blockSize = 3;
	int maxDisparity = (int)(maxDisparity_ * scale_factor);
	stereo_matching(w, h, baseImg.type(), baseImg.data, secondImg.data,
		blockSize, maxDisparity, disp, "disparity_map.txt");

	/*if (baseImage == 0)
	{
		std::ofstream outFile("disparity_map_h" + std::to_string(level) + ".txt");
		for (int j = 0; j < h; ++j)
		{
			for (int i = 0; i < w; ++i)
			{
				outFile << disp[j*w + i] << " ";
			}
			outFile << "\n";
		}
		outFile.close();
	}*/

	// If the base image is the right image, flip the disparity matrix.
	// One should be careful when the correspondences are set later.
	if (baseImage == 1)
	{
		// flip data
		for (int j = 0; j < h; ++j)
		{
			for (int i = 0; i < w / 2; ++i)
			{
				std::swap(disp[j*w + i], disp[j*w + (w - 1 - i)]);
			}
		}
	}
}


void FaceStereo::stereo_matching_2(int w, int h, int type, unsigned char* baseImgData,
	unsigned char* secondImgData, int blockSize, int minDisparity, int maxDisparity,
	double* disp, const char* disp_file_name)
{
	std::cout << "max disparity=" << maxDisparity << std::endl;

	for (int j = 0; j < h; ++j)
	{
		for (int i = 0; i < w; ++i)
		{
			int pid = j * w + i;

			Eigen::VectorXd nf(9);
			normalized_intensity_vector(w, h, baseImgData, pid, nf);

			int k0 = i - maxDisparity; if (k0 < 0)  k0 = 0;
			int k1 = i - minDisparity; if (k1 < 0)  k1 = 0;

			std::vector< double > ncc(k1 - k0 + 1);

			double max_ncc = -1.0;
			int max_ncc_index = 0;

			for (int k = k0; k <= k1; ++k)
			{
				Eigen::VectorXd ng(9);
				int qid = j * w + k;
				normalized_intensity_vector(w, h, secondImgData, qid, ng);

				double tempNcc = nf.dot(ng);
				ncc[k - k0] = tempNcc;
				//std::cout << tempNcc << " ";

				if (tempNcc > max_ncc)
				{
					max_ncc = tempNcc;
					max_ncc_index = k - k0;
				}
			}// end for(k)
			//std::cout << std::endl;

			//disp[pid] = i - (k0 + max_ncc_index);
			//std::cout << "pid=" << pid << ", (" << i << "," << j << ") : d=" << disp[pid] << " ("
			//    << ncc[max_ncc_index] << ")" << std::endl;

			if (k1 - k0 >= 3 && max_ncc_index > 0 && max_ncc_index < k1 - k0)
			{
				double f[3];
				f[0] = max_ncc;
				f[1] = ncc[max_ncc_index + 1];
				f[2] = ncc[max_ncc_index - 1];

				const double a = 0.5*(f[1] + f[2]) - f[0];
				const double b = 0.5*(f[1] - f[2]);

				const double dx = -0.5*b / a;

				if (dx >= -1.0 && dx <= 1.0)
					disp[pid] = i - (k0 + max_ncc_index + dx);
				else
					disp[pid] = i - (k0 + max_ncc_index);

				//if (f[0] <= f[1] || f[0] <= f[2])
				//{
				//    std::cout << f[0] << " " << f[1] << " " << f[2]
				//        << " : " << -0.5*b / a << std::endl;
				//}


			}
			else
			{
				disp[pid] = i - (k0 + max_ncc_index);
			}

		}// end for(i)
	}// end for(j)
}

void FaceStereo::compute_raw_disparity_block_matching_2(int level, int baseImage)
{
	// NOTE : There is no down-scaled images to refer. 
	// Compute disparities from block matching algorithm
	cv::Mat baseImg, secondImg;
	pyramid_gray_[0][level].copyTo(baseImg);
	pyramid_gray_[1][level].copyTo(secondImg);

	//if (baseImage == 1)   std::swap(baseImg, secondImg);
	if (baseImage == 1)
	{
		cv::Mat tempImg;
		baseImg.copyTo(tempImg);
		secondImg.copyTo(baseImg);
		tempImg.copyTo(secondImg);

		cv::flip(baseImg, baseImg, 1);
		cv::flip(secondImg, secondImg, 1);
	}

	int w = baseImg.cols;
	int h = baseImg.rows;

	double scale_factor = imgScale_[level];

	if (baseImage == 0)
		disp_l_[level].resize(w*h);
	else
		disp_r_[level].resize(w*h);

	double* disp = disp_l_[level].data();
	if (baseImage == 1)   disp = disp_r_[level].data();

	int blockSize = 3;
	int maxDisparity = (int)(maxDisparity_ * scale_factor);
	//stereo_matching(w, h, baseImg.type(), baseImg.data, secondImg.data,
	//    blockSize, maxDisparity, disp, "disparity_map.txt");

	int minDisparity = (int)(minDisparity_ * scale_factor);
	stereo_matching_2(w, h, baseImg.type(), baseImg.data, secondImg.data,
		blockSize, minDisparity, maxDisparity, disp, "disparity_map.txt");

	// If the base image is the right image, flip the disparity matrix.
	// One should be careful when the correspondences are set later.
	if (baseImage == 1)
	{
		// flip data
		for (int j = 0; j < h; ++j)
		{
			for (int i = 0; i < w / 2; ++i)
			{
				std::swap(disp[j*w + i], disp[j*w + (w - 1 - i)]);
			}
		}
	}
}

void FaceStereo::check_disparity_smoothness(int level, std::vector<int>& index)
{
	double* disp = disp_l_[level].data();

	const int w = width_[level];
	const int h = height_[level];
	index.reserve(w*h);
	index.resize(0);

	double min_disp = disp_min_[level];
	double max_disp = disp_max_[level];

	int maskIndex[9];
	maskIndex[0] = -w - 1, maskIndex[1] = -w, maskIndex[2] = -w + 1;
	maskIndex[3] = -1, maskIndex[4] = 0, maskIndex[5] = 1;
	maskIndex[6] = w - 1, maskIndex[7] = w, maskIndex[8] = w + 1;

	int numNonSmoothPoints = 0;
	for (int j = 1; j < h - 1; ++j)
	{
		for (int i = 1; i < w - 1; ++i)
		{
			int pid = j * w + i;

			double d_c = disp[pid];
			if (d_c > min_disp && d_c < max_disp)
			{

				int count = 0;

				for (int k = 0; k < 9; ++k)
				{
					int wpid = pid + maskIndex[k];

					if (fabs(disp[wpid] - d_c) < 1.0)    count++;
				}

				if (count < 5)
				{
					//disp[pid] = 0.0;
					index.push_back(pid);
					numNonSmoothPoints++;
				}
			}
			else
				index.push_back(pid);
		}
	}

	std::cout << "#of non-smooth points = " << numNonSmoothPoints << std::endl;
}

void FaceStereo::check_disparity_uniqueness(int level, std::vector<int>& index)
{
	const int w = width_[level];
	const int h = height_[level];

	index.reserve(w*h);
	index.resize(0);

	double* left_disparity = disp_l_[level].data();
	double* right_disparity = disp_r_[level].data();

	double min_disp = disp_min_[level];
	double max_disp = disp_max_[level];

	int count = 0;
	for (int j = 0; j < h; ++j)
	{
		for (int i = 0; i < w; ++i)
		{
			int pid[2];
			double d[2];

			pid[0] = j * w + i; // pixel id in the left image
			d[0] = left_disparity[pid[0]];

			if (d[0] > min_disp && d[0] < max_disp)
			{
				if (i - d[0] > 1 && i - d[0] < w - 1)
				{
					pid[1] = (int)(floor)(pid[0] - d[0] + 0.5);

					d[1] = right_disparity[pid[1]];

					if (abs(d[0] - d[1]) > 1.0)
					{
						index.push_back(pid[0]);
						count++;
					}
				}
			}
			else
			{
				index.push_back(pid[0]);
				count++;
			}
		}
	}

	std::cout << ". # of non-unique points = " << count << std::endl;
}

void FaceStereo::check_disparity_ordering(int level, std::vector<int>& index)
{
	const int w = width_[level];
	const int h = height_[level];

	double min_disp = disp_min_[level];
	double max_disp = disp_max_[level];

	index.reserve(w*h);
	index.resize(0);

	double* disp = disp_l_[level].data();

	for (int j = 0; j < h; ++j)
	{
		for (int i = 0; i < w - 1; ++i)
		{
			int pid = j * w + i;
			if (disp[pid] > min_disp && disp[pid] < max_disp &&
				disp[pid + 1] > min_disp && disp[pid + 1] < max_disp &&
				disp[pid] - disp[pid + 1] < -1.0) //disp[pid] - disp[pid + 1] > 1.0)
			{
				index.push_back(pid);
			}
		}
	}
	std::cout << ". # of disorder points = " << index.size() << std::endl;
	std::cout << "End of check_disparity_ordering" << std::endl;
	return;
}

void FaceStereo::expand_lower_level_disparity(int level, double * disp_expanded)
{
	const int w = width_[level];
	const int h = height_[level];

	// initialize
	std::fill(disp_expanded, disp_expanded + w * h, 0.0);

	const int tw = width_[level + 1];
	const int th = height_[level + 1];

	double* d_x = disp_l_[level + 1].data();

	const int min_disp_x = disp_min_[level + 1];
	const int max_disp_x = disp_max_[level + 1];

	std::vector<double> weight(w*h, 0.0);

	int mask[9];
	mask[0] = 0;
	mask[1] = +1;
	mask[2] = -1;
	mask[3] = -w;
	mask[4] = -w + 1;
	mask[5] = -w - 1;
	mask[6] = w;
	mask[7] = w + 1;
	mask[8] = w - 1;

	// Set first by simple projection
	for (int tj = 1; tj < th - 1; ++tj)
	{
		const int j = 2 * tj;

		for (int ti = 1; ti < tw - 1; ++ti)
		{
			const int i = 2 * ti;

			const int pid = j * w + i;
			const int pid_x = tj * tw + ti;

			const double disp_x = 1.0*d_x[pid_x];

			if (disp_x > min_disp_x && disp_x < max_disp_x)
			{
				disp_expanded[pid + mask[0]] += 2.0* disp_x;
				disp_expanded[pid + mask[1]] += 2.0* disp_x;
				disp_expanded[pid + mask[2]] += 2.0* disp_x;

				disp_expanded[pid + mask[3]] += 2.0* disp_x;
				disp_expanded[pid + mask[4]] += 2.0* disp_x;
				disp_expanded[pid + mask[5]] += 2.0* disp_x;

				disp_expanded[pid + mask[6]] += 2.0* disp_x;
				disp_expanded[pid + mask[7]] += 2.0* disp_x;
				disp_expanded[pid + mask[8]] += 2.0* disp_x;


				weight[pid + mask[0]] += 1.0;
				weight[pid + mask[1]] += 1.0;
				weight[pid + mask[2]] += 1.0;
				weight[pid + mask[3]] += 1.0;
				weight[pid + mask[4]] += 1.0;
				weight[pid + mask[5]] += 1.0;
				weight[pid + mask[6]] += 1.0;
				weight[pid + mask[7]] += 1.0;
				weight[pid + mask[8]] += 1.0;

			}
		}
	}

	// Apply weights properly
	for (int j = 0; j < h; ++j)
	{
		for (int i = 0; i < w; ++i)
		{
			const int pid = j * w + i;

			if (weight[pid] > 0.0)
				disp_expanded[pid] /= weight[pid];
		}
	}

	//
	/*cv::Mat img(h, w, CV_64FC1, disp_expanded);
	cv::imshow("Disparity expanded", img);
	cv::waitKey(0);*/
}

double compute_curvature(int pid, int* mask, double* disp, double f, double px, double py, double cx, double cy)
{
	// ASSUME : n=9
	std::vector< Eigen::Vector3d > p(9);

	//double tx = f / sqrt(f*f + (py - cy)*(py - cy));
	//double ty = tx;
	Eigen::Vector3d center(0.0, 0.0, 0.0);

	for (int j = 0; j < 3; ++j)
	{
		double ty = py - j - cy;
		double k = f / sqrt(f*f + ty * ty);

		for (int i = 0; i < 3; ++i)
		{
			int nid = pid + mask[j * 3 + i];

			double y = k * ty / disp[nid];
			double x = k * (px - i - cx) / disp[nid];

			double z = k * f / disp[nid];

			p[j * 3 + i] = Eigen::Vector3d(x, y, z);

			center += p[j * 3 + i];
		}
	}

	center /= 9.0;

	Eigen::Matrix3d Cov;
	compute_covariance_matrix(9, p.data()->data(), center.data(), Cov.data());

	// Find all eigenvalues of the covariance matrix
	Eigen::MatrixXd A = Cov;

	Eigen::JacobiSVD<Eigen::MatrixXd> svd;
	svd = Eigen::JacobiSVD<Eigen::MatrixXd>(A, Eigen::ComputeFullU | Eigen::ComputeFullV);

	Eigen::VectorXd sv = svd.singularValues();
	//std::cout << svd.singularValues() << std::endl;

	/*if (pid == 73993)
	{
		std::cout << "neighbor pts of pid=" << pid << std::endl;
		for (int i = 0; i < 9; ++i)
		{
			std::cout << i << " : (" << p[i].transpose() << ")" << "\n";
		}
		std::cout << "cov=\n" << A << std::endl;
		std::cout << "eigenvalue=" << sv.transpose() << std::endl;
	}*/



	return sv[0] / (sv[0] + sv[1] + sv[2]);
}

void FaceStereo::hierarchical_block_matching(int level, double* disp_expanded)
{
	const int w = width_[level];
	const int h = height_[level];

	double* d0 = disp_expanded;
	double* d1 = disp_l_[level].data();

	temp_disp_l_[level].resize(w*h);
	double* d2 = temp_disp_l_[level].data();

	unsigned char* leftImgData = pyramid_gray_[0][level].data;
	unsigned char* rightImgData = pyramid_gray_[1][level].data;
	double min_disp = disp_min_[level];
	double max_disp = disp_max_[level];

	// First, reset the diparity from the previous layer for the pixels with negative disparity
	// if the disparity from the previous layer is available.
	for (int j = 0; j < h; ++j)
	{
		for (int i = 0; i < w; ++i)
		{
			const int pid = j * w + i;

			if ((d1[pid] <= min_disp || d1[pid] >= max_disp)
				&& (d0[pid] > min_disp && d0[pid] < max_disp))
				d1[pid] = d0[pid];
		}
	}

	double f = 0.5*(K_l_[level](0, 0) + K_l_[level](1, 1));
	std::cout << "focus=" << f << std::endl;
	std::cout << "distance : " << f / max_disp << "~" << f / min_disp << std::endl;
	std::cout << "max depth change ratio : " << max_depth_change_ratio_ << std::endl;
	std::cout << "max curv change ratio : " << max_curv_change_ratio_ << std::endl;

	int ipx = (int)(2312 / pow(2.0, level));
	int ipy = (int)(1188 / pow(2.0, level));
	//std::cout << "interest pixel(level=" << level << ") : [" << ipx << ", " << ipy << "]" << std::endl;

	int mask[9];
	mask[0] = -w - 1, mask[1] = -w, mask[2] = -w + 1;
	mask[3] = -1, mask[4] = 0, mask[5] = 1;
	mask[6] = w - 1, mask[7] = w, mask[8] = w + 1;

	double k_ratio_max = max_curv_change_ratio_;
	double k_ratio_min = 1.0 / max_curv_change_ratio_;

	for (int j = 1; j < h - 1; ++j)
	{
		for (int i = 1; i < w - 1; ++i)
		{
			const int pid = j * w + i;
			if (d0[pid] > min_disp && d0[pid] < max_disp)
			{
				// Compute the curvatures for the current and the previous layers
				double curv[2];
				const int num_neighbors = 9;
				curv[0] = compute_curvature(pid, mask, d0, f, i, j, w / 2, h / 2);
				curv[1] = compute_curvature(pid, mask, d1, f, i, j, w / 2, h / 2);

				double k_ratio = abs(curv[1] / curv[0]);

				d2[pid] = d0[pid];

				//if( d1[pid] > min_disp && d1[pid] < max_disp 
				//    && abs(1.0 - d0[pid]/ d1[pid])<max_depth_change_ratio_) // dz/z < 0.02
				if (d1[pid] > min_disp && d1[pid] < max_disp
					&& abs(1.0 - d0[pid] / d1[pid]) < max_depth_change_ratio_ // dz/z < 0.02
					&& k_ratio < k_ratio_max && k_ratio > k_ratio_min)
				{
					d2[pid] = d1[pid];
				}

				/*if (i == ipx && j == ipy)
				{
					std::cout << "depth change between layers : " << abs(1.0 - d0[pid] / d1[pid])
						<< ", d0=" << d0[pid] << ", d1=" << d1[pid]
						<< ", z0 = " << f / d0[pid] << ", z1=" << f / d1[pid]
						<< ", d2=" << d2[pid] << "\n"
						<< "real depth = " << f/d2[pid]*f/sqrt(f*f+(i-0.5*w)*(i-0.5*w)) << "\n"
						<< "k0 = " << curv[0] << ", k1=" << curv[1] << std::endl;

					std::cout << "neighbor disparity(d0) : ";
					for (int k = 0; k < 9; ++k)
					{
						std::cout << d0[pid + mask[k]] << "(" << pid+mask[k] << ") ";
					}
					std::cout << std::endl;

					std::cout << "neighbor disparity(d1) : ";
					for (int k = 0; k < 9; ++k)
					{
						std::cout << d1[pid + mask[k]] << "(" << pid + mask[k] << ") ";
					}
					std::cout << std::endl;
				}*/

				//else
				//{
				//    d2[pid] = d0[pid];
				//}
				//else
				//{
				//    Eigen::VectorXd nf(9);
				//    normalized_intensity_vector(w, h, leftImgData, pid, nf);

				//    const int q = (int)(i - d0[pid] + 0.5);
				//    int qid = j*w + q;
				//    Eigen::VectorXd ng0(9), ng1(9), ng2(9);
				//    normalized_intensity_vector(w, h, rightImgData, qid    , ng0);
				//    normalized_intensity_vector(w, h, rightImgData, qid + 1, ng1);
				//    normalized_intensity_vector(w, h, rightImgData, qid - 1, ng2);

				//    double ncc[3];
				//    ncc[0] = nf.dot(ng0);
				//    ncc[1] = nf.dot(ng1);
				//    ncc[2] = nf.dot(ng2);

				//    if (ncc[0] >= ncc[1] && ncc[0] >= ncc[2])
				//    {
				//        if (abs(ncc[0]-ncc[1]) > 1.0e-6 || abs(ncc[0] - ncc[2]) > 1.0e-6)
				//        {
				//            // TODO : dq = 0.5*(ncc[-1]-ncc[1])/(ncc[1]+ncc[-1]-2.0*ncc[0]);
				//            // disp[pid] = q-dq
				//            double dq = 0.5*(ncc[2] - ncc[1]) / (ncc[1] + ncc[2] - 2.0*ncc[0]);
				//            d2[pid] = q - dq;
				//        }
				//        else
				//        {
				//            d2[pid] = q;
				//        }
				//    }
				//    else if (ncc[1] >= ncc[0] && ncc[1] >= ncc[2])
				//    {
				//        d2[pid] = q - 0.5;
				//    }
				//    else //if (ncc[2] <= ncc[0] && ncc[1] <= ncc[1])
				//    {
				//        d2[pid] = q + 0.5;
				//    }

				//    if (d2[pid] == std::numeric_limits<double>::infinity() ||
				//        d2[pid] == -std::numeric_limits<double>::infinity())
				//    {
				//        std::cout << "pid=" << pid << ", d0=" << d0[pid] 
				//            << ", d1=" << d1[pid] << ", d2=" << d2[pid]  
				//            << ", ncc=" << ncc[0] << " " << ncc[1] << " " << ncc[2] << std::endl;
				//    }
				//}

				//d2[pid] = d0[pid];
			}
			else
				d2[pid] = -1;


		}// for(i)
	}// for(j)

	/*if (level == 3)
	{
		std::cout << "disparity[74490] =" << d2[74490] << std::endl;
	}*/

	memcpy(d1, d2, sizeof(double)*w*h);
}

void FaceStereo::hierarchical_block_matching_multi_thread(int level, double * disp_expanded)
{

	const int w = width_[level];
	const int h = height_[level];

	temp_disp_l_[level].resize(w*h);
	double* d1 = disp_l_[level].data();

	double* d2 = temp_disp_l_[level].data();
	int threadSize = std::thread::hardware_concurrency();
	std::cout << "Thread Size : " << threadSize << std::endl;
	std::vector< std::thread> threads(threadSize);
	std::mutex mtx;
	int syncBlock = 0;
	for (int i = 0; i < threadSize; ++i) {
		threads[i] = std::thread(&FaceStereo::hierarchical_block_matching_block, this, level, disp_expanded, &syncBlock, i, threadSize, &mtx);
	}
	for (int i = 0; i < threadSize; ++i) {
		threads[i].join();
	}

	// First, reset the diparity from the previous layer for the pixels with negative disparity
	// if the disparity from the previous layer is available.


	memcpy(d1, d2, sizeof(double)*w*h);
}

void FaceStereo::hierarchical_block_matching_block(int level, double * disp_expanded, int * syncBlock, int tid, int threadNum, std::mutex* mtx)
{
	int sum = 0;
	for (int i = 1; i <= threadNum; ++i) {
		sum += i;
	}
	const int w = width_[level];
	const int h = height_[level];

	double* d0 = disp_expanded;
	double* d1 = disp_l_[level].data();

	double* d2 = temp_disp_l_[level].data();

	unsigned char* leftImgData = pyramid_gray_[0][level].data;
	unsigned char* rightImgData = pyramid_gray_[1][level].data;
	double min_disp = disp_min_[level];
	double max_disp = disp_max_[level];

	int start = (int)((tid / (float)threadNum)*h);
	int end = (int)(((tid + 1) / (float)threadNum)*h);

	for (int j = start; j < end; ++j)
	{
		for (int i = 0; i < w; ++i)
		{
			const int pid = j * w + i;

			if ((d1[pid] <= min_disp || d1[pid] >= max_disp)
				&& (d0[pid] > min_disp && d0[pid] < max_disp))
				d1[pid] = d0[pid];
		}
	}
	mtx->lock();
	*syncBlock += (tid + 1);
	mtx->unlock();
	while (*syncBlock < sum) {
		std::this_thread::yield();
	}

	double f = 0.5*(K_l_[level](0, 0) + K_l_[level](1, 1));
	if (tid == 0) {
		std::cout << "focus=" << f << std::endl;
		std::cout << "distance : " << f / max_disp << "~" << f / min_disp << std::endl;
		std::cout << "max depth change ratio : " << max_depth_change_ratio_ << std::endl;
		std::cout << "max curv change ratio : " << max_curv_change_ratio_ << std::endl;
	}

	int ipx = (int)(2312 / pow(2.0, level));
	int ipy = (int)(1188 / pow(2.0, level));
	//std::cout << "interest pixel(level=" << level << ") : [" << ipx << ", " << ipy << "]" << std::endl;

	int mask[9];
	mask[0] = -w - 1, mask[1] = -w, mask[2] = -w + 1;
	mask[3] = -1, mask[4] = 0, mask[5] = 1;
	mask[6] = w - 1, mask[7] = w, mask[8] = w + 1;

	double k_ratio_max = max_curv_change_ratio_;
	double k_ratio_min = 1.0 / max_curv_change_ratio_;
	if (start == 0)
		start = 1;
	if (end == h)
		end = h - 1;
	for (int j = start; j < end; ++j)
	{
		for (int i = 1; i < w - 1; ++i)
		{
			const int pid = j * w + i;
			if (d0[pid] > min_disp && d0[pid] < max_disp)
			{
				// Compute the curvatures for the current and the previous layers
				double curv[2];
				const int num_neighbors = 9;
				curv[0] = compute_curvature(pid, mask, d0, f, i, j, w / 2, h / 2);
				curv[1] = compute_curvature(pid, mask, d1, f, i, j, w / 2, h / 2);

				double k_ratio = abs(curv[1] / curv[0]);

				d2[pid] = d0[pid];

				if (d1[pid] > min_disp && d1[pid] < max_disp
					&& abs(1.0 - d0[pid] / d1[pid]) < max_depth_change_ratio_ // dz/z < 0.02
					&& k_ratio < k_ratio_max && k_ratio > k_ratio_min)
				{
					d2[pid] = d1[pid];
				}
			}
			else
				d2[pid] = -1;


		}// for(i)
	}// for(j)

}



void FaceStereo::rematch(int level)
{
	// Rematch for pixels with good matched neighbors
	const int w = width_[level];
	const int h = height_[level];

	double* disp = disp_l_[level].data();

	double min_disp = disp_min_[level];
	double max_disp = disp_max_[level];

	unsigned char* leftImgData = pyramid_gray_[0][level].data;
	unsigned char* rightImgData = pyramid_gray_[1][level].data;

	int count = 0;
	for (int j = 1; j < h - 1; ++j)
	{
		for (int i = 1; i < w - 1; ++i)
		{
			const int pid = j * w + i;

			if (disp[pid] < -10) // pixel to be rematched
			{
				Eigen::VectorXd nf(9);
				normalized_intensity_vector(w, h, leftImgData, pid, nf);

				double temp_disp;
				// Case 1 : horizontal neighbors
				if (disp[pid - 1] > min_disp && disp[pid - 1] < max_disp &&
					disp[pid + 1] > min_disp && disp[pid + 1] < max_disp)
				{
					temp_disp = 0.5*(disp[pid - 1] + disp[pid + 1]);
				}
				/*else if (disp[pid - w] > min_disp && disp[pid - w] < max_disp &&
					disp[pid + w] > min_disp && disp[pid + w] < max_disp)
				{
					temp_disp = 0.5*(disp[pid - w] + disp[pid + w]);
				}*/
				else
					continue;

				int q = (int)(temp_disp + 0.5);

				int qid = j * w + q;
				Eigen::VectorXd ng0(9), ng1(9), ng2(9);
				normalized_intensity_vector(w, h, rightImgData, qid, ng0);
				normalized_intensity_vector(w, h, rightImgData, qid + 1, ng1);
				normalized_intensity_vector(w, h, rightImgData, qid - 1, ng2);

				double ncc[3];
				ncc[0] = nf.dot(ng0);
				ncc[1] = nf.dot(ng1);
				ncc[2] = nf.dot(ng2);

				if (ncc[0] >= ncc[1] && ncc[0] >= ncc[2])
				{
					if (abs(ncc[0] - ncc[1]) > 1.0e-6 || abs(ncc[0] - ncc[2]) > 1.0e-6)
					{
						// TODO : dq = 0.5*(ncc[-1]-ncc[1])/(ncc[1]+ncc[-1]-2.0*ncc[0]);
						// disp[pid] = q-dq
						double dq = 0.5*(ncc[2] - ncc[1]) / (ncc[1] + ncc[2] - 2.0*ncc[0]);
						disp[pid] = q - dq;
					}
					else
					{
						disp[pid] = q;
					}
				}
				else if (ncc[1] >= ncc[0] && ncc[1] >= ncc[2])
				{
					disp[pid] = q - 0.5;
				}
				else if (ncc[2] <= ncc[0] && ncc[1] <= ncc[1])
				{
					disp[pid] = q + 0.5;
				}

				// 

				count++;

			}// end if(pixel to be rematched)
		}
	}

	std::cout << ". # of rematched pixels = " << count << std::endl;
}

double bilinear_interpolation(double* f, double dx, double dy)
{
	return (1.0 - dx)*(1.0 - dy)*f[0] + dx * (1.0 - dy)*f[1]
		+ (1.0 - dx)*dy*f[2] + dx * dy*f[3];
}

void FaceStereo::check_level_consistency(int level, std::vector<int>& index)
{
	// Compare the current disparity with that of the lower level
	double* d_curr = disp_l_[level].data();
	double* d_prev = disp_l_[level + 1].data();

	int width_curr = width_[level];
	int height_curr = height_[level];

	int width_prev = width_[level + 1];
	int height_prev = height_[level + 1];

	double K_curr = 0.5*(K_l_[level](0, 0) + K_l_[level](1, 1));
	double K_prev = 0.5*(K_l_[level + 1](0, 0) + K_l_[level + 1](1, 1));

	index.reserve(width_curr*height_curr);
	index.resize(0);

	std::cout << "search range=" << search_range_ << std::endl;

	for (int j = 2; j < height_curr - 2; ++j)
	{
		double tj = 0.5*j;
		double dj = tj - (int)tj;

		for (int i = 2; i < width_curr - 2; ++i)
		{
			const int pid_curr = j * width_curr + i;

			double d = d_curr[pid_curr];
			double z_curr = K_curr / d;
			//double dz_curr = 2.0*K_curr / (d*d);

			double ti = 0.5*i;
			double di = ti - (int)ti;

			const int pid_prev = ((int)tj)*width_prev + (int)ti;

			const int x = (int)ti;
			const int y = (int)tj;
			if (x < 0 || x >= width_prev || y < 0 || y >= height_prev)
			{
				std::cout << "(x,y)=" << x << " " << y << std::endl;
			}
			if (pid_prev < 0 || pid_prev >= width_prev * height_prev - (width_prev + 1))
			{
				std::cout << pid_prev << std::endl;
			}
			// interpolate
			double f[4];
			f[0] = d_prev[pid_prev];
			f[1] = d_prev[pid_prev + 1];
			f[2] = d_prev[pid_prev + width_prev];
			f[3] = d_prev[pid_prev + width_prev + 1];

			double D = bilinear_interpolation(f, di, dj);
			double z_prev = K_prev / D;
			//double dz_prev = 2.0*K_prev / (D*D);

			if (level == 2 && (int)ti == 167 && (int)tj == 133)
			{
				std::cout << "z_prev=" << z_prev << ", z_curr=" << z_curr << std::endl;
				std::cout << "d_prev=" << D << ", d_curr=" << d << ", delta(d)=" << 2.0*D - d << std::endl;
			}
			if (level == 1 && (int)ti == 334 && (int)tj == 265)
			{
				std::cout << "z_prev=" << z_prev << ", z_curr=" << z_curr << std::endl;
				std::cout << "d_prev=" << D << ", d_curr=" << d << ", delta(d)=" << 2.0*D - d << std::endl;
			}

			if (abs(z_curr - z_prev) > search_range_)
			{
				index.push_back(pid_curr);
			}
		}// for(i)
	}// for(j)

	std::cout << ". # of level-inconsistency points = " << index.size() << std::endl;
}

double linear_interpolation(double* f, int index, int offset, double dx)
{
	return (1.0 - dx)*f[index] + dx * f[index + offset + 1];
}

void FaceStereo::compute_disparity_photometric_consistency(int level)
{
	const int w = width_[level];
	const int h = height_[level];

	double* d = disp_l_[level].data();
	double* td = temp_disp_l_[level].data();
	std::fill(td, td + w * h, 0.0);

	// Image data
	unsigned char * l_data = pyramid_gray_[0][level].data;
	unsigned char * r_data = pyramid_gray_[1][level].data;

	std::vector<double> ul(w*h), ur(w*h);

	for (int i = 0; i < w*h; ++i)
	{
		ul[i] = (double)l_data[i];
		ur[i] = (double)r_data[i];
	}

	// Compute the derivative of intensity along x-direction
	std::vector< double > ur_x(w*h, 0.0);

	for (int j = 0; j < h; ++j)
	{
		for (int i = 1; i < w - 1; ++i)
		{
			ur_x[j*w + i] = ur[j*w + i + 1] - ur[j*w + i];
		}
	}

	// Method 1 : no normalization
	// NOTE : At the moment, it does not seem to minimize energy.
	//        Try the normalized cross correlation.
	/*double min_disp = disp_min_[level];
	double max_disp = disp_max_[level];

	int mask[9];
	mask[0] = -w - 1;
	mask[1] = -w;
	mask[2] = -w + 1;
	mask[3] = -1;
	mask[4] =  0;
	mask[5] =  1;
	mask[6] =  w - 1;
	mask[7] =  w;
	mask[8] =  w + 1;

	for (int j = 1; j < h - 1; ++j)
	{
		for (int i = 1; i < w - 1; ++i)
		{
			int pid = j*w + i;

			double q = i - d[pid];
			double qid = (int)floor(q);
			double dq = q - qid;

			if (dq < 0.0 || dq > 1.0)
			{
				std::cout << "ERROR, dq=" << dq << std::endl;
			}

			td[pid] = d[pid];

			if (qid > 0 && qid < w-1 && d[pid] > min_disp && d[pid] < max_disp)
			{

				Eigen::VectorXd dH(9), H(9);

				// initial energy
				for (int k = 0; k < 9; ++k)
				{
					H(k) = ul[pid + mask[k]] - linear_interpolation(ur.data(), j*w + qid + mask[k], dq);
				}

				double initEnergy = H.dot(H);

				for (int k = 0; k < 9; ++k)
				{
					dH(k) = linear_interpolation(ur_x.data(), j*w + qid + mask[k], dq);
					H(k) = ul[pid + mask[k]] - linear_interpolation(ur.data(), j*w + qid + mask[k], dq);
				}

				double dH_norm2 = dH.squaredNorm();
				if (dH_norm2 > 1.0e-6)
				{
					double t = -dH.dot(H) / dH_norm2;
					if (t < -0.5)   t = -0.5;
					else if (t>0.5) t = 0.5;

					if (t > 0) t = -0.0001;
					else if (t < 0.0)
						t = 0.0001;
					else
						t = 0.0;

					double tq = q + t;
					int tqid = (int)floor(tq);
					double tdq = tq - tqid;
					for (int k = 0; k < 9; ++k)
					{
						H(k) = ul[pid + mask[k]] - linear_interpolation(ur.data(), j*w + tqid + mask[k], tdq);
					}

					double updatedEnergy = H.dot(H);
					if (updatedEnergy > initEnergy)
					{
						std::cout << "WARNING, energy is increasing! pid=" << pid << ", t=" << t
							<< ", " << initEnergy << "->" << updatedEnergy << std::endl;
					}

					td[pid] = d[pid] - t;

					//if (level==pyramidDepth_-1)
					//    std::cout << t << " ";
				}
			}
		}// end for(i)
		//if (level == pyramidDepth_-1)
		//    std::cout << std::endl;
	}// end for(j)
	//*/

	// Method 2: Normalized cross correlation
	std::vector< double > F(9 * w*h), G(9 * w*h);
	std::fill(F.begin(), F.end(), 0.0);
	std::fill(G.begin(), G.end(), 0.0);

	int mask[9];
	mask[0] = -w - 1;
	mask[1] = -w;
	mask[2] = -w + 1;
	mask[3] = -1;
	mask[4] = 0;
	mask[5] = 1;
	mask[6] = w - 1;
	mask[7] = w;
	mask[8] = w + 1;

	for (int j = 1; j < h - 1; ++j)
	{
		for (int i = 1; i < w - 1; ++i)
		{
			int pid = j * w + i;

			for (int k = 0; k < 9; ++k)
			{
				F[9 * pid + k] = ul[pid + mask[k]];
				G[9 * pid + k] = ur[pid + mask[k]];
			}

			Eigen::Map<Eigen::VectorXd> tF(F.data() + 9 * pid, 9);
			Eigen::Map<Eigen::VectorXd> tG(G.data() + 9 * pid, 9);

			Eigen::VectorXd v(9);
			v.setConstant(1.0);

			tF = tF - (tF.dot(v) / 9.0)*v;
			tG = tG - (tG.dot(v) / 9.0)*v;

			if (tF.squaredNorm() > 1.0e-12)
				tF.normalize();
			if (tG.squaredNorm() > 1.0e-12)
				tG.normalize();

			//double* ttF = F.data() + 9 * pid;
			//double l = ttF[0] * ttF[0] + ttF[1] * ttF[1] + ttF[2] * ttF[2]
			//    + ttF[3] * ttF[3] + ttF[4] * ttF[4] + ttF[5] * ttF[5]
			//    + ttF[6] * ttF[6] + ttF[7] * ttF[7] + ttF[8] * ttF[8];

			//if (l < 1.0 - 1.0e-9 || l > 1.0 + 1.0e-9)
			//{
			//    std::cout << "l=" << l << std::endl;
			//}
		}
	}

	double min_disp = disp_min_[level];
	double max_disp = disp_max_[level];

	for (int j = 1; j < h - 1; ++j)
	{
		for (int i = 1; i < w - 1; ++i)
		{
			int pid = j * w + i;

			td[pid] = d[pid]; // just copy first

			if (d[pid] > min_disp && d[pid] < max_disp)
			{
				double q = i - d[pid];
				int ti = (int)floor(q + 0.5);
				int qid = j * w + ti;

				if (ti > 1.0 && ti < w - 1)
				{
					Eigen::Map<Eigen::VectorXd> tF(F.data() + 9 * pid, 9);

					Eigen::Map<Eigen::VectorXd> G0(G.data() + 9 * (qid + 0), 9),
						G1(G.data() + 9 * (qid + 1), 9),
						G2(G.data() + 9 * (qid - 1), 9);

					double ncc[3];
					ncc[0] = 0.5*(1.0 - tF.dot(G0));
					ncc[1] = 0.5*(1.0 - tF.dot(G1));
					ncc[2] = 0.5*(1.0 - tF.dot(G2));

					if (ncc[0] <= ncc[1] && ncc[0] <= ncc[2])
					{
						double denom = ncc[1] + ncc[2] - 2.0*ncc[0];

						if (denom < 1.0e-6)
						{
							td[pid] = i - ti;
						}
						else
						{
							// convex
							double t = 0.5*(ncc[2] - ncc[1]) / denom;
							if (t >= 0.25) t = 0.25;
							else if (t <= -0.25) t = -0.25;

							td[pid] = i - (ti + t);
						}
					}
					else if (ncc[1] <= ncc[0] && ncc[1] <= ncc[2])
					{
						if (ncc[1] == ncc[2])
							td[pid] = i - ti;
						else
							// ti += 0.5
							td[pid] = i - (ti + 0.25);
					}
					else
					{
						if (ncc[2] == ncc[1])
							td[pid] = i - ti;
						else
							// ti -= 0.5
							td[pid] = i - (ti - 0.25);
					}

					if (abs(d[pid] - td[pid]) > 0.0)
					{
						std::cout << "(" << i << "," << j << ") : " << d[pid] << "->" << td[pid]
							<< "(" << ncc[2] << ", " << ncc[0] << ", " << ncc[1] << ")"
							<< ", ti=" << ti << std::endl;
					}
				}
			}
		}
	}
	/*
	// Derivatives
	std::vector< double > dG(9 * w*h);
	std::fill(dG.begin(), dG.end(), 0.0);

	for (int j = 0; j < h; ++j)
	{
		for (int i = 1; i < w-1; ++i)
		{
			const int pid = j*w + i;

			for (int k = 0; k < 9; ++k){
				dG[9 * pid + k] = G[9 * (pid+1) + k] - G[9 * pid + k];
			}
		}
	}

	//
	double min_disp = disp_min_[level];
	double max_disp = disp_max_[level];
	for (int j = 1; j < h - 1; ++j)
	{
		for (int i = 1; i < w - 1; ++i)
		{
			int lpid = j*w + i;

			double q = i - d[lpid];
			double qid = floor(q);
			double dq = q - qid;

			int rpid = j*w + (int)qid;

			if (dq < 0.0 || dq > 1.0)
			{
				std::cout << "ERROR, dq=" << dq << std::endl;
			}

			td[lpid] = d[lpid];

			if (qid > 0 && qid < w - 1 && d[lpid] > min_disp && d[lpid] < max_disp)
			{
				//std::cout << "pid=" << lpid << std::endl;
				Eigen::VectorXd dH(9), H(9);

				// initial energy
				for (int k = 0; k < 9; ++k)
				{
					H(k) = F[9 * lpid + k] - linear_interpolation(G.data() + 9 * (rpid), k, 9, dq);
				}

				double initEnergy = H.dot(H);

				for (int k = 0; k < 9; ++k)
				{
					dH(k) = linear_interpolation(dG.data() + 9 * (rpid), k, 9, dq);
					H(k) = F[9 * lpid + k] - linear_interpolation(G.data() + 9 * (rpid), k, 9, dq);
				}

				double dH_norm2 = dH.squaredNorm();
				if (dH_norm2 > 1.0e-6)
				{
					double t = -dH.dot(H) / dH_norm2;
					if (t < -0.2)   t = -0.2;
					else if (t>0.2) t = 0.2;

					//if (t > 0) t = 0.00001;
					//else if (t < 0.0)
					//    t = -0.00001;
					//else
					//   t = 0.0;

					double tq = q + t;
					int tqid = (int)floor(tq);
					double tdq = tq - tqid;

					//std::cout << "j=" << j << "i=" << i << ", tqid=" << tqid
					//    << "q=" <<q <<  ", t=" << t << " dH^2=" << dH_norm2 <<")"
					//    << "H=" << H.transpose() << std::endl;
					//Eigen::Map<Eigen::VectorXd> tF(F.data() + 9 * lpid, 9);
					//Eigen::Map<Eigen::VectorXd> tG1(G.data() + 9 * rpid, 9);
					//Eigen::Map<Eigen::VectorXd> tG2(G.data() + 9 * rpid, 9);
					//std::cout << "F=" << tF.transpose() << "\n"
					//    << "G1=" << tG1.transpose() << "\n"
					//    << "G2=" << tG2.transpose() << std::endl;

					for (int k = 0; k < 9; ++k)
					{
						H(k) = F[9 * lpid + k] - linear_interpolation(G.data() + 9 * (j*w + tqid), k, 9, tdq);
					}

					double updatedEnergy = H.dot(H);
					if (updatedEnergy > initEnergy)
					{
						//std::cout << "WARNING, energy is increasing! pid=" << lpid << ", t=" << t
						//    << ", " << initEnergy << "->" << updatedEnergy << std::endl;
					}
					else
					{
						//std::cout << "improved energy : " << initEnergy << "->" << updatedEnergy << std::endl;
						td[lpid] = d[lpid] - t;
					}



					//if (level==pyramidDepth_-1)
					//    std::cout << t << " ";
				}
				else
				{
					//std::cout << "zero gradient" << std::endl;
				}
			}
		}// end for(i)
		//if (level == pyramidDepth_-1)
		//    std::cout << std::endl;
	}// end for(j)//*/
}

void FaceStereo::compute_disparity_surf_consistency(int level, double* disp_surf)
{
	//std::cout << "Start of SURF CONSISTENCY" << std::endl;
	const int w = width_[level];
	const int h = height_[level];

	double* d = disp_l_[level].data();

	for (int j = 1; j < h - 1; ++j)
	{
		for (int i = 1; i < w - 1; ++i)
		{
			const int pid = j * w + i;

			double td[5];
			td[0] = d[pid];
			td[1] = d[pid - 1];
			td[2] = d[pid + 1];
			td[3] = d[pid - w];
			td[4] = d[pid + w];

			if (td[0] > disp_min_[level] && td[0] < disp_max_[level]
				&& td[1] > disp_min_[level] && td[1] < disp_max_[level]
				&& td[2] > disp_min_[level] && td[2] < disp_max_[level]
				&& td[3] > disp_min_[level] && td[3] < disp_max_[level]
				&& td[4] > disp_min_[level] && td[4] < disp_max_[level])
			{

				double temp[2];
				temp[0] = abs(d[pid - 1] - d[pid]) - abs(d[pid + 1] - d[pid]);
				temp[1] = abs(d[pid] - d[pid - w]) - abs(d[pid] - d[pid + w]);

				double weight[2];
				weight[0] = exp(-temp[0] * temp[0]);
				weight[1] = exp(-temp[1] * temp[1]);

				if (weight[0] + weight[1] < 1.0e-9)
					disp_surf[pid] = 0.25*(d[pid - 1] + d[pid + 1] + d[pid - w] + d[pid + w]);
				else
					disp_surf[pid] = 0.5*(weight[0] * (d[pid - 1] + d[pid + 1]) + weight[1] * (d[pid - w] + d[pid + w])) / (weight[0] + weight[1]);

				/*if (pid==343)
				std::cout << "pid=343: "<< temp[0] << ", " << temp[1] << ", w0=" << weight[0] << ", w1=" << weight[1]
				<< ", " << d[pid - 1] << ", " << d[pid + 1] << ", " << d[pid - w] << ", " << d[pid + w] << " "
				<< disp_surf[pid] << std::endl;//*/
			}
			else
				disp_surf[pid] = d[pid];
		}
	}
	//std::cout << "End of SURF CONSISTENCY" << std::endl;
}

void FaceStereo::compute_disparity_surf_consistency(int level, double* d, double* disp_surf)
{
	//std::cout << "Start of SURF CONSISTENCY" << std::endl;
	const int w = width_[level];
	const int h = height_[level];

	for (int j = 1; j < h - 1; ++j)
	{
		for (int i = 1; i < w - 1; ++i)
		{
			const int pid = j * w + i;

			double td[5];
			td[0] = d[pid];
			td[1] = d[pid - 1];
			td[2] = d[pid + 1];
			td[3] = d[pid - w];
			td[4] = d[pid + w];

			if (td[0] > disp_min_[level] && td[0] < disp_max_[level]
				&& td[1] > disp_min_[level] && td[1] < disp_max_[level]
				&& td[2] > disp_min_[level] && td[2] < disp_max_[level]
				&& td[3] > disp_min_[level] && td[3] < disp_max_[level]
				&& td[4] > disp_min_[level] && td[4] < disp_max_[level])
			{

				double temp[2];
				temp[0] = abs(d[pid - 1] - d[pid]) - abs(d[pid + 1] - d[pid]);
				temp[1] = abs(d[pid] - d[pid - w]) - abs(d[pid] - d[pid + w]);

				double weight[2];
				weight[0] = exp(-temp[0] * temp[0]);
				weight[1] = exp(-temp[1] * temp[1]);

				if (weight[0] + weight[1] < 1.0e-9)
					disp_surf[pid] = 0.25*(d[pid - 1] + d[pid + 1] + d[pid - w] + d[pid + w]);
				else
					disp_surf[pid] = 0.5*(weight[0] * (d[pid - 1] + d[pid + 1]) + weight[1] * (d[pid - w] + d[pid + w])) / (weight[0] + weight[1]);

				/*if (pid==343)
				std::cout << "pid=343: "<< temp[0] << ", " << temp[1] << ", w0=" << weight[0] << ", w1=" << weight[1]
				<< ", " << d[pid - 1] << ", " << d[pid + 1] << ", " << d[pid - w] << ", " << d[pid + w] << " "
				<< disp_surf[pid] << std::endl;//*/
			}
			else
				disp_surf[pid] = d[pid];
		}
	}
	//std::cout << "End of SURF CONSISTENCY" << std::endl;
}

void FaceStereo::update_disparity_photemetric_surface(int level, double ws, double* disp_surf)
{
	//std::cout << "Start of update disparity" << std::endl;
	const int w = width_[level];
	const int h = height_[level];
	//std::cout << "w=" << w << ", h=" << h << std::endl;

	double* d = disp_l_[level].data();

	unsigned char* leftImgData = pyramid_gray_[0][level].data;
	unsigned char* rightImgData = pyramid_gray_[1][level].data;

	for (int j = 0; j < h; ++j)
	{
		for (int i = 0; i < w; ++i)
		{
			// Compute the normalized NCC, ie.(1-ncc)/2
			int pid = j * w + i;
			//std::cout << "pid=" << pid << std::endl;

			temp_disp_l_[level][pid] = d[pid];

			if (d[pid] > disp_min_[level] && d[pid] < disp_max_[level]
				&& disp_surf[pid] > disp_min_[level] && disp_surf[pid] < disp_max_[level])
			{
				static Eigen::VectorXd nf(9);
				normalized_intensity_vector(w, h, leftImgData, pid, nf);

				int qid = (int)(pid - d[pid] + 0.5);
				const int ti = (int)(i - d[pid] + 0.5);
				if (ti > 0 && ti < w - 1)
				{
					//std::cout << "qid=" << qid << ", d=" << d[pid] << " ";

					static Eigen::VectorXd ng0(9), ng1(9), ng2(9); // ng2 = ng[-1]
					normalized_intensity_vector(w, h, rightImgData, qid, ng0);      //std::cout << "0";
					normalized_intensity_vector(w, h, rightImgData, qid + 1, ng1);  //std::cout << "1";
					normalized_intensity_vector(w, h, rightImgData, qid - 1, ng2);  //std::cout << "2" << std::endl;
					/*if (qid > pid || d[pid] < 0.0 || qid < 0)
					{
					std::cout << "[" << i << "," << j << "] : pid=" << pid << ", d=" << d[pid] << ", qid=" << qid << std::endl;
					std::cout << "nf=" << nf.transpose() << std::endl;
					std::cout << "ng0=" << ng0.transpose() << std::endl;
					std::cout << "ng1=" << ng1.transpose() << std::endl;
					std::cout << "ng2=" << ng2.transpose() << std::endl;
					}*/


					double eta[3];
					eta[0] = 0.5*(1.0 - ncc(nf, ng0));
					eta[1] = 0.5*(1.0 - ncc(nf, ng1));
					eta[2] = 0.5*(1.0 - ncc(nf, ng2));

					/*
					// Update disparity based on photometric consistency first
					double td_p = pid - qid;
					if (eta[0] <= eta[1] && eta[0] <= eta[2])
					{
					if (eta[0] != eta[1] || eta[0] != eta[2])
					td_p -= 0.5*(eta[2] - eta[1]) / (eta[1] + eta[2] - 2.0*eta[0]);
					}
					else if (eta[1] <= eta[0] && eta[1] <= eta[2])
					{
					if (eta[1] != eta[0] || eta[1] != eta[2])
					td_p -= 0.5;
					}
					else if (eta[2] <= eta[0] && eta[2] <= eta[1])
					{
					if (eta[2] != eta[0] || eta[2] != eta[1])
					td_p += 0.5;
					}//*/

					double wp;
					if (eta[0] <= eta[1] && eta[0] <= eta[2])
						wp = 0.5*(eta[1] + eta[2] - 2.0*eta[0]);
					else if (eta[2] <= eta[0] && eta[2] <= eta[1])
						wp = eta[0] - eta[2];
					else
						wp = eta[0] - eta[1];

					//std::cout << "wp=" << wp << std::endl;
					//std::cout << "ds=" << disp_surf[pid] <<  std::endl;
					//wp = 0.0;

					double d_p = d[pid];
					//double d_p = temp_disp_l_[level][pid];
					temp_disp_l_[level][pid] = (wp*d_p + ws * disp_surf[pid]) / (wp + ws);

					//if (abs(d[pid] - d_p) > 0.0)
					//{
					//    std::cout << "(" << i << "," << j << ") : " << d[pid] << "->" << d_p 
					//        << "(" << eta[2] << ", " << eta[0] << ", " << eta[1] << ")"
					//        << ", (wp, ws)=" << wp << ", " << ws  << std::endl;
					//}

					//if ((i == 163) && j == 137)
					//{
					//    std::cout << "d=" << d[pid] << ", delta_d = " << 0.5*(eta[2] - eta[1]) / (eta[1] + eta[2] - 2.0*eta[0])
					//        << "dp =" << d_p << ", wp=" << wp
					//        << ", q=" << ((int)(pid-td_p+0.5))%w << ", " << eta[0] << " " << eta[1] << " " << eta[2] 
					//        << ", d_final=" << temp_disp_l_[level][pid] << std::endl;
					//    //std::cout << "nf=" << nf.transpose() << "\n"
					//    //    << "ng0=" << ng0.transpose() << "\n"
					//    //    << "ng1=" << ng1.transpose() << "\n"
					//    //    << "ng2=" << ng2.transpose() << std::endl;
					//}

					/*if (pid == 343)
					{
					std::cout << "wp=" << wp << ", ws=" << ws << ", dp=" << d[pid] << ", ds=" << disp_surf[pid] << std::endl;
					}*/
				}
			}
			/*else
			{
				temp_disp_l_[level][pid] = d[pid];
			}*/
		}
	}
	//std::cout << "Start of update disparity" << std::endl;
}
void FaceStereo::update_disparity_photemetric_surface_use_multithread(int level, double ws, double* disp_surf)
{
	//std::cout << "Start of update disparity" << std::endl;

	//std::cout << "w=" << w << ", h=" << h << std::endl;
	int threadSize = std::thread::hardware_concurrency();
	//std::cout << "Thread Size : " << threadSize << std::endl;
	std::vector< std::thread> threads(threadSize);
	for (int i = 0; i < threadSize; ++i) {
		threads[i] = std::thread(&FaceStereo::update_disparity_photemetric_surface_block, this, level, ws, disp_surf, i, threadSize);
	}
	for (int i = 0; i < threadSize; ++i) {
		threads[i].join();
	}
}

void FaceStereo::update_disparity_photemetric_surface_block(int level, double ws, double * disp_surf, int tid, int threadNum)
{

	const int w = width_[level];
	const int h = height_[level];
	double* d = disp_l_[level].data();

	unsigned char* leftImgData = pyramid_gray_[0][level].data;
	unsigned char* rightImgData = pyramid_gray_[1][level].data;
	Eigen::VectorXd nf(9);
	//Eigen::VectorXd ng0(9), ng1(9), ng2(9);
	std::vector<Eigen::VectorXd> f, temp, ng;
	//std::vector <std::thread> threadPoor(3);
	for (int i = 0; i < 3; ++i)
	{
		Eigen::VectorXd fi(9);
		f.push_back(fi);
		temp.push_back(fi);
		ng.push_back(fi);
	}
	int start = (int)((tid / (float)threadNum)*h);
	int end = (int)(((tid + 1) / (float)threadNum)*h);

	for (int j = start; j < end; ++j) {
		for (int i = 0; i < w; ++i) {
			// Compute the normalized NCC, ie.(1-ncc)/2
			int pid = j * w + i;
			temp_disp_l_[level][pid] = d[pid];

			if (d[pid] > disp_min_[level] && d[pid] < disp_max_[level]
				&& disp_surf[pid] > disp_min_[level] && disp_surf[pid] < disp_max_[level])
			{
				normalized_intensity_vector_block(w, h, leftImgData, pid, &nf, &(f[0]), &(temp[0]));

				int qid = (int)(pid - d[pid] + 0.5);
				const int ti = (int)(i - d[pid] + 0.5);
				if (ti > 0 && ti < w - 1)
				{
					double eta[3];
					normalized_intensity_vector_block(w, h, rightImgData, qid, &(ng[0]), &(f[0]), &(temp[0]));//std::cout << "0";
					normalized_intensity_vector_block(w, h, rightImgData, qid + 1, &(ng[1]), &(f[0]), &(temp[0]));  //std::cout << "1";
					normalized_intensity_vector_block(w, h, rightImgData, qid - 1, &(ng[2]), &(f[0]), &(temp[0]));  //std::cout << "2" << std::endl;				
					eta[0] = 0.5*(1.0 - ncc(nf, ng[0]));
					eta[1] = 0.5*(1.0 - ncc(nf, ng[1]));
					eta[2] = 0.5*(1.0 - ncc(nf, ng[2]));

					double wp;
					if (eta[0] <= eta[1] && eta[0] <= eta[2])
						wp = 0.5*(eta[1] + eta[2] - 2.0*eta[0]);
					else if (eta[2] <= eta[0] && eta[2] <= eta[1])
						wp = eta[0] - eta[2];
					else
						wp = eta[0] - eta[1];
					double d_p = d[pid];
					temp_disp_l_[level][pid] = (wp*d_p + ws * disp_surf[pid]) / (wp + ws);

				}
			}
		}
	}

	if (false)
	{

		const int w = width_[level];
		const int h = height_[level];
		double* d = disp_l_[level].data();

		unsigned char* leftImgData = pyramid_gray_[0][level].data;
		unsigned char* rightImgData = pyramid_gray_[1][level].data;
		Eigen::VectorXd nf(9);
		//Eigen::VectorXd ng0(9), ng1(9), ng2(9);
		std::vector<Eigen::VectorXd> f, temp, ng;
		//std::vector <std::thread> threadPoor(3);
		for (int i = 0; i < 3; ++i)
		{
			Eigen::VectorXd fi(9);
			f.push_back(fi);
			temp.push_back(fi);
			ng.push_back(fi);
		}
		int i, j;
		//std::function<void(int, int, unsigned char*, int, Eigen::VectorXd*, Eigen::VectorXd*, Eigen::VectorXd*, Eigen::VectorXd*, double*)> threadBlock[3] = {
		////std::bind(&FaceStereo::normalized_intensity_vector_block0,this,std::placeholders::_1,std::placeholders::_2,std::placeholders::_3,std::placeholders::_4,std::placeholders::_5,std::placeholders::_6,std::placeholders::_7,std::placeholders::_8,std::placeholders::_9),
		////std::bind(&FaceStereo::normalized_intensity_vector_block1,this,std::placeholders::_1,std::placeholders::_2,std::placeholders::_3,std::placeholders::_4,std::placeholders::_5,std::placeholders::_6,std::placeholders::_7,std::placeholders::_8,std::placeholders::_9),
		////std::bind(&FaceStereo::normalized_intensity_vector_block2,this,std::placeholders::_1,std::placeholders::_2,std::placeholders::_3,std::placeholders::_4,std::placeholders::_5,std::placeholders::_6,std::placeholders::_7,std::placeholders::_8,std::placeholders::_9)
		//	FaceStereo::normalized_intensity_vector_block0,
		//	FaceStereo::normalized_intensity_vector_block1,
		//	FaceStereo::normalized_intensity_vector_block2
		//};
		//void(FaceStereo:: *threadBlock[3])(int, int, unsigned char*, int, Eigen::VectorXd*, Eigen::VectorXd*, Eigen::VectorXd*, Eigen::VectorXd*, double*) = {
		//	&FaceStereo::normalized_intensity_vector_block0,
		//	&FaceStereo::normalized_intensity_vector_block1,
		//	&FaceStereo::normalized_intensity_vector_block2
		//};

		//auto dd = FaceStereo::normalized_intensity_vector_block0;


		for (int pid = start; pid < w*h; pid += threadNum)
		{
			j = (pid / w);
			i = pid - (j*w);

			// Compute the normalized NCC, ie.(1-ncc)/2

			temp_disp_l_[level][pid] = d[pid];

			if (d[pid] > disp_min_[level] && d[pid] < disp_max_[level]
				&& disp_surf[pid] > disp_min_[level] && disp_surf[pid] < disp_max_[level])
			{
				normalized_intensity_vector_block(w, h, leftImgData, pid, &nf, &(f[0]), &(temp[0]));

				int qid = (int)(pid - d[pid] + 0.5);
				const int ti = (int)(i - d[pid] + 0.5);
				if (ti > 0 && ti < w - 1)
				{
					double eta[3];
					normalized_intensity_vector_block(w, h, rightImgData, qid, &(ng[0]), &(f[0]), &(temp[0]));//std::cout << "0";
					normalized_intensity_vector_block(w, h, rightImgData, qid + 1, &(ng[1]), &(f[0]), &(temp[0]));  //std::cout << "1";
					normalized_intensity_vector_block(w, h, rightImgData, qid - 1, &(ng[2]), &(f[0]), &(temp[0]));  //std::cout << "2" << std::endl;				
					eta[0] = 0.5*(1.0 - ncc(nf, ng[0]));
					eta[1] = 0.5*(1.0 - ncc(nf, ng[1]));
					eta[2] = 0.5*(1.0 - ncc(nf, ng[2]));

					double wp;
					if (eta[0] <= eta[1] && eta[0] <= eta[2])
						wp = 0.5*(eta[1] + eta[2] - 2.0*eta[0]);
					else if (eta[2] <= eta[0] && eta[2] <= eta[1])
						wp = eta[0] - eta[2];
					else
						wp = eta[0] - eta[1];
					double d_p = d[pid];
					temp_disp_l_[level][pid] = (wp*d_p + ws * disp_surf[pid]) / (wp + ws);

				}
			}
		}
	}
}

void FaceStereo::update_disparity_photemetric_surface(int level, double* d_p, double ws, double* disp_surf, double* disp_updated)
{
	//std::cout << "Start of update disparity" << std::endl;
	const int w = width_[level];
	const int h = height_[level];
	//std::cout << "w=" << w << ", h=" << h << std::endl;

	double* d = d_p; // disp_l_[level].data();

	unsigned char* leftImgData = pyramid_gray_[0][level].data;
	unsigned char* rightImgData = pyramid_gray_[1][level].data;

	for (int j = 0; j < h; ++j)
	{
		for (int i = 0; i < w; ++i)
		{
			// Compute the normalized NCC, ie.(1-ncc)/2
			int pid = j * w + i;
			//std::cout << "pid=" << pid << std::endl;

			if (d[pid] > disp_min_[level] && d[pid] < disp_max_[level]
				&& disp_surf[pid] > disp_min_[level] && disp_surf[pid] < disp_max_[level])
			{
				Eigen::VectorXd nf(9);
				normalized_intensity_vector(w, h, leftImgData, pid, nf);

				int qid = (int)(pid - d[pid] + 0.5);
				//std::cout << "qid=" << qid << ", d=" << d[pid] << " ";

				Eigen::VectorXd ng0(9), ng1(9), ng2(9); // ng2 = ng[-1]
				normalized_intensity_vector(w, h, rightImgData, qid, ng0);      //std::cout << "0";
				normalized_intensity_vector(w, h, rightImgData, qid + 1, ng1);  //std::cout << "1";
				normalized_intensity_vector(w, h, rightImgData, qid - 1, ng2);  //std::cout << "2" << std::endl;
				/*if (qid > pid || d[pid] < 0.0 || qid < 0)
				{
				std::cout << "[" << i << "," << j << "] : pid=" << pid << ", d=" << d[pid] << ", qid=" << qid << std::endl;
				std::cout << "nf=" << nf.transpose() << std::endl;
				std::cout << "ng0=" << ng0.transpose() << std::endl;
				std::cout << "ng1=" << ng1.transpose() << std::endl;
				std::cout << "ng2=" << ng2.transpose() << std::endl;
				}*/


				double eta[3];
				eta[0] = 0.5*(1.0 - ncc(nf, ng0));
				eta[1] = 0.5*(1.0 - ncc(nf, ng1));
				eta[2] = 0.5*(1.0 - ncc(nf, ng2));
				//std::cout << eta[0] << " " << eta[1] << " " << eta[2] << std::endl;

				// Update disparity based on photometric consistency first
				double td_p = pid - qid;
				//if (eta[0] <= eta[1] && eta[0] <= eta[2])
				//{
				//    if (eta[0] != eta[1] || eta[0] != eta[2])
				//        d_p -= 0.5*(eta[2] - eta[1]) / (eta[1] + eta[2] - 2.0*eta[0]);
				//}
				//else if (eta[1] <= eta[0] && eta[1] <= eta[2])
				//{
				//    if (eta[1] != eta[0] || eta[1] != eta[2])
				//        d_p -= 0.5;
				//}
				//else if (eta[2] <= eta[0] && eta[2] <= eta[1])
				//{
				//    if (eta[2] != eta[0] || eta[2] != eta[1])
				//        d_p += 0.5;
				//}

				double wp;
				if (eta[2] <= eta[0] && eta[2] <= eta[1])
					wp = eta[0] - eta[2];
				else if (eta[0] <= eta[1] && eta[0] <= eta[2])
					wp = 0.5*(eta[1] + eta[2] - 2.0*eta[0]);
				else
					wp = eta[0] - eta[1];

				//std::cout << "wp=" << wp << std::endl;
				//std::cout << "ds=" << disp_surf[pid] <<  std::endl;
				//wp = 0.0;

				td_p = d[pid];
				//d_p = temp_disp_l_[level][pid];
				disp_updated[pid] = (wp*td_p + ws * disp_surf[pid]) / (wp + ws);

				/*if (pid == 343)
				{
				std::cout << "wp=" << wp << ", ws=" << ws << ", dp=" << d[pid] << ", ds=" << disp_surf[pid] << std::endl;
				}*/
			}
			else
			{
				disp_updated[pid] = d[pid];
			}
		}
	}
	//std::cout << "Start of update disparity" << std::endl;
}

void FaceStereo::refinement_iteration(int level, int numIter)
{
	const int w = width_[level];
	const int h = height_[level];

	std::vector< double > disp_surf(w*h);
	temp_disp_l_[level].resize(w*h);

	for (int i = 0; i < numIter; ++i)
	{
		// Compute photometric consistency
		//compute_disparity_photometric_consistency(level);

		// Compute surface consistency component
		compute_disparity_surf_consistency(level, disp_surf.data());

		// Update disparity
		double ws = disp_smoothness_; // 0.012;
		//update_disparity_photemetric_surface(level, ws, disp_surf.data());
		update_disparity_photemetric_surface_use_multithread(level, ws, disp_surf.data());

		//std::swap(disp_l_[level], temp_disp_l_[level]);
		memcpy(disp_l_[level].data(), temp_disp_l_[level].data(), sizeof(double)*w*h);
		//std::cout << i << " ";
	}
	std::cout << std::endl;
}

void FaceStereo::refinement_iteration_2(int level, int numIter)
{
	const int w = width_[level];
	const int h = height_[level];

	std::vector< double > disp_surf(w*h);
	temp_disp_l_[level].resize(w*h);

	for (int i = 0; i < numIter; ++i)
	{
		// Compute photometric consistency
		//compute_disparity_photometric_consistency(level);

		// Compute surface consistency component
		compute_disparity_surf_consistency(level, temp_disp_l_[level].data(), disp_surf.data());

		// Update disparity
		double ws = disp_smoothness_; // 0.012;
		update_disparity_photemetric_surface(level, disp_l_[level].data(), ws, disp_surf.data(), temp_disp_l_[level].data());

		//std::swap(disp_l_[level], temp_disp_l_[level]);
		//memcpy(disp_l_[level].data(), temp_disp_l_[level].data(), sizeof(double)*w*h);
		//std::cout << i << " ";
	}
	std::cout << std::endl;

	memcpy(disp_l_[level].data(), temp_disp_l_[level].data(), sizeof(double)*w*h);
}

void FaceStereo::apply_face_mask_to_disparity(int level)
{
	if (!use_face_mask_)    return;

	const int w = width_[level];
	const int h = height_[level];

	temp_disp_l_[level].resize(w*h);

	unsigned char* maskData = face_mask_[0][level].data;

	for (int i = 0; i < w*h; ++i)
	{
		if ((int)(maskData[i]) == 0)
			temp_disp_l_[level][i] = -2;
		else
			temp_disp_l_[level][i] = disp_l_[level][i];
	}

	memcpy(disp_l_[level].data(), temp_disp_l_[level].data(), sizeof(double)*w*h);

	//for (int i = 0; i < numIter; ++i)
	//{
	//    // Compute photometric consistency
	//    //compute_disparity_photometric_consistency(level);

	//    // Compute surface consistency component
	//    compute_disparity_surf_consistency(level, disp_surf.data());

	//    // Update disparity
	//    double ws = disp_smoothness_; // 0.012;
	//    update_disparity_photemetric_surface(level, ws, disp_surf.data());

	//    //std::swap(disp_l_[level], temp_disp_l_[level]);
	//    memcpy(disp_l_[level].data(), temp_disp_l_[level].data(), sizeof(double)*w*h);
	//    std::cout << i << " ";
	//}
	//std::cout << std::endl;
}

void FaceStereo::compute_disparity(int level)
{
	std::cout
		<< "============================" << "\n"
		<< "**** Disparity Comp. (L=" << level << ")" << "\n"
		<< "============================" << std::endl;

	if (level == pyramidDepth_ - 1) // lowest scale image
	{
		// Compute disparity twice for the left and right images
		int baseImage = 0; // left image
#ifdef THREAD_2
		std::vector<std::thread> th(2);

		th[0] = std::thread(&FaceStereo::compute_raw_disparity_block_matching, this, level, baseImage);
		//compute_raw_disparity_block_matching_2(level, baseImage);
		baseImage = 1; // right image
		//compute_raw_disparity_block_matching(level, baseImage);
		th[1] = std::thread(&FaceStereo::compute_raw_disparity_block_matching, this, level, baseImage);
		th[0].join();
		th[1].join();
#else
		compute_raw_disparity_block_matching(level, baseImage);
		baseImage = 1; // right image
		compute_raw_disparity_block_matching(level, baseImage);
#endif

		if (useRightDisparity_)
			set_raw_disparity_from_the_other(level, 0);

		double* disp = disp_l_[level].data();
		//if (level == 3)
		//{
		//    std::cout << disp[133 * 306 + 167] << std::endl;
		//}

		// Check smooth constraint of disparity
		std::vector<int>  non_smooth_index;
		check_disparity_smoothness(level, non_smooth_index);
		for (int i = 0; i < (int)non_smooth_index.size(); ++i)
		{
			disp[non_smooth_index[i]] = -11.0;
		}

		// Check unique constraint of disparity
		std::vector<int> non_unique_index;
		check_disparity_uniqueness(level, non_unique_index);
		for (int i = 0; i < (int)non_unique_index.size(); ++i)
		{
			disp[non_unique_index[i]] = -12.0;
		}

		// Check ordering constraint of disparity
		std::vector<int> disorder_index;
		check_disparity_ordering(level, disorder_index);
		std::cout << disorder_index.size() << std::endl;
		for (int i = 0; i < (int)disorder_index.size(); ++i)
		{
			disp[disorder_index[i]] = -13.0;
		}

		//{
		//    int pi = 1207 / pow(2.0, level);
		//    int pj = 869 / pow(2.0, level);

		//    const int w = width_[level];
		//    const int h = height_[level];

		//    std::vector<double> left_grad_x(w*h, 0.0), left_grad_y(w*h, 0.0),
		//        right_grad_x(w*h, 0.0), right_grad_y(w*h, 0.0);

		//    unsigned char * f = pyramid_gray_[0][level].data;
		//    for (int j = 1; j < h - 1; ++j)
		//    {
		//        for (int i = 1; i < w - 1; ++i)
		//        {
		//            left_grad_x[j*w + i] = (double)f[j*w + i + 1] - (double)f[j*w + i];
		//            left_grad_y[j*w + i] = (double)f[(j + 1)*w + i] - (double)f[j*w + i];
		//        }
		//    }

		//    unsigned char * g = pyramid_gray_[1][level].data;
		//    for (int j = 1; j < h - 1; ++j)
		//    {
		//        for (int i = 1; i < w - 1; ++i)
		//        {
		//            right_grad_x[j*w + i] = (double)g[j*w + i + 1] - (double)g[j*w + i];
		//            right_grad_y[j*w + i] = (double)g[(j + 1)*w + i] - (double)g[j*w + 1];
		//        }
		//    }

		//    Eigen::VectorXd nf(9);
		//    normalized_intensity_vector(w, h, pyramid_gray_[0][level].data, pj*w + pi, nf);

		//    Eigen::VectorXd grad_f_x(9), grad_f_y(9);
		//    unnormalized_window_vector(w, h, left_grad_x.data(), pj*w + pi, grad_f_x);
		//    unnormalized_window_vector(w, h, left_grad_y.data(), pj*w + pi, grad_f_y);

		//    std::vector< double > ncc(w);
		//    for (int i = 0; i < w; ++i)
		//    {
		//        Eigen::VectorXd ng(9);
		//        normalized_intensity_vector(w, h, pyramid_gray_[1][level].data, pj*w + i, ng);

		//        ncc[i] = -nf.dot(ng);

		//        if (ncc[i] < -1.0 || ncc[i] > 1.0)
		//        {

		//            std::cout << "error : " << ncc[i] << ", " << nf.norm() << ", " << ng.norm() << std::endl;
		//        }

		//        //Eigen::VectorXd grad_g_x(9), grad_g_y(9);
		//        //unnormalized_window_vector(w, h, right_grad_x.data(), pj*w + i, grad_g_x);
		//        //unnormalized_window_vector(w, h, right_grad_y.data(), pj*w + pi, grad_g_y);

		//        //ncc[i] += (grad_f_x - grad_g_x).squaredNorm();// +(grad_f_y - grad_g_y).squaredNorm();
		//    }

		//    std::ofstream outFile(std::string("ncc_vec_") + std::to_string(level) + ".txt");
		//    for (int i = 0; i < w; ++i)
		//    {
		//        outFile << ncc[i] << " ";
		//    }
		//    outFile.close();
		//}

		 //===========================================================
		// TODO : Re-match
		//===========================================================
		//rematch(level);

		//===========================================================
		// Refinement
		//===========================================================
		int numIter = (pyramidDepth_ - level)*refine_iter_;// 20;
		std::cout << "numiter=" << numIter << std::endl;
		refinement_iteration(level, numIter);//*/

		//===========================================================
		// Apply face mask
		//===========================================================
		apply_face_mask_to_disparity(level);

		std::ostringstream padded_frame;
		padded_frame << std::setw(paddings_) << std::setfill('0') << currentFrame_;

		std::ofstream outFile(folder_ + "disparity_map_h" + std::to_string(level) + "_" + padded_frame.str() + ".txt");
		const int w = width_[level];
		const int h = height_[level];
		//outFile << w << " " << h << "\n";

		for (int i = 0; i < h; ++i)
		{
			for (int j = 0; j < w; ++j)
			{
				outFile << disp[i*w + j] << " ";
			}
			outFile << "\n";
		}
		outFile.close();//*/

		cv::Mat dispImg(h, w, CV_8UC1);
		unsigned char* imgData = dispImg.data;
		double f_max = 0.0;
		double f_min = std::numeric_limits<double>::infinity();
		for (int j = 0; j < h; ++j)
		{
			for (int i = 0; i < w; ++i)
			{
				double temp = disp[j*w + i] - disp_min_[level];
				if (temp < 0.0) temp = 0.0;

				if (temp > f_max)   f_max = temp;
				else if (temp < f_min)  f_min = temp;

			}
		}

		for (int j = 0; j < h; ++j)
		{
			for (int i = 0; i < w; ++i)
			{
				double temp = disp[j*w + i] - disp_min_[level];
				if (temp < 0.0) temp = 0.0;

				imgData[j*w + i] = (int)((temp - f_min) / (f_max - f_min)*255.999);
			}
		}

		cv::imwrite(folder_ + "disparity_h" + std::to_string(level) + "_" + padded_frame.str() + ".bmp", dispImg);
	}
	else
	{
		// Compute disparity twice for the left and right images
		int baseImage = 0; // left image
		//compute_raw_disparity_block_matching(level, baseImage);
		////compute_raw_disparity_block_matching_2(level, baseImage);
		//baseImage = 1; // right image
		//compute_raw_disparity_block_matching(level, baseImage);
		//compute_raw_disparity_block_matching_2(level, baseImage);
		//std::vector<std::thread> th(2);

		//th[0] = std::thread(&FaceStereo::compute_raw_disparity_block_matching, this, level, baseImage);
		////compute_raw_disparity_block_matching_2(level, baseImage);
		//baseImage = 1; // right image
		//			   //compute_raw_disparity_block_matching(level, baseImage);
		//th[1] = std::thread(&FaceStereo::compute_raw_disparity_block_matching, this, level, baseImage);
		//th[0].join();
		//th[1].join();
#ifdef THREAD_2
		std::vector<std::thread> th(2);

		th[0] = std::thread(&FaceStereo::compute_raw_disparity_block_matching, this, level, baseImage);
		//compute_raw_disparity_block_matching_2(level, baseImage);
		baseImage = 1; // right image
					   //compute_raw_disparity_block_matching(level, baseImage);
		th[1] = std::thread(&FaceStereo::compute_raw_disparity_block_matching, this, level, baseImage);
		th[0].join();
		th[1].join();
#else
		compute_raw_disparity_block_matching(level, baseImage);
		baseImage = 1; // right image
		compute_raw_disparity_block_matching(level, baseImage);
#endif

		if (useRightDisparity_)
			set_raw_disparity_from_the_other(level, 0);

		double* disp = disp_l_[level].data();

		// Check smooth constraint of disparity
		std::vector<int>  non_smooth_index;
		check_disparity_smoothness(level, non_smooth_index);
		for (int i = 0; i < (int)non_smooth_index.size(); ++i)
		{
			disp[non_smooth_index[i]] = -11.0;
		}

		// Check unique constraint of disparity
		std::vector<int> non_unique_index;
		check_disparity_uniqueness(level, non_unique_index);
		for (int i = 0; i < (int)non_unique_index.size(); ++i)
		{
			disp[non_unique_index[i]] = -12.0;
		}

		// Check ordering constraint of disparity
		std::vector<int> disorder_index;
		check_disparity_ordering(level, disorder_index);
		//std::cout << disorder_index.size() << std::endl;
		for (int i = 0; i < (int)disorder_index.size(); ++i)
		{
			disp[disorder_index[i]] = -13.0;
		}

		// Check consistency with the previous level disparity
		//std::vector<int> level_inconsistency_index;
		//check_level_consistency(level, level_inconsistency_index);
		//for (int i = 0; i < (int)level_inconsistency_index.size(); ++i)
		//{
		//    disp[level_inconsistency_index[i]] = -21.0;
		//}

		const int w = width_[level];
		const int h = height_[level];
		if (use_hierarchical_block_matching_)
		{
			std::vector<double> disp_expanded(w*h);
			expand_lower_level_disparity(level, disp_expanded.data());
			//std::cout << "-1" << std::endl;
			//hierarchical_block_matching_multi_thread(level, disp_expanded.data());
			hierarchical_block_matching(level, disp_expanded.data());
			//std::cout << "-2" << std::endl;
		}

		//std::cout << "-3" << std::endl;

		//std::cout << "1 : pixel(163,137) : d=" << disp[137 * width_[level] + 163] << std::endl;
		//std::cout << "1 : pixel(164,136) : d=" << disp[136 * width_[level] + 164] << std::endl;
		//if (level == 0)
		//{
		//    int pi = 1207/pow(2.0, level);
		//    int pj = 869/pow(2.0, level);

		//    const int w = width_[level];
		//    const int h = height_[level];

		//    std::vector<double> left_grad_x(w*h, 0.0), left_grad_y(w*h,0.0),
		//        right_grad_x(w*h, 0.0), right_grad_y(w*h, 0.0);

		//    unsigned char * f = pyramid_gray_[0][level].data;
		//    for (int j = 1; j < h-1; ++j)
		//    {
		//        for (int i = 1; i < w - 1; ++i)
		//        {
		//            left_grad_x[j*w + i] = (double)f[j*w + i + 1] - (double)f[j*w + i];
		//            left_grad_y[j*w + i] = (double)f[(j+1)*w + i] - (double)f[j*w + i];
		//        }
		//    }

		//    unsigned char * g = pyramid_gray_[1][level].data;
		//    for (int j = 1; j < h - 1; ++j)
		//    {
		//        for (int i = 1; i < w - 1; ++i)
		//        {
		//            right_grad_x[j*w + i] = (double)g[j*w + i + 1] - (double)g[j*w + i];
		//            right_grad_y[j*w + i] = (double)g[(j + 1)*w + i] - (double)g[j*w + 1];
		//        }
		//    }

		//    Eigen::VectorXd nf(9);
		//    normalized_intensity_vector(w, h, pyramid_gray_[0][level].data, pj*w + pi, nf);

		//    Eigen::VectorXd grad_f_x(9), grad_f_y(9);
		//    unnormalized_window_vector(w, h, left_grad_x.data(), pj*w + pi, grad_f_x);
		//    unnormalized_window_vector(w, h, left_grad_y.data(), pj*w + pi, grad_f_y);

		//    std::vector< double > ncc(w);
		//    for (int i = 0; i < w; ++i)
		//    {
		//        Eigen::VectorXd ng(9);
		//        normalized_intensity_vector(w, h, pyramid_gray_[1][level].data, pj*w + i, ng);

		//        ncc[i] = -nf.dot(ng);

		//        //Eigen::VectorXd grad_g_x(9), grad_g_y(9);
		//        //unnormalized_window_vector(w, h, right_grad_x.data(), pj*w + i, grad_g_x);
		//        //unnormalized_window_vector(w, h, right_grad_y.data(), pj*w + i, grad_g_y);

		//        //ncc[i] += (grad_f_x - grad_g_x).squaredNorm();// +(grad_f_y - grad_g_y).squaredNorm();
		//    }

		//    std::ofstream outFile(std::string("ncc_vec_")+std::to_string(level)+".txt");
		//    for (int i = 0; i < w; ++i)
		//    {
		//        outFile << ncc[i] << " ";
		//    }
		//    outFile.close();
		//}


		//===========================================================
		// TODO : Re-match
		//===========================================================
		//rematch(level);


		//===========================================================
		// Refinement
		//===========================================================
		int numIter = (pyramidDepth_ - level)*refine_iter_;// 20;
		std::cout << "numiter=" << numIter << std::endl;
		refinement_iteration(level, numIter);//*/

		//===========================================================
		// Apply face mask
		//===========================================================
		apply_face_mask_to_disparity(level);

		bool fileOut = false;
		if (fileOut) {
			std::ostringstream padded_frame;
			padded_frame << std::setw(paddings_) << std::setfill('0') << currentFrame_;

			std::ofstream outFile(folder_ + "disparity_map_h" + std::to_string(level) + "_" + padded_frame.str() + ".txt");
			//const int w = width_[level];
			//const int h = height_[level];
			//outFile << w << " " << h << "\n";

			for (int i = 0; i < h; ++i)
			{
				for (int j = 0; j < w; ++j)
				{
					outFile << disp[i*w + j] << " ";
				}
				outFile << "\n";
			}
			outFile.close();//*/

			cv::Mat dispImg(h, w, CV_8UC1);
			unsigned char* imgData = dispImg.data;
			double f_max = 0.0;
			double f_min = std::numeric_limits<double>::infinity();
			for (int j = 0; j < h; ++j)
			{
				for (int i = 0; i < w; ++i)
				{
					double temp = disp[j*w + i] - disp_min_[level];
					if (temp < 0.0) temp = 0.0;

					if (temp > f_max)   f_max = temp;
					else if (temp < f_min)  f_min = temp;

				}
			}
			std::cout << "f : " << f_min << "-" << f_max << std::endl;

			for (int j = 0; j < h; ++j)
			{
				for (int i = 0; i < w; ++i)
				{
					double temp = disp[j*w + i] - disp_min_[level];
					if (temp < 0.0) temp = 0.0;

					imgData[j*w + i] = (int)((temp - f_min) / (f_max - f_min)*255.999);
				}
			}

			cv::imwrite(folder_ + "disparity_h" + std::to_string(level) + "_" + padded_frame.str() + ".bmp", dispImg);
		}
	}
}

void FaceStereo::compute_disparity_2(int level)
{
	std::cout
		<< "============================" << "\n"
		<< "**** Disparity Comp. (L=" << level << ")" << "\n"
		<< "============================" << std::endl;

	if (level == pyramidDepth_ - 1) // lowest scale image
	{
		// Compute disparity twice for the left and right images
		int baseImage = 0; // left image
		compute_raw_disparity_block_matching(level, baseImage);
		//compute_raw_disparity_block_matching_2(level, baseImage);
		baseImage = 1; // right image
		compute_raw_disparity_block_matching(level, baseImage);
		//compute_raw_disparity_block_matching_2(level, baseImage);

		double* disp = disp_l_[level].data();
		//if (level == 3)
		//{
		//    std::cout << disp[133 * 306 + 167] << std::endl;
		//}

		// Check smooth constraint of disparity
		std::vector<int>  non_smooth_index;
		check_disparity_smoothness(level, non_smooth_index);
		for (int i = 0; i < (int)non_smooth_index.size(); ++i)
		{
			disp[non_smooth_index[i]] = -11.0;
		}

		// Check unique constraint of disparity
		std::vector<int> non_unique_index;
		check_disparity_uniqueness(level, non_unique_index);
		for (int i = 0; i < (int)non_unique_index.size(); ++i)
		{
			disp[non_unique_index[i]] = -12.0;
		}

		// Check ordering constraint of disparity
		std::vector<int> disorder_index;
		check_disparity_ordering(level, disorder_index);
		for (int i = 0; i < (int)disorder_index.size(); ++i)
		{
			disp[disorder_index[i]] = -13.0;
		}

		/*
		{
			int pi = 1207 / pow(2.0, level);
			int pj = 869 / pow(2.0, level);

			const int w = width_[level];
			const int h = height_[level];

			std::vector<double> left_grad_x(w*h, 0.0), left_grad_y(w*h, 0.0),
				right_grad_x(w*h, 0.0), right_grad_y(w*h, 0.0);

			unsigned char * f = pyramid_gray_[0][level].data;
			for (int j = 1; j < h - 1; ++j)
			{
				for (int i = 1; i < w - 1; ++i)
				{
					left_grad_x[j*w + i] = (double)f[j*w + i + 1] - (double)f[j*w + i];
					left_grad_y[j*w + i] = (double)f[(j + 1)*w + i] - (double)f[j*w + i];
				}
			}

			unsigned char * g = pyramid_gray_[1][level].data;
			for (int j = 1; j < h - 1; ++j)
			{
				for (int i = 1; i < w - 1; ++i)
				{
					right_grad_x[j*w + i] = (double)g[j*w + i + 1] - (double)g[j*w + i];
					right_grad_y[j*w + i] = (double)g[(j + 1)*w + i] - (double)g[j*w + 1];
				}
			}

			Eigen::VectorXd nf(9);
			normalized_intensity_vector(w, h, pyramid_gray_[0][level].data, pj*w + pi, nf);

			Eigen::VectorXd grad_f_x(9), grad_f_y(9);
			unnormalized_window_vector(w, h, left_grad_x.data(), pj*w + pi, grad_f_x);
			unnormalized_window_vector(w, h, left_grad_y.data(), pj*w + pi, grad_f_y);

			std::vector< double > ncc(w);
			for (int i = 0; i < w; ++i)
			{
				Eigen::VectorXd ng(9);
				normalized_intensity_vector(w, h, pyramid_gray_[1][level].data, pj*w + i, ng);

				ncc[i] = -nf.dot(ng);

				if (ncc[i] < -1.0 || ncc[i] > 1.0)
				{

					std::cout << "error : " << ncc[i] << ", " << nf.norm() << ", " << ng.norm() << std::endl;
				}

				//Eigen::VectorXd grad_g_x(9), grad_g_y(9);
				//unnormalized_window_vector(w, h, right_grad_x.data(), pj*w + i, grad_g_x);
				//unnormalized_window_vector(w, h, right_grad_y.data(), pj*w + pi, grad_g_y);

				//ncc[i] += (grad_f_x - grad_g_x).squaredNorm();// +(grad_f_y - grad_g_y).squaredNorm();
			}

			std::ofstream outFile(std::string("ncc_vec_") + std::to_string(level) + ".txt");
			for (int i = 0; i < w; ++i)
			{
				outFile << ncc[i] << " ";
			}
			outFile.close();
		}//*/

		//===========================================================
		// TODO : Re-match
		//===========================================================
		//rematch(level);

		//===========================================================
		// Refinement
		//===========================================================
		int numIter = (pyramidDepth_ - level)*refine_iter_;// 20;
		std::cout << "numiter=" << numIter << std::endl;
		refinement_iteration(level, numIter);//*/

		std::ofstream outFile(folder_ + "disparity_map_h" + std::to_string(level) + ".txt");
		const int w = width_[level];
		const int h = height_[level];
		//outFile << w << " " << h << "\n";

		for (int i = 0; i < h; ++i)
		{
			for (int j = 0; j < w; ++j)
			{
				outFile << disp[i*w + j] << " ";
			}
			outFile << "\n";
		}
		outFile.close();//*/

		cv::Mat dispImg(h, w, CV_8UC1);
		unsigned char* imgData = dispImg.data;
		double f_max = 0.0;
		double f_min = std::numeric_limits<double>::infinity();
		for (int j = 0; j < h; ++j)
		{
			for (int i = 0; i < w; ++i)
			{
				double temp = disp[j*w + i] - disp_min_[level];
				if (temp < 0.0) temp = 0.0;

				if (temp > f_max)   f_max = temp;
				else if (temp < f_min)  f_min = temp;

			}
		}

		for (int j = 0; j < h; ++j)
		{
			for (int i = 0; i < w; ++i)
			{
				double temp = disp[j*w + i] - disp_min_[level];
				if (temp < 0.0) temp = 0.0;

				imgData[j*w + i] = (int)((temp - f_min) / (f_max - f_min)*255.999);
			}
		}

		cv::imwrite(folder_ + "disparity_h" + std::to_string(level) + ".bmp", dispImg);
	}
	else
	{

	}
}

void Morphology(const cv::Mat &imgIn, cv::Mat &imgOut, int morpOp = cv::MORPH_CLOSE,
	int minThickess = 2, int shape = cv::MORPH_ELLIPSE)
{
	int size = minThickess / 2;
	cv::Point anchor = cv::Point(size, size);
	cv::Mat element = getStructuringElement(shape, cv::Size(2 * size + 1, 2 * size + 1), anchor);
	cv::morphologyEx(imgIn, imgOut, morpOp, element, anchor);
}

double find_hue_and_mask(cv::Mat src, cv::Rect rect, cv::Point2i& hue_range, cv::Mat& mask, cv::Mat& masked)
{
	const int w = src.cols;
	const int h = src.rows;

	cv::Mat blur;
	cv::GaussianBlur(src, blur, cv::Size(), 2, 2);

	//convert to HSV
	cv::Mat src_hsv;
	cvtColor(blur, src_hsv, cv::COLOR_BGR2HSV);

	std::vector<cv::Mat> hsv_planes;
	split(src_hsv, hsv_planes);

	cv::Mat hue = hsv_planes[0];
	//std::cout << "type=" << hue.type() << "-" << CV_8UC1 << std::endl;

	int num_pixels = rect.width*rect.height;
	//std::cout << "num pixels = " << num_pixels << std::endl;

	unsigned char* data = hue.data;

	double mean = 0.0, dev = 0.0;
	for (int j = 0; j < rect.height; ++j)
	{
		for (int i = 0; i < rect.width; ++i)
		{
			int pindex = (j + rect.y)*w + i + rect.x;

			const int value = data[pindex];
			mean += value;
			dev += value * value;
		}
	}

	if (num_pixels > 0)
	{
		mean /= num_pixels;
		dev = sqrt(dev / num_pixels - mean * mean);
	}

	//std::cout << "hue : mean=" << mean << ", dev=" << dev << std::endl;

	hue_range.x = 1; // (mean - dev) > 1 ? mean - dev : 1;
	hue_range.y = (mean + 0.5*dev) > 255.0 ? 255 : (int)(mean + 0.5*dev);
	//std::cout << "hue range : " << hue_range << std::endl;


	masked = cv::Mat::zeros(h, w, CV_8UC3); //cv::Mat masked = cv::Mat::zeros(h, w, CV_8UC3);
	mask = cv::Mat::zeros(h, w, CV_8UC1); //cv::Mat mask = cv::Mat::zeros(h, w, CV_8UC1);

	for (int j = 0; j < h; ++j)
	{
		for (int i = 0; i < w; ++i)
		{
			const int pindex = j * w + i;

			if (data[pindex] > hue_range.x && data[pindex] < hue_range.y)
			{
				masked.data[3 * pindex + 0] = src.data[3 * pindex + 0];
				masked.data[3 * pindex + 1] = src.data[3 * pindex + 1];
				masked.data[3 * pindex + 2] = src.data[3 * pindex + 2];

				mask.data[pindex] = 255;
			}
		}
	}

	//cv::imwrite("masked.bmp", masked);
	//cv::imwrite("mask.bmp", mask);

	return mean;
}

bool FaceStereo::get_face_mask(cv::Mat& srcImg, cv::Mat& mask)
{
	//std::string face_xml_file = argv[1];
	//cv::Mat tempImg = cv::imread(argv[2]);

	//cv::Mat inputImg;
	//cv::resize(tempImg, inputImg, cv::Size(tempImg.cols / 2, tempImg.rows / 2));

	const int w = srcImg.cols;
	const int h = srcImg.rows;

	int dx = (int)(w * 0.2);
	int dy = (int)(h * 0.2);

	cv::Mat padded = cv::Mat::zeros(h + 2 * dy, w + 2 * dx, srcImg.type());
	srcImg.copyTo(padded(cv::Rect(dx, dy, w, h)));

	cv::Mat gray_img;
	cv::cvtColor(padded, gray_img, cv::COLOR_BGR2GRAY);

	std::vector<cv::Rect_<int> > faces;
	//cv::CascadeClassifier face_cascade;
	//face_cascade.load(face_xml_file);

	face_cascade_.detectMultiScale(gray_img, faces, 1.15, 3, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));

	if (faces.size() == 0)  return false;

	// NOTE : Find the region with the largest area
	cv::Rect face = faces[faces.size() - 1];

	face.x -= dx;
	face.y -= dy;
	if (face.x + face.width > w - 1) face.width = w - 1 - face.x;
	if (face.y + face.height > h - 1) face.height = h - 1 - face.y;

	//std::cout << "- Face region : " << face << std::endl;

	//cv::Point2i bmin, bmax;
	//bmin = cv::Point2i(face.x, face.y);
	//bmax = cv::Point2i(face.x + face.width, face.y + face.height);

	//cv::rectangle(inputImg, face, cv::Scalar(0, 0, 255));

	//cv::imshow("face_detection", inputImg);
	//cv::waitKey(0);

	cv::Point2i hue_range;
	cv::Mat masked;
	find_hue_and_mask(padded(cv::Rect(dx, dy, w, h)), face, hue_range, mask, masked);

	return true;

	/*cv::Mat maskedImg;
	maskedImg = cv::Mat::zeros(h, w, CV_8UC3);

	unsigned char* maskedData = maskedImg.data;
	unsigned char* srcData = inputImg.data;

	for (int j = 0; j < h; ++j)
	{
	for (int i = 0; i < w; ++i)
	{
	int pindex = j*w + i;

	if (i > bmin.x && i < bmax.x && j > bmin.y && j < bmax.y)
	{
	maskedData[3 * pindex + 0] = srcData[3 * pindex + 0];
	maskedData[3 * pindex + 1] = srcData[3 * pindex + 1];
	maskedData[3 * pindex + 2] = srcData[3 * pindex + 2];
	}
	}
	}

	cv::imwrite("masked_img.bmp", maskedImg);*/

	//threshold(saturation, mask, minSatur, 255, THRESH_BINARY)
}

void FaceStereo::construct_image_pyramid()
{
	std::cout
		<< "============================" << "\n"
		<< "**** Image Pyramid Construction" << "\n"
		<< "============================" << std::endl;

	const int pyrDepth = this->pyramidDepth_;

	rectInputImg_[0].copyTo(pyramid_[0][0]);
	rectInputImg_[1].copyTo(pyramid_[1][0]);

	cv::cvtColor(pyramid_[0][0], pyramid_gray_[0][0], cv::COLOR_BGR2GRAY);
	cv::cvtColor(pyramid_[1][0], pyramid_gray_[1][0], cv::COLOR_BGR2GRAY);

	// mask
	if (use_face_mask_)
	{
		std::cout << "- Finding face masks in the stereo images...";
		if (!get_face_mask(pyramid_[0][0], face_mask_[0][0])
			|| !get_face_mask(pyramid_[1][0], face_mask_[1][0]))
		{
			std::cout << "\n"
				<< "Warning, can't find a face in a source image. Face detection is automatically ignored." << std::endl;
			use_face_mask_ = false;
		}
		else
			std::cout << "done" << std::endl;
	}

	// Just use the green channel instead of the gray image
	//get_channel_green(pyramid_[0][0], pyramid_gray_[0][0]);
	//get_channel_green(pyramid_[1][0], pyramid_gray_[1][0]);
	//get_channel_brightness(pyramid_[0][0], pyramid_gray_[0][0]);
	//get_channel_brightness(pyramid_[1][0], pyramid_gray_[1][0]);
	//cv::Mat gray[2];
	//cv::cvtColor(pyramid_[0][0], gray[0], CV_BGR2GRAY);
	//cv::cvtColor(pyramid_[1][0], gray[1], CV_BGR2GRAY);
	//auto_adjust_gray(gray[0], pyramid_gray_[0][0]);
	//auto_adjust_gray(gray[1], pyramid_gray_[1][0]);

	width_[0] = rectInputImg_[0].cols;
	height_[0] = rectInputImg_[0].rows;
	imgScale_[0] = 1.0;
	disp_min_[0] = minDisparity_;
	disp_max_[0] = maxDisparity_;

	//=====================================================================================
	// NOTE : K_l, K_r are for rectified images. 
	// For level 0, they were set already in rectification(the same for P_l, P_r, campose_l, campose_r).
	// The camera martix for the rectified right image is equal to the left image.
	//=====================================================================================

	cv::imwrite(folder_ + std::string("pyramid_left_") + std::to_string(0) + ".bmp", pyramid_[0][0]);
	cv::imwrite(folder_ + std::string("pyramid_right_") + std::to_string(0) + ".bmp", pyramid_[1][0]);
	cv::imwrite(folder_ + std::string("gray_left_") + std::to_string(0) + ".bmp", pyramid_gray_[0][0]);
	cv::imwrite(folder_ + std::string("gray_right_") + std::to_string(0) + ".bmp", pyramid_gray_[1][0]);

	if (use_face_mask_)
	{
		cv::imwrite(folder_ + std::string("mask_left_") + std::to_string(0) + ".bmp", face_mask_[0][0]);
		cv::imwrite(folder_ + std::string("mask_right_") + std::to_string(0) + ".bmp", face_mask_[1][0]);
	}

	cv::Mat* src[2];
	src[0] = &pyramid_[0][0];
	src[1] = &pyramid_[1][0];

	cv::Mat* srcGray[2];
	srcGray[0] = &pyramid_gray_[0][0];
	srcGray[1] = &pyramid_gray_[1][0];

	cv::Mat* mask[2];
	if (use_face_mask_)
	{
		mask[0] = &face_mask_[0][0];
		mask[1] = &face_mask_[1][0];
	}

	// Scale down by factor 2
	for (int i = 1; i < pyrDepth; ++i)
	{
		cv::pyrDown(*src[0], pyramid_[0][i], cv::Size(src[0]->cols / 2, src[0]->rows / 2));
		cv::pyrDown(*src[1], pyramid_[1][i], cv::Size(src[1]->cols / 2, src[1]->rows / 2));
		cv::pyrDown(*srcGray[0], pyramid_gray_[0][i], cv::Size(srcGray[0]->cols / 2, srcGray[0]->rows / 2));
		cv::pyrDown(*srcGray[1], pyramid_gray_[1][i], cv::Size(srcGray[1]->cols / 2, srcGray[1]->rows / 2));

		if (use_face_mask_)
		{
			cv::pyrDown(*mask[0], face_mask_[0][i], cv::Size(mask[0]->cols / 2, mask[0]->rows / 2));
			cv::pyrDown(*mask[1], face_mask_[1][i], cv::Size(mask[1]->cols / 2, mask[1]->rows / 2));
		}

		src[0] = &pyramid_[0][i];
		src[1] = &pyramid_[1][i];
		srcGray[0] = &pyramid_gray_[0][i];
		srcGray[1] = &pyramid_gray_[1][i];

		if (use_face_mask_)
		{
			mask[0] = &face_mask_[0][i];
			mask[1] = &face_mask_[1][i];
		}

		cv::imwrite(folder_ + std::string("pyramid_left_") + std::to_string(i) + ".bmp", pyramid_[0][i]);
		cv::imwrite(folder_ + std::string("pyramid_right_") + std::to_string(i) + ".bmp", pyramid_[1][i]);
		cv::imwrite(folder_ + std::string("gray_left_") + std::to_string(i) + ".bmp", pyramid_gray_[0][i]);
		cv::imwrite(folder_ + std::string("gray_right_") + std::to_string(i) + ".bmp", pyramid_gray_[1][i]);

		if (use_face_mask_)
		{
			cv::imwrite(folder_ + std::string("mask_left") + std::to_string(i) + ".bmp", face_mask_[0][i]);
			cv::imwrite(folder_ + std::string("mask_right") + std::to_string(i) + ".bmp", face_mask_[1][i]);
		}

		width_[i] = pyramid_[0][i].cols;
		height_[i] = pyramid_[0][i].rows;

		imgScale_[i] = 0.5*imgScale_[i - 1];

		disp_min_[i] = (int)(minDisparity_ * imgScale_[i]);
		disp_max_[i] = (int)(maxDisparity_ * imgScale_[i]);

		Eigen::Matrix3d S;
		S.setIdentity();
		S(0, 0) = S(1, 1) = imgScale_[i];
		K_l_[i] = S * K_l_[0];
		K_r_[i] = S * K_r_[0];
		//std::cout << "K_l_[" << i << "]=\n" << K_l_[i] << "\n"
		//    << "K_r_[" << i << "]=\n" << K_r_[i] << std::endl;

		campose_l_[i] = campose_l_[0];
		campose_r_[i] = campose_r_[0];
		//std::cout << "campose_l_[" << i << "]=\n" << campose_l_[i] << "\n"
		//    << "campose_r_[" << i << "]=\n" << campose_r_[i] << std::endl;

		Eigen::Matrix4d T1, T2;
		T1 = campose_l_[i].inverse();
		T2 = campose_r_[i].inverse();

		P_l_[i].block<3, 4>(0, 0) = T1.block<3, 4>(0, 0);
		P_r_[i].block<3, 4>(0, 0) = T2.block<3, 4>(0, 0);
		//std::cout << "P_l_[" << i << "]=\n" << P_l_[i] << "\n"
		//    << "P_r_[" << i << "]=\n" << P_r_[i] << std::endl;
	}
}

void FaceStereo::rectify_images()
{
	std::cout
		<< "============================" << "\n"
		<< "**** Rectification ";

	if (rectified_input_image_)
	{
		std::cout << ": skipped : the input images are already rectified" << std::endl;

		inputImg_[0].copyTo(rectInputImg_[0]);
		inputImg_[1].copyTo(rectInputImg_[1]);

		H_rect_.setIdentity();

		K_l_[0] = K_[0];
		K_r_[0] = K_[1];
		//std::cout << "K_r[0]=\n" << K_r_[0] << std::endl;

		P_l_[0] = K_[0] * (campose_[0].inverse().block<3, 4>(0, 0));
		P_r_[0] = K_[1] * (campose_[1].inverse().block<3, 4>(0, 0));

		campose_l_[0] = campose_[0];
		campose_r_[0] = campose_[1];

		// Export the homography to disk
		std::ofstream outFile(folder_ + "rect_homography.txt");
		outFile << H_rect_;
		outFile.close();
	}
	else
	{
		std::cout << std::endl;

		std::cout << "============================" << std::endl;

		Eigen::MatrixXd P1(3, 4);
		Eigen::Vector4d camCenter[2];
		camCenter[0] = campose_[0].block<4, 1>(0, 3);
		camCenter[1] = campose_[1].block<4, 1>(0, 3);

		double B = (camCenter[0] - camCenter[1]).norm();
		std::cout << " . Baseline = " << B << std::endl;

		Eigen::Matrix4d T1;
		T1 = campose_[0].inverse();
		P1.block<3, 4>(0, 0) = T1.block<3, 4>(0, 0);
		P1 = K_[0] * P1;

		Eigen::Matrix3d H;
		int w = inputImg_[0].cols;
		int h = inputImg_[0].rows;
		int type = inputImg_[0].type();

		rectInputImg_[0] = cv::Mat(h, w, type);
		rectInputImg_[1] = cv::Mat(h, w, type);

		// NOTE : Left image is unchanged during the recrification
		inputImg_[0].copyTo(rectInputImg_[0]);

		// TEST : Homography for rectification
		//Eigen::Matrix3d tempH = K_[0] * campose_[1].block<3, 3>(0, 0)*K_[1].inverse();
		//std::cout << "Temp homography for rectification :\n " << tempH << std::endl;

		Eigen::Matrix3d tK2;
		Eigen::MatrixXd tP2(3, 4);
		rectify(K_[0].data(), P1.data(), B, (int)cp_.size(), cp_r_.data()->data(), cp_.data()->data(),
			tK2.data(), tP2.data(),
			H.data(), w, h, type, inputImg_[0].data, inputImg_[1].data, rectInputImg_[1].data, folder_);//*/

		/*Eigen::Matrix3d tK2 = K_[1];
		Eigen::Matrix3d tCamPose2 = campose_[1].block<3, 3>(0, 0);
		Eigen::MatrixXd tP2(3, 4);
		rectify2(K_[0].data(), P1.data(), B, cp_.size(), cp_r_.data()->data(),
			tK2.data(), tCamPose2.data(), tP2.data(),
			H.data(), w, h, type, inputImg_[0].data, inputImg_[1].data, rectInputImg_[1].data);*/

		H_rect_ = H;

		// Now set camera matrix, projection matrix and camera pose for image pyramid
		K_l_[0] = K_[0];
		K_r_[0] = tK2;
		//std::cout << "K_r[0]=\n" << K_r_[0] << std::endl;

		P_l_[0] = P1;
		P_r_[0] = tP2;
		//std::cout << "P_r[0]=\n" << P_r_[0] << std::endl;

		campose_l_[0] = campose_[0];
		campose_r_[0].setIdentity();
		campose_r_[0](0, 3) = B;

		// Export the homography to disk
		std::ofstream outFile(folder_ + "rect_homography.txt");
		outFile << H;
		outFile.close();
	}
}

bool load_matrix(const char* filename, int rows, int cols, double* M)
{
	std::ifstream inFile(filename);
	if (!inFile.is_open())
	{
		std::cout << "ERROR, can't open matrix file : " << filename << std::endl;
		return false;
	}

	// Column-major
	for (int i = 0; i < rows; ++i)
	{
		for (int j = 0; j < cols; ++j)
		{
			inFile >> M[j*rows + i];
		}
	}

	inFile.close();

	return true;
}

bool read_2d_points(const char* filename, std::vector< Eigen::Vector2d >& x)
{
	std::ifstream inFile(filename);

	if (!inFile.is_open())
	{
		std::cout << "ERROR, can't open 2d point file : " << filename << std::endl;
		return false;
	}

	std::string temp;

	int offset = 0;
	do
	{
		if (temp == "NUM_PTS")
		{
			int numPts;
			inFile >> temp >> numPts;
			x.resize(numPts);

			std::getline(inFile, temp, '\n');
		}
		else if (temp == "OFFSET")
		{
			inFile >> temp >> offset;
			std::getline(inFile, temp, '\n');
		}
		else if (temp == "PTS_DATA")
		{
			inFile >> temp;

			// skip points given by offset
			for (int i = 0; i < offset; ++i)
			{
				inFile >> temp >> temp;
			}

			// Now, read the point coordinates
			for (int i = 0; i < x.size(); ++i)
			{
				inFile >> x[i](0) >> x[i](1);
				//x[i](2) = 1.0;
			}
		}
		else
			inFile >> temp;

	} while (!inFile.eof());

	inFile.close();

	std::cout << x[0].transpose() << ", " << x[x.size() - 1].transpose() << std::endl;

	return true;
}

bool read_3d_points(const char* filename, std::vector<Eigen::Vector3d>& X)
{
	std::ifstream inFile(filename);
	if (!inFile.is_open())
	{
		std::cout << "ERROR, can't open 3d points file : " << filename << std::endl;
		return false;
	}

	int N;
	inFile >> N;
	//std::cout << "N=" << N << std::endl;
	X.resize(N);

	for (int i = 0; i < N; ++i)
	{
		inFile >> X[i](0) >> X[i](1) >> X[i](2);
	}
	inFile.close();

	std::cout << X[0].transpose() << ", " << X[1].transpose() << std::endl;

	return true;
}

void decompose_image_file_name(std::string imgFileName, std::string& prefix, int& paddings, int& startFrame, std::string& ext)
{
	size_t found = imgFileName.find_last_of('.');
	ext = imgFileName.substr(found + 1);

	int count = 0;
	for (size_t i = found - 1; i >= 0; --i)
	{
		int ascii = (int)imgFileName[i];

		if (ascii >= 48 && ascii <= 57)
			count++;
		else
			break;
	}

	prefix = imgFileName.substr(0, imgFileName.size() - (count + 1 + ext.size()));
	paddings = count;

	std::string numStr = imgFileName.substr(prefix.size(), count);

	startFrame = atoi(numStr.c_str());

	std::cout << prefix << " " << count << "(" << startFrame << ") " << ext << std::endl;

}

bool FaceStereo::read_input_file(const char* filename)
{
	std::ifstream inFile(filename);
	if (!inFile.is_open())
	{
		std::cout << "ERROR, can't open input file : " << filename << std::endl;
		return false;
	}

	std::string temp;

	std::string leftImageFile, rightImageFile;
	std::string left_K_file, right_K_file;
	std::string left_campose_file, right_campose_file;
	std::string corner_reconst_file;
	std::string corner_left_file, corner_right_file;

	do
	{
		if (temp == "LEFT_IMAGE_FILE")
		{
			//std::cout << temp << std::endl;
			inFile >> temp >> leftImageFile;

			std::getline(inFile, temp, '\n');
		}
		else if (temp == "RIGHT_IMAGE_FILE")
		{
			//std::cout << temp << std::endl;
			inFile >> temp >> rightImageFile;

			std::getline(inFile, temp, '\n');
		}
		else if (temp == "NUM_FRAMES")
		{
			inFile >> temp >> numFrames_;
			std::getline(inFile, temp, '\n');
		}
		else if (temp == "RECTIFIED")
		{
			//std::cout << temp << std::endl;
			inFile >> temp >> temp;
			if (temp == "F" || temp == "f")
			{
				rectified_input_image_ = false;
			}
			else
			{
				rectified_input_image_ = true;
			}

			std::getline(inFile, temp, '\n');
		}
		else if (temp == "OUTPUT_FORMAT")
		{
			inFile >> temp >> output_format_;
			std::getline(inFile, temp, '\n');
		}
		else if (temp == "USE_RIGHT_DISPARITY")
		{
			//std::cout << temp << std::endl;
			inFile >> temp >> temp;
			if (temp == "F" || temp == "f")
			{
				useRightDisparity_ = false;
			}
			else
			{
				useRightDisparity_ = true;
			}

			std::getline(inFile, temp, '\n');
		}
		else if (temp == "MAX_DISPARITY")
		{
			//std::cout << temp << std::endl;
			inFile >> temp >> maxDisparity_;

			std::getline(inFile, temp, '\n');
		}
		else if (temp == "MIN_DISPARITY")
		{
			//std::cout << temp << std::endl;
			inFile >> temp >> minDisparity_;

			std::getline(inFile, temp, '\n');
		}
		else if (temp == "DISP_SMOOTHNESS")
		{
			//std::cout << temp << std::endl;
			inFile >> temp >> disp_smoothness_;

			std::getline(inFile, temp, '\n');
		}
		else if (temp == "DISP_REFINE_ITERATION")
		{
			//std::cout << temp << std::endl;
			inFile >> temp >> refine_iter_;

			std::getline(inFile, temp, '\n');
		}
		else if (temp == "DISP_SEARCH_RANGE")
		{
			//std::cout << temp << std::endl;
			inFile >> temp >> search_range_;

			std::getline(inFile, temp, '\n');
		}
		else if (temp == "PYRAMID_DEPTH")
		{
			//std::cout << temp << std::endl;
			inFile >> temp >> pyramidDepth_;

			std::getline(inFile, temp, '\n');
		}
		else if (temp == "TERMINATE_LEVEL")
		{
			inFile >> temp >> terminate_level_;

			std::getline(inFile, temp, '\n');
		}
		else if (temp == "HIERARCH_BLOCK_MATCHING")
		{
			//std::cout << temp << std::endl;
			inFile >> temp >> temp;
			if (temp == "F" || temp == "f")
			{
				use_hierarchical_block_matching_ = false;
			}
			else
			{
				use_hierarchical_block_matching_ = true;
			}

			std::getline(inFile, temp, '\n');
		}
		else if (temp == "MAX_DEPTH_CHANGE_RATIO")
		{
			inFile >> temp >> max_depth_change_ratio_;

			std::getline(inFile, temp, '\n');
		}
		else if (temp == "MAX_CURV_CHANGE_RATIO")
		{
			inFile >> temp >> max_curv_change_ratio_;

			std::getline(inFile, temp, '\n');
		}
		else if (temp == "INTEREST_REGION_X")
		{
			//std::cout << temp << std::endl;
			inFile >> temp >> region_min_[0] >> region_max_[0];

			std::getline(inFile, temp, '\n');
		}
		else if (temp == "INTEREST_REGION_Y")
		{
			//std::cout << temp << std::endl;
			inFile >> temp >> region_min_[1] >> region_max_[1];

			std::getline(inFile, temp, '\n');
		}
		else if (temp == "AUTO_FACE_DETECT")
		{
			//std::cout << temp << std::endl;
			int auto_face_detect;
			inFile >> temp >> auto_face_detect;
			auto_face_detect > 0 ? use_face_mask_ = true : use_face_mask_ = false;
			std::getline(inFile, temp, '\n');
		}
		else if (temp == "FACE_DETECT_XML")
		{
			//std::cout << temp << std::endl;
			inFile >> temp >> face_xml_file_;
			std::getline(inFile, temp, '\n');
		}
		else if (temp == "INTRINSIC_PARAMETER")
		{
			//std::cout << temp << std::endl;
			inFile >> temp >> left_K_file >> right_K_file;

			std::getline(inFile, temp, '\n');
		}
		else if (temp == "CAMERA_POSE")
		{
			//std::cout << temp << std::endl;
			inFile >> temp >> left_campose_file >> right_campose_file;

			std::getline(inFile, temp, '\n');
		}
		else if (temp == "CHECK_RECONST_CORNERS")
		{
			//std::cout << temp << std::endl;
			inFile >> temp >> corner_reconst_file;

			std::getline(inFile, temp, '\n');
		}
		else if (temp == "CHECK_CORNER_ON_IMAGE")
		{
			//std::cout << temp << std::endl;
			inFile >> temp >> corner_left_file >> corner_right_file;

			std::getline(inFile, temp, '\n');
		}
		else
		{
			inFile >> temp;
		}
	} while (!inFile.eof());

	inFile.close();

	// For verification
	std::cout
		<< "============================" << "\n"
		<< "**** INPUT PARAMETERS : " << "\n"
		<< "============================" << "\n"
		<< ". STEREO IMAGES     = " << leftImageFile << ", " << rightImageFile
		<< "(rectified=" << rectified_input_image_ << ")" << "\n"
		<< ". DISPARITY         = " << minDisparity_ << "~" << maxDisparity_ << "\n"
		<< ". DISPARITY SMOOTH  = " << disp_smoothness_ << "\n"
		<< ". DISPARITY_REFINE_ITER = " << refine_iter_ << "\n"
		<< ". DISPARITY_SEARCH_RANGE = " << search_range_ << "\n"
		<< ". PYRAMID_DEPTH     = " << pyramidDepth_ << "\n"
		<< ". HIERARCH_BLOCK_MATCHING = " << use_hierarchical_block_matching_ << "\n"
		<< ". MAX_DEPTH_CHANGE_RATIO = " << max_depth_change_ratio_ << "\n"
		<< ". INTEREST REGION   = " << "(" << region_min_[0] << ", " << region_max_[0] << ")x"
		<< "(" << region_min_[1] << ", " << region_max_[1] << ")" << "\n"
		<< ". INTRINSIC PARAM.  = " << left_K_file << ", " << right_K_file << "\n"
		<< ". CAMERA POSE       = " << left_campose_file << ", " << right_campose_file << "\n"
		<< ". CHECK CORNER PTS  = " << corner_reconst_file << ", " << corner_left_file << ", " << corner_right_file << "\n"
		<< std::endl;

	// Load files
	// Extract prefix, paddings and extensions from image file name
	decompose_image_file_name(leftImageFile, leftImgPrefix_, paddings_, startFrame_, imgExt_);
	decompose_image_file_name(rightImageFile, rightImgPrefix_, paddings_, startFrame_, imgExt_);

	// The following image reading block is moved to a separate function
	// which is called at the start of the for-loop.
	//inputImg_[0] = cv::imread(leftImageFile);
	//if (inputImg_[0].empty())
	//{
	//    std::cout << "ERROR, can't load left image : " << leftImageFile << std::endl;
	//    return false;
	//}

	//inputImg_[1] = cv::imread(rightImageFile);
	//if (inputImg_[1].empty())
	//{
	//    std::cout << "ERROR, can't load right image : " << rightImageFile << std::endl;
	//    return false;
	//}

	if (!load_matrix(left_K_file.c_str(), 3, 3, K_[0].data()))       return false;
	std::cout << ". Left K : \n" << K_[0] << std::endl;
	if (!load_matrix(right_K_file.c_str(), 3, 3, K_[1].data()))      return false;
	std::cout << ". Right K : \n" << K_[1] << std::endl;

	if (!load_matrix(left_campose_file.c_str(), 4, 4, campose_[0].data()))      return false;
	std::cout << ". Left Cam. Pose : \n" << campose_[0] << std::endl;
	if (!load_matrix(right_campose_file.c_str(), 4, 4, campose_[1].data()))      return false;
	std::cout << ". Right Cam. Pose : \n" << campose_[1] << std::endl;


	if (!rectified_input_image_ && !read_2d_points(corner_left_file.c_str(), cp_l_))   return false;
	if (!rectified_input_image_ && !read_2d_points(corner_right_file.c_str(), cp_r_))   return false;
	if (!rectified_input_image_ && !read_3d_points(corner_reconst_file.c_str(), cp_))   return false;

	// Allocations
	int pyrDepth = pyramidDepth_;
	pyramid_[0].resize(pyrDepth);
	pyramid_[1].resize(pyrDepth);
	pyramid_gray_[0].resize(pyrDepth);
	pyramid_gray_[1].resize(pyrDepth);

	// Face masking
	face_mask_[0].resize(pyrDepth);
	face_mask_[1].resize(pyrDepth);

	if (use_face_mask_)
	{
		if (!face_cascade_.load(face_xml_file_))
		{
			std::cout << "***********************************************************************************" << "\n"
				<< "*** Warning, can't load an XML file for face detection : " << face_xml_file_ << "\n"
				<< "*** Face detection option will be turned off automatically." << "\n"
				<< "***********************************************************************************" << "\n"
				<< std::endl;
			use_face_mask_ = false;
		}
	}

	width_.resize(pyrDepth);
	height_.resize(pyrDepth);

	imgScale_.resize(pyrDepth);

	disp_min_.resize(pyrDepth);
	disp_max_.resize(pyrDepth);

	K_l_.resize(pyrDepth);
	K_r_.resize(pyrDepth);

	campose_l_.resize(pyrDepth);
	campose_r_.resize(pyrDepth);

	P_l_.resize(pyrDepth);
	P_r_.resize(pyrDepth);

	disp_l_.resize(pyrDepth);
	disp_r_.resize(pyrDepth);
	temp_disp_l_.resize(pyrDepth);
	temp_disp_r_.resize(pyrDepth);
	//disp_surf_.resize(pyrDepth);

	return true;
}

bool FaceStereo::init(std::string & leftImageFile, std::string & rightImageFile,
	int numFrames, bool rectified, int output_format,
	bool useRightDisparity, int max_disparity, int min_disparity,
	double disp_smooth, int disp_iter, double disp_search_range, int pyramid_depth, int termiate_level, bool hierarch_block_matching,
	double max_depth_change_ratio, double max_curv_change_ratio,
	int region_x_min, int region_x_max, int region_y_min, int region_y_max, int auto_face_detect, std::string & face_xml_file,
	std::string  left_K_file, std::string  right_K_file,
	std::string  left_campose_file, std::string  right_campose_file,
	std::string  corner_reconst_file, std::string  corner_left_file, std::string corner_right_file, std::string folder, std::string prefix)
{
	prefix_ = prefix;
	numFrames_ = numFrames;
	folder_ = folder;
	rectified_input_image_ = rectified;

	output_format_ = output_format;

	useRightDisparity_ = useRightDisparity;
	maxDisparity_ = max_disparity;
	minDisparity_ = min_disparity;
	disp_smoothness_ = disp_smooth;
	refine_iter_ = disp_iter;
	search_range_ = disp_search_range;
	pyramidDepth_ = pyramid_depth;
	terminate_level_ = termiate_level;
	use_hierarchical_block_matching_ = hierarch_block_matching;
	max_depth_change_ratio_ = max_depth_change_ratio;
	max_curv_change_ratio_ = max_curv_change_ratio;
	region_min_[0] = region_x_min;
	region_min_[1] = region_y_min;
	region_max_[0] = region_x_max;
	region_max_[1] = region_y_max;
	auto_face_detect > 0 ? use_face_mask_ = true : use_face_mask_ = false;
	face_xml_file_ = face_xml_file;

	std::cout
		<< "============================" << "\n"
		<< "**** INPUT PARAMETERS : " << "\n"
		<< "============================" << "\n"
		<< ". STEREO IMAGES     = " << leftImageFile << ", " << rightImageFile
		<< "(rectified=" << rectified_input_image_ << ")" << "\n"
		<< ". USE_RIGHT_DISPARITY = " << useRightDisparity_ << "\n"
		<< ". DISPARITY         = " << minDisparity_ << "~" << maxDisparity_ << "\n"
		<< ". DISPARITY SMOOTH  = " << disp_smoothness_ << "\n"
		<< ". DISPARITY_REFINE_ITER = " << refine_iter_ << "\n"
		<< ". DISPARITY_SEARCH_RANGE = " << search_range_ << "\n"
		<< ". PYRAMID_DEPTH     = " << pyramidDepth_ << "\n"
		<< ". HIERARCH_BLOCK_MATCHING = " << use_hierarchical_block_matching_ << "\n"
		<< ". MAX_DEPTH_CHANGE_RATIO = " << max_depth_change_ratio_ << "\n"
		<< ". INTEREST REGION   = " << "(" << region_min_[0] << ", " << region_max_[0] << ")x"
		<< "(" << region_min_[1] << ", " << region_max_[1] << ")" << "\n"
		<< ". INTRINSIC PARAM.  = " << left_K_file << ", " << right_K_file << "\n"
		<< ". CAMERA POSE       = " << left_campose_file << ", " << right_campose_file << "\n"
		<< ". CHECK CORNER PTS  = " << corner_reconst_file << ", " << corner_left_file << ", " << corner_right_file << "\n"
		<< std::endl;
	decompose_image_file_name(leftImageFile, leftImgPrefix_, paddings_, startFrame_, imgExt_);
	decompose_image_file_name(rightImageFile, rightImgPrefix_, paddings_, startFrame_, imgExt_);


	if (!load_matrix(left_K_file.c_str(), 3, 3, K_[0].data()))       return false;
	std::cout << ". Left K : \n" << K_[0] << std::endl;
	if (!load_matrix(right_K_file.c_str(), 3, 3, K_[1].data()))      return false;
	std::cout << ". Right K : \n" << K_[1] << std::endl;

	if (!load_matrix(left_campose_file.c_str(), 4, 4, campose_[0].data()))      return false;
	std::cout << ". Left Cam. Pose : \n" << campose_[0] << std::endl;
	if (!load_matrix(right_campose_file.c_str(), 4, 4, campose_[1].data()))      return false;
	std::cout << ". Right Cam. Pose : \n" << campose_[1] << std::endl;

	if (!rectified_input_image_) {
		if (!read_2d_points(corner_left_file.c_str(), cp_l_))   return false;
		if (!read_2d_points(corner_right_file.c_str(), cp_r_))   return false;
		if (!read_3d_points(corner_reconst_file.c_str(), cp_))   return false;
	}
	// Allocations
	int pyrDepth = pyramidDepth_;
	pyramid_[0].resize(pyrDepth);
	pyramid_[1].resize(pyrDepth);
	pyramid_gray_[0].resize(pyrDepth);
	pyramid_gray_[1].resize(pyrDepth);

	// Face masking
	face_mask_[0].resize(pyrDepth);
	face_mask_[1].resize(pyrDepth);

	if (use_face_mask_)
	{
		if (!face_cascade_.load(face_xml_file_))
		{
			std::cout << "***********************************************************************************" << "\n"
				<< "*** Warning, can't load an XML file for face detection : " << face_xml_file_ << "\n"
				<< "*** Face detection option will be turned off automatically." << "\n"
				<< "***********************************************************************************" << "\n"
				<< std::endl;
			use_face_mask_ = false;
		}
	}

	width_.resize(pyrDepth);
	height_.resize(pyrDepth);

	imgScale_.resize(pyrDepth);

	disp_min_.resize(pyrDepth);
	disp_max_.resize(pyrDepth);

	K_l_.resize(pyrDepth);
	K_r_.resize(pyrDepth);

	campose_l_.resize(pyrDepth);
	campose_r_.resize(pyrDepth);

	P_l_.resize(pyrDepth);
	P_r_.resize(pyrDepth);

	disp_l_.resize(pyrDepth);
	disp_r_.resize(pyrDepth);
	temp_disp_l_.resize(pyrDepth);
	temp_disp_r_.resize(pyrDepth);
	//disp_surf_.resize(pyrDepth);

	return true;
}

bool FaceStereo::loadStereoImages(int frame)
{
	currentFrame_ = frame;

	std::ostringstream leftImageFile, rightImageFile;
	leftImageFile << leftImgPrefix_ << std::setw(paddings_) << std::setfill('0') << frame << "." << imgExt_;
	rightImageFile << rightImgPrefix_ << std::setw(paddings_) << std::setfill('0') << frame << "." << imgExt_;

	std::cout << "============================" << "\n"
		<< "**** Loading stereo images" << "\n"
		<< "============================" << "\n";
	std::cout << "- Left image : " << leftImageFile.str() << std::endl;
	inputImg_[0] = cv::imread(leftImageFile.str());
	if (inputImg_[0].empty())
	{
		std::cout << "ERROR, can't load left image : " << leftImageFile.str() << std::endl;
		return false;
	}

	std::cout << "- Right image : " << rightImageFile.str() << std::endl;
	inputImg_[1] = cv::imread(rightImageFile.str());
	if (inputImg_[1].empty())
	{
		std::cout << "ERROR, can't load right image : " << rightImageFile.str() << std::endl;
		return false;
	}

	return true;
}


void FaceStereo::init_parameter(bool use_right_disparity, int disparity_min, int disparity_max
	, double disparity_smooth, int disparity_refine_iter, double disparity_search_range
	, bool hierarch_block_matching, int pyramid_depth, int terminate_level
	, double region_min_x, double region_max_x, double region_min_y, double region_max_y
	, double max_depth_change_ratio, double max_curv_change_ratio
	, bool auto_face_detection, std::string face_xml_file)
{
	useRightDisparity_ = use_right_disparity;
	minDisparity_ = disparity_min;
	maxDisparity_ = disparity_max;
	disp_smoothness_ = disparity_smooth;
	refine_iter_ = disparity_refine_iter;
	search_range_ = disparity_search_range;

	use_hierarchical_block_matching_ = hierarch_block_matching;
	pyramidDepth_ = pyramid_depth;
	terminate_level_ = terminate_level;

	region_min_[0] = region_min_x;
	region_max_[0] = region_max_x;
	region_min_[1] = region_min_y;
	region_max_[1] = region_max_y;

	max_depth_change_ratio_ = max_depth_change_ratio;
	max_curv_change_ratio_ = max_curv_change_ratio;

	use_face_mask_ = auto_face_detection;
	face_xml_file_ = face_xml_file;
	if (use_face_mask_)
	{
		if (!face_cascade_.load(face_xml_file_))
		{
			std::cout
				<< "***********************************************************************************\n"
				<< "*** Warning, can't load an XML file for face detection : " << face_xml_file_ << "\n"
				<< "*** Face detection option will be turned off automatically.\n"
				<< "***********************************************************************************\n"
				<< std::endl;
			use_face_mask_ = false;
		}
	}

	// Allocations
	pyramid_[0].resize(pyramidDepth_);
	pyramid_[1].resize(pyramidDepth_);
	pyramid_gray_[0].resize(pyramidDepth_);
	pyramid_gray_[1].resize(pyramidDepth_);

	// Face masking
	face_mask_[0].resize(pyramidDepth_);
	face_mask_[1].resize(pyramidDepth_);

	width_.resize(pyramidDepth_);
	height_.resize(pyramidDepth_);

	imgScale_.resize(pyramidDepth_);

	disp_min_.resize(pyramidDepth_);
	disp_max_.resize(pyramidDepth_);

	K_l_.resize(pyramidDepth_);
	K_r_.resize(pyramidDepth_);

	campose_l_.resize(pyramidDepth_);
	campose_r_.resize(pyramidDepth_);

	P_l_.resize(pyramidDepth_);
	P_r_.resize(pyramidDepth_);

	disp_l_.resize(pyramidDepth_);
	disp_r_.resize(pyramidDepth_);
	temp_disp_l_.resize(pyramidDepth_);
	temp_disp_r_.resize(pyramidDepth_);

	std::cout
		<< "==========================\n"
		<< "**** INPUT PARAMETERS ****\n"
		<< "==========================\n"
		<< ". USE_RIGHT_DISPARITY     : " << useRightDisparity_ << "\n"
		<< ". DISPARITY               : " << minDisparity_ << " ~ " << maxDisparity_ << "\n"
		<< ". DISPARITY SMOOTH        : " << disp_smoothness_ << "\n"
		<< ". DISPARITY_REFINE_ITER   : " << refine_iter_ << "\n"
		<< ". DISPARITY_SEARCH_RANGE  : " << search_range_ << "\n"
		<< "--------------------------\n"
		<< ". HIERARCH_BLOCK_MATCHING : " << use_hierarchical_block_matching_ << "\n"
		<< ". PYRAMID_DEPTH           : " << pyramidDepth_ << " > " << terminate_level_ << "\n"
		<< "--------------------------\n"
		<< ". INTEREST REGION         : X[" << region_min_[0] << ", " << region_max_[0] << "]\n"
		<< "                            Y[" << region_min_[1] << ", " << region_max_[1] << "]\n"
		<< "--------------------------\n"
		<< ". MAX_DEPTH_CHANGE_RATIO  : " << max_depth_change_ratio_ << "\n"
		<< ". MAX_CURV_CHANGE_RATIO   : " << max_curv_change_ratio_ << "\n"
		<< std::endl;
}

void FaceStereo::init_calibration(Eigen::Matrix4d cam_pos_l, Eigen::Matrix4d cam_pos_r
	, Eigen::Matrix3d k_l, Eigen::Matrix3d k_r
	, std::vector<Eigen::Vector2d> corner2d_l, std::vector<Eigen::Vector2d> corner2d_r
	, std::vector<Eigen::Vector3d> corner3d)
{
	K_[0] = k_l;
	K_[1] = k_r;
	campose_[0] = cam_pos_l;
	campose_[1] = cam_pos_r;
	cp_l_ = corner2d_l;
	cp_r_ = corner2d_r;
	cp_ = corner3d;

	std::cout
		<< "==========================\n"
		<< "**** CALIBRATION DATA ****\n"
		<< "==========================\n"
		<< ". Left Cam Pose           :\n" << campose_[0] << "\n"
		<< ". Right Cam Pose          :\n" << campose_[1] << "\n"
		<< ". Left K                  :\n" << K_[0] << "\n"
		<< ". Right K                 :\n" << K_[1] << "\n";
}

void FaceStereo::reconstruct_stereo_image(std::string intermediate_path, int frame, int paddings
	, bool rectify_images, cv::Mat image_l, cv::Mat image_r
	, std::vector<Eigen::Vector3d>* point_data, std::vector<uchar>* color_data)
{
	folder_ = intermediate_path;
	currentFrame_ = frame;
	paddings_ = paddings;

	prefix_ = "Reconst_h";
	numFrames_ = 1;

	inputImg_[0] = image_l;
	inputImg_[1] = image_r;

	rectify_image(rectify_images);
	make_image_pyramid();
	

	std::cout
		<< "============================\n"
		<< "*    3D Reconstruction     *\n"
		<< "Frame:" << currentFrame_ << "\n"
		<< "============================"
		<< std::endl;

	int baseImage = useRightDisparity_ ? 0 : 1;
	for (int i = pyramidDepth_ - 1; i >= 0; --i)
	{
		compute_disparity(i);

		set_correspondences_from_disparity(i, baseImage);

		// Correspondence points in terms of normalized coordinates
		std::vector< Eigen::Vector3d > np, nq;
		Eigen::Matrix3d K1 = K_l_[i];
		Eigen::Matrix3d K2 = K_r_[i];

		convert_image_pts_normalized_coord(x1_, K1, np);
		convert_image_pts_normalized_coord(x2_, K2, nq);

		Eigen::MatrixXd P1(3, 4), P2(3, 4);
		P1.block<3, 3>(0, 0).setIdentity();
		P1.block<3, 1>(0, 3).setZero();

		double b = (campose_[0].block<3, 1>(0, 3) - campose_[1].block<3, 1>(0, 3)).norm();
		Eigen::Vector3d t(b, 0, 0);
		P2.block<3, 3>(0, 0).setIdentity();
		P2.block<3, 1>(0, 3) = -t;

		auto npSize = np.size();
		std::vector<Eigen::Vector3d> X(npSize);


		int threadSize = std::thread::hardware_concurrency();
		//std::cout << "Thread Size : " << threadSize << std::endl;
		std::vector< std::thread> threads(threadSize);
		std::mutex mtx;
		int syncBlock = 0;
		for (int n = 0; n < threadSize; ++n)
		{
			threads[n] = std::thread([&](int tid, int threadNum)
			{
				int start = (int)((tid / (float)threadNum)*npSize);
				int end = (int)(((tid + 1) / (float)threadNum)*npSize);
				for (int i = start; i < end; ++i)
				{
					linear_triangulation(np[i].data(), nq[i].data(), P1.data(), P2.data(), X[i].data());
					X[i](1) *= -1.0;
					X[i](2) *= -1.0;
				}
			}
			, i, threadSize);
		}
		for (int n = 0; n < threadSize; ++n) {
			threads[n].join();
		}



		// Set texture
		// NOTE : Texture is obtained from the base image. 
		struct BGR { unsigned char b, g, r; };

		cv::Mat * baseImg = &(pyramid_[0][i]);
		Eigen::Vector2d* x = x1_.data();
		if (baseImage == 1)
		{
			baseImg = &(pyramid_[1][i]);
			x = x2_.data();
		}

		BGR* imgData = (BGR*)(baseImg->data);

		std::vector<BGR> pt_col(x1_.size());
		const int w = baseImg->cols;
		const int h = baseImg->rows;

		for (int i = 0; i < x1_.size(); ++i)
		{
			const int px = (int)x[i](0);
			const int py = (int)x[i](1);
			pt_col[i] = imgData[w*py + px];
		}


		// std::ostringstream filename;
		// if (output_format_ == 0) // pcd
		// {
		// 	filename << folder_ << "reconst_h" << i + 1 << "_" << std::setw(paddings_) << std::setfill('0') << currentFrame_ << ".pcd";
		// 	write_point_cloud_in_pcd(filename.str().c_str(), (int)X.size(), X.data()->data(), (unsigned char*)pt_col.data());
		// }
		// else // ply
		// {
		// 	filename << folder_ << "reconst_h" << i + 1 << "_" << std::setw(paddings_) << std::setfill('0') << currentFrame_ << ".ply";
		// 	write_point_cloud_in_ply(filename.str().c_str(), (int)X.size(), X.data()->data(), (unsigned char*)pt_col.data());
		// }

		// Export the depth map image
		std::ostringstream filename2;
		filename2 << folder_ << "depth_h" << i + 1 << "_" << std::setw(paddings_) << std::setfill('0') << currentFrame_ << +".png";
		export_depth_map_image(filename2.str().c_str(), (int)X.size(), X.data()->data(), width_[i], height_[i], corr_pid_.data()->data());

		// Export the disparity map image in RGBA
		std::ostringstream dispFileName;
		dispFileName << folder_ << "disp32_h" << i + 1 << "_" << std::setw(paddings_) << std::setfill('0') << currentFrame_ << +".png";
		export_disparity_map_image(dispFileName.str().c_str(), width_[i], height_[i], disp_l_[i].data());

		if (i + 1 <= terminate_level_)
		{
			for (int t = 0; t < X.size(); ++t)
			{
				(*point_data).push_back(X[t]);
			}

			(*color_data).clear();
			for (int t = 0; t < pt_col.size(); ++t)
			{
				(*color_data).push_back(pt_col[t].b);
				(*color_data).push_back(pt_col[t].g);
				(*color_data).push_back(pt_col[t].r);
			}
			// (*color_data).push_back((uchar)pt_col.data());
			break;
		}
	}
}

void FaceStereo::rectify_image(bool rectify_images)
{
	if (rectify_images)
	{
		Eigen::MatrixXd P1(3, 4);
		Eigen::Vector4d camCenter[2];
		camCenter[0] = campose_[0].block<4, 1>(0, 3);
		camCenter[1] = campose_[1].block<4, 1>(0, 3);

		double B = (camCenter[0] - camCenter[1]).norm();

		Eigen::Matrix4d T1;
		T1 = campose_[0].inverse();

		P1.block<3, 4>(0, 0) = T1.block<3, 4>(0, 0);
		P1 = K_[0] * P1;

		Eigen::Matrix3d H;
		int w = inputImg_[0].cols;
		int h = inputImg_[0].rows;
		int type = inputImg_[0].type();

		rectInputImg_[0] = cv::Mat(h, w, type);
		rectInputImg_[1] = cv::Mat(h, w, type);

		// NOTE : Left image is unchanged during the recrification
		inputImg_[0].copyTo(rectInputImg_[0]);



		Eigen::Matrix3d tK2;
		Eigen::MatrixXd tP2(3, 4);
		rectify(K_[0].data(), P1.data(), B, (int)cp_.size()
			, cp_r_.data()->data(), cp_.data()->data()
			, tK2.data(), tP2.data(), H.data(), w, h, type
			, inputImg_[0].data, inputImg_[1].data, rectInputImg_[1].data, folder_);

		H_rect_ = H;

		// Now set camera matrix, projection matrix and camera pose for image pyramid
		K_l_[0] = K_[0];
		K_r_[0] = tK2;
		//std::cout << "K_r[0]=\n" << K_r_[0] << std::endl;

		P_l_[0] = P1;
		P_r_[0] = tP2;
		//std::cout << "P_r[0]=\n" << P_r_[0] << std::endl;

		campose_l_[0] = campose_[0];
		campose_r_[0].setIdentity();
		campose_r_[0](0, 3) = B;

		// Export the homography to disk
		std::ofstream out_file(folder_ + "rect_homography.txt");
		out_file << H;
		out_file.close();
	}
	else
	{
		inputImg_[0].copyTo(rectInputImg_[0]);
		inputImg_[1].copyTo(rectInputImg_[1]);

		H_rect_.setIdentity();

		K_l_[0] = K_[0];
		K_r_[0] = K_[1];

		P_l_[0] = K_[0] * (campose_[0].inverse().block<3, 4>(0, 0));
		P_r_[0] = K_[1] * (campose_[1].inverse().block<3, 4>(0, 0));

		campose_l_[0] = campose_[0];
		campose_r_[0] = campose_[1];

		// Export the homography to disk
		std::ofstream out_file(folder_ + "rect_homography.txt");
		out_file << H_rect_;
		out_file.close();
	}
}

void FaceStereo::make_image_pyramid()
{
	rectInputImg_[0].copyTo(pyramid_[0][0]);
	rectInputImg_[1].copyTo(pyramid_[1][0]);

	cv::cvtColor(pyramid_[0][0], pyramid_gray_[0][0], cv::COLOR_BGR2GRAY);
	cv::cvtColor(pyramid_[1][0], pyramid_gray_[1][0], cv::COLOR_BGR2GRAY);

	width_[0] = rectInputImg_[0].cols;
	height_[0] = rectInputImg_[0].rows;
	imgScale_[0] = 1.0;
	disp_min_[0] = minDisparity_;
	disp_max_[0] = maxDisparity_;

	//=====================================================================================
	// NOTE : K_l, K_r are for rectified images. 
	// For level 0, they were set already in rectification(the same for P_l, P_r, campose_l, campose_r).
	// The camera martix for the rectified right image is equal to the left image.
	//=====================================================================================

	cv::imwrite(folder_ + std::string("pyramid_left_") + std::to_string(0) + ".bmp", pyramid_[0][0]);
	cv::imwrite(folder_ + std::string("pyramid_right_") + std::to_string(0) + ".bmp", pyramid_[1][0]);
	cv::imwrite(folder_ + std::string("gray_left_") + std::to_string(0) + ".bmp", pyramid_gray_[0][0]);
	cv::imwrite(folder_ + std::string("gray_right_") + std::to_string(0) + ".bmp", pyramid_gray_[1][0]);

	cv::Mat* src[2];
	src[0] = &pyramid_[0][0];
	src[1] = &pyramid_[1][0];

	cv::Mat* srcGray[2];
	srcGray[0] = &pyramid_gray_[0][0];
	srcGray[1] = &pyramid_gray_[1][0];

	cv::Mat* mask[2];
	if (use_face_mask_)
	{
		mask[0] = &face_mask_[0][0];
		mask[1] = &face_mask_[1][0];
	}

	// Scale down by factor 2
	for (int i = 1; i < pyramidDepth_; ++i)
	{
		cv::pyrDown(*src[0], pyramid_[0][i], cv::Size(src[0]->cols / 2, src[0]->rows / 2));
		cv::pyrDown(*src[1], pyramid_[1][i], cv::Size(src[1]->cols / 2, src[1]->rows / 2));
		cv::pyrDown(*srcGray[0], pyramid_gray_[0][i], cv::Size(srcGray[0]->cols / 2, srcGray[0]->rows / 2));
		cv::pyrDown(*srcGray[1], pyramid_gray_[1][i], cv::Size(srcGray[1]->cols / 2, srcGray[1]->rows / 2));

		src[0] = &pyramid_[0][i];
		src[1] = &pyramid_[1][i];
		srcGray[0] = &pyramid_gray_[0][i];
		srcGray[1] = &pyramid_gray_[1][i];

		cv::imwrite(folder_ + std::string("pyramid_left_") + std::to_string(i) + ".bmp", pyramid_[0][i]);
		cv::imwrite(folder_ + std::string("pyramid_right_") + std::to_string(i) + ".bmp", pyramid_[1][i]);
		cv::imwrite(folder_ + std::string("gray_left_") + std::to_string(i) + ".bmp", pyramid_gray_[0][i]);
		cv::imwrite(folder_ + std::string("gray_right_") + std::to_string(i) + ".bmp", pyramid_gray_[1][i]);

		width_[i] = pyramid_[0][i].cols;
		height_[i] = pyramid_[0][i].rows;

		imgScale_[i] = 0.5*imgScale_[i - 1];

		disp_min_[i] = (int)(minDisparity_ * imgScale_[i]);
		disp_max_[i] = (int)(maxDisparity_ * imgScale_[i]);

		Eigen::Matrix3d S;
		S.setIdentity();
		S(0, 0) = S(1, 1) = imgScale_[i];
		K_l_[i] = S * K_l_[0];
		K_r_[i] = S * K_r_[0];

		campose_l_[i] = campose_l_[0];
		campose_r_[i] = campose_r_[0];

		Eigen::Matrix4d T1, T2;
		T1 = campose_l_[i].inverse();
		T2 = campose_r_[i].inverse();

		P_l_[i].block<3, 4>(0, 0) = T1.block<3, 4>(0, 0);
		P_r_[i].block<3, 4>(0, 0) = T2.block<3, 4>(0, 0);
	}
}
