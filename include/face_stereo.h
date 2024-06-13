#ifndef MG_FACE_STEREO_RECONST_H_
#define MG_FACE_STEREO_RECONST_H_

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <mutex>

class FaceStereo
{
public:
    FaceStereo() : numFrames_(1), rectified_input_image_(true), 
		useRightDisparity_(false),
		maxDisparity_(1024), minDisparity_(128),
        pyramidDepth_(4), terminate_level_(0),
        use_hierarchical_block_matching_(true), max_curv_change_ratio_(1.2),
        output_format_(0),
        use_face_mask_(true), minSatur_(125), saturDilateSize_(10),
        face_xml_file_("haarcascade_frontalface_alt.xml")
    {
        region_min_[0] = region_min_[1] = 0;
        region_max_[0] = region_max_[1] = 1;

        K_[0].setIdentity();
        K_[1].setIdentity();
        campose_[0].setIdentity();
        campose_[1].setIdentity();
    }

	bool init(std::string & leftImageFile, std::string & rightImageFile,
		int numFrames, bool rectified, int output_format, bool useRightDisparity, 
		int max_disparity, int min_disparity,
		double disp_smooth, int disp_iter, double disp_search_range, int pyramid_depth, int termiate_level, bool hierarch_block_matching,
		double max_depth_change_ratio, double max_curv_change_ratio,
		int region_x_min, int region_x_max, int region_y_min, int region_y_max, int auto_face_detect, std::string & face_xml_file,
		std::string  left_K_file, std::string  right_K_file,
		std::string  left_campose_file, std::string  right_campose_file,
		std::string  corner_reconst_file, std::string  corner_left_file, std::string corner_right_file, std::string folder, std::string prefix);

    bool read_input_file(const char* filename);
    bool loadStereoImages(int frame);
    void rectify_images();
    void construct_image_pyramid();

    bool get_face_mask(cv::Mat& srcImg, cv::Mat& mask);

    void compute_disparity(int level);
    void compute_raw_disparity_block_matching(int level, int baseImage);
    void compute_raw_disparity_block_matching_2(int level, int baseImage);
	void set_raw_disparity_from_the_other(int level, int baseImage);
	//void extend_raw_disparity_from_the_other(int level);
    void stereo_matching_2(int w, int h, int type, unsigned char* baseImgData, unsigned char* secondImgData,
        int blockSize, int minDisparity, int maxDisparity, double* disp, const char* disp_file_name);
    void check_disparity_smoothness(int level, std::vector<int>& index);
    void check_disparity_uniqueness(int level, std::vector<int>& index);
    void check_disparity_ordering(int level, std::vector<int>& index);
    void check_level_consistency(int level, std::vector<int>& index);
    void expand_lower_level_disparity(int level, double * disp_expanded);
    void hierarchical_block_matching(int level, double* disp_expanded);
	void hierarchical_block_matching_multi_thread(int level, double * disp_expanded);
	void hierarchical_block_matching_block(int level, double * disp_expanded, int * syncBlock, int tid, int threadNum, std::mutex* mtx);
    void rematch(int level);
	inline void normalized_intensity_vector_block(int w, int h, unsigned char* imgData, int pid, Eigen::VectorXd* nf, Eigen::VectorXd* f, Eigen::VectorXd* temp);
    void normalized_intensity_vector(int w, int h, unsigned char* imgData, int pid, Eigen::VectorXd& nf);
    double ncc(const Eigen::VectorXd& nf1, const Eigen::VectorXd& nf2);
    void unnormalized_window_vector(int w, int h, double* data, int pid, Eigen::VectorXd& f);
    void refinement_iteration(int level, int numIter);
    void refinement_iteration_2(int level, int numIter);
    void compute_disparity_photometric_consistency(int level);
    void compute_disparity_surf_consistency(int level, double* disp_surf);
    void compute_disparity_surf_consistency(int level, double* d, double* disp_surf);
    void update_disparity_photemetric_surface(int level, double ws, double* disp_surf);
    void update_disparity_photemetric_surface(int level, double* d_p, double ws, double* disp_surf, double* disp_updated);
	void update_disparity_photemetric_surface_use_multithread(int level, double ws, double* disp_surf);
	void update_disparity_photemetric_surface_block(int level, double ws, double * disp_surf, int tid, int threadNum);

    void compute_disparity_2(int level);

    void apply_face_mask_to_disparity(int level);

    void set_correspondences_from_disparity(int level, int baseImage);
    void convert_image_pts_normalized_coord(std::vector<Eigen::Vector2d>& x, Eigen::Matrix3d K,
        std::vector<Eigen::Vector3d>& nx);
    void write_depth_map_mesh(const char* filename, int numPts, double* p, int w, int h, int* pt_pixel_id);
    void export_depth_map_image(const char* filename, int numPts, double* p, int w, int h, int* pt_pixel_id);
	void export_disparity_map_image(const char* filename, int w, int h, double* disparity);
    void reconstruction(int level, int baseImage, std::string folder="");
	void reconstruction_multi_thread(int level, int baseImage, std::string folder = "");
    void reconstruction_geom_corr(int level, int baseImage);

public: // JK
	void init_parameter(bool use_right_disparity, int disparity_min, int disparity_max
		, double disparity_smooth, int disparity_refine_iter, double disparity_search_range
		, bool hierarch_block_matching, int pyramid_depth, int terminate_level
		, double region_min_x, double region_max_x, double region_min_y, double region_max_y
		, double max_depth_change_ratio, double max_curv_change_ratio
		, bool auto_face_detection, std::string face_xml_file);

	void init_calibration(Eigen::Matrix4d cam_pos_l, Eigen::Matrix4d cam_pos_r
		, Eigen::Matrix3d k_l, Eigen::Matrix3d k_r
		, std::vector<Eigen::Vector2d> corner2d_l, std::vector<Eigen::Vector2d> corner2d_r
		, std::vector<Eigen::Vector3d> corner3d);

	void reconstruct_stereo_image(std::string intermediate_path, int frame, int paddings
		, bool rectify_images, cv::Mat image_l, cv::Mat image_r
		, std::vector<Eigen::Vector3d>* point_data, std::vector<uchar>* color_data);

private: // JK

	void rectify_image(bool rectify_images);
	void make_image_pyramid();



public:
    cv::Mat inputImg_[2], rectInputImg_[2];           // 0: left, 1: right
    int numFrames_, startFrame_, currentFrame_;
    std::string leftImgPrefix_, rightImgPrefix_;
    int paddings_;
    std::string imgExt_;
    bool rectified_input_image_;

    int output_format_;

    std::vector<cv::Mat> pyramid_[2], pyramid_gray_[2];
    std::vector<int> width_, height_;
    std::vector<double> imgScale_;

    std::vector<cv::Mat> face_mask_[2];
    bool use_face_mask_;
    int minSatur_, saturDilateSize_;
    std::string face_xml_file_;
    cv::CascadeClassifier face_cascade_;

    int maxDisparity_;
    int minDisparity_;
    double disp_smoothness_;
    int refine_iter_;
    double search_range_;
    std::vector<int> disp_min_, disp_max_;
    
    bool use_hierarchical_block_matching_;
    double max_depth_change_ratio_;
    double max_curv_change_ratio_;

    std::vector< std::vector<double> > disp_l_, disp_r_;
    std::vector< std::vector<double> > temp_disp_l_, temp_disp_r_;
    //std::vector< std::vector<double> > disp_surf_;
	bool useRightDisparity_;

    Eigen::Matrix3d H_rect_;

    std::vector< Eigen::Vector2d > x1_, x2_; // correspondences
    std::vector< Eigen::Vector2i > corr_pid_;

    int pyramidDepth_;
    int terminate_level_;

    int region_min_[2], region_max_[2];

    Eigen::Matrix3d K_[2];   // for original input images(may differ from K_l_[0], K_r_[0])
    Eigen::Matrix4d campose_[2]; // for original input images(may differ from campose_l_[0], campose_r_[0])

    std::vector< Eigen::Matrix3d > K_l_, K_r_;              // for rectified image pyramid
    std::vector< Eigen::Matrix<double, 3, 4> > P_l_, P_r_;  // for rectified image pyramid
    std::vector< Eigen::Matrix4d> campose_l_, campose_r_;    // for rectified image pyramid

    // Known reconstructed points for rectification
    // At the moment, corner points in check images used in camera calibration
    std::vector< Eigen::Vector3d > cp_;             // 3D reconstructed points
    std::vector< Eigen::Vector2d > cp_l_, cp_r_;    // projection points in images

	std::string folder_, prefix_;
};

#endif