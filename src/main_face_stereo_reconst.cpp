#include <iostream>
#include <fstream>

#include "face_stereo.h"

int main(int argc, char** argv)
{
    std::cout
        << "=============================================================================" << "\n"
        << "               3D FACE RECONSTRUCTION WITH STEREO-IMAGES " << "\n"
        << "=============================================================================" << "\n"
        << std::endl;

    if (argc <= 1)
    {
        std::cout << "Usage : face_stereo_reconst.exe input_file" << std::endl;
        return -1;
    }

    FaceStereo faceStereo;

    // Read input file
    faceStereo.read_input_file(argv[1]);
    std::cout << faceStereo.startFrame_ << ", " << faceStereo.paddings_ << std::endl;

    int numFrames = faceStereo.numFrames_;

    for (int frame = 0; frame < numFrames; ++frame)
    {
        int currentFrame = faceStereo.startFrame_ + frame;
        
        std::cout << "\n" << "FRAME = " << currentFrame << std::endl;

        if (!faceStereo.loadStereoImages(currentFrame))
        {
            std::cout << "- WARNING! Skipped this frame!!!!" << std::endl;
            continue;
        }

        // Rectify if necessary
        faceStereo.rectify_images();

        // Prepare image pyramid
        faceStereo.construct_image_pyramid();

        int pyrDepth = faceStereo.pyramidDepth_;

        // Process the image pyramid in bottom-up fashion
        for (int i = pyrDepth - 1; i >= 0; --i)
        {
            // Compute disparity
            faceStereo.compute_disparity(i);
            //faceStereo.compute_disparity_2(i);

            // Triangulation for reconstruction
            //faceStereo.set_correspondences_from_disparity(i, 0);
            faceStereo.reconstruction(i, 0);

            if (i == faceStereo.terminate_level_) break;
        }//*/

    }
    std::cout << "\n" 
        << "============================" << "\n"
        << "*** DONE !!!" << "\n"
        << "============================" << std::endl;
    

    return 0;
}