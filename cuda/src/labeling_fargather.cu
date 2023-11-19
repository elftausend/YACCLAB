
#include <opencv2/cudafeatures2d.hpp>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "labeling_algorithms.h"
#include "register.h"

#define BLOCK_SIZE 32   // this must be multiple of the warp size (leave it to 32)
#define PATCH_SIZE (BLOCK_SIZE + 2)

using namespace cv;
using namespace std;

namespace {
__global__ void Init(const cuda::PtrStepSzb img, cuda::PtrStepSzi labels, ushort4* links) {
    const unsigned r = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    const unsigned c = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const unsigned labels_index = r * (labels.step / labels.elem_size) + c;
    const bool in_limits = r < img.rows && c < img.cols;


    // const unsigned img_patch_index = (threadIdx.y + 1) * PATCH_SIZE + threadIdx.x + 1;
    // const unsigned local_linear_index = threadIdx.y * BLOCK_SIZE + threadIdx.x;

    // __shared__ unsigned char img_patch[PATCH_SIZE * PATCH_SIZE];


    // // Load 34 x 34 matrix from input image

    // // Convert local_linear_index to coordinates of the 34 x 34 matrix

    // // Round 1
    // const int patch_r1 = local_linear_index / PATCH_SIZE;
    // const int patch_c1 = local_linear_index % PATCH_SIZE;
    // const int patch_img_r1 = blockIdx.y * BLOCK_SIZE - 1 + patch_r1;
    // const int patch_img_c1 = blockIdx.x * BLOCK_SIZE - 1 + patch_c1;
    // const int patch_img_index1 = patch_img_r1 * img.step + patch_img_c1;
    // const bool patch_in_limits1 = patch_img_r1 >= 0 && patch_img_c1 >= 0 && patch_img_r1 < img.rows&& patch_img_c1 < img.cols;
    // img_patch[patch_r1 * PATCH_SIZE + patch_c1] = patch_in_limits1 ? img[patch_img_index1] : 0;

    // // Round 2
    // const int patch_r2 = (local_linear_index + BLOCK_SIZE * BLOCK_SIZE) / PATCH_SIZE;
    // const int patch_c2 = (local_linear_index + BLOCK_SIZE * BLOCK_SIZE) % PATCH_SIZE;
    // if (patch_r2 < PATCH_SIZE) {
    //     const int patch_img_r2 = blockIdx.y * BLOCK_SIZE - 1 + patch_r2;
    //     const int patch_img_c2 = blockIdx.x * BLOCK_SIZE - 1 + patch_c2;
    //     const int patch_img_index2 = patch_img_r2 * img.step + patch_img_c2;
    //     const bool patch_in_limits2 = patch_img_r2 >= 0 && patch_img_c2 >= 0 && patch_img_r2 < img.rows&& patch_img_c2 < img.cols;
    //     img_patch[patch_r2 * PATCH_SIZE + patch_c2] = patch_in_limits2 ? img[patch_img_index2] : 0;
    // }

    // __syncthreads();

    
    // if (in_limits) {
    //     unsigned label = 0;
    //     if (img_patch[img_patch_index]) {
    //         label = labels_index + 1;
    //     }

    //     labels[labels_index] = label;
    // }

    if (!in_limits) {
        return;
    }

    labels[labels_index] = labels_index + 1;

    
    int step_width = img.step / img.elem_size;
    int threshold = 1;
    unsigned int connections = 0;

    unsigned char currentPixel = img[r * step_width + c];
    ushort4 currentLink = make_ushort4(0, 0, 0, 0);

    int height = img.rows;
    // label |= connections;

    // right
    for (int i = c + 1; i < img.cols; i++) {
        // break; 
        int rightPixel = (int)img[r * step_width + i];
        if (!(abs(rightPixel - (int)currentPixel) < threshold)) {
            // printf("not in threshold");
            break;
        }

        // printf("current pixel: %d, right pixel: %d \n", currentPixel, rightPixel);
        unsigned short farRightLink = (unsigned short) (i - c);
        // printf("right link: %d \n", farRightLink);
        currentLink.x = farRightLink;
    }

    // down 
    for (int i = r + 1; i < height; i++) { 
        // break;
        int rightPixel = (int)img[i * step_width + c];
        // printf("%d \n", rightPixel.x);
        if (!(abs(rightPixel - (int)currentPixel) < threshold)) {
            break;
        }
        unsigned short farDownLabel = (unsigned short) (i - r);
        currentLink.y = farDownLabel;
    }
    
    // left
    for (int i = c - 1; i >= 0; i--) {
        // break; 
        int leftPixel = (int)img[r * step_width + i];
        if (!(abs(leftPixel - (int)currentPixel) < threshold)) {
            break;
        }
        unsigned short farLeftLabel = (unsigned short) (c - i);
        // printf("right link: %d \n", farLeftLabel);
        currentLink.z = farLeftLabel;
    }
   
    // up 
    for (int i = r - 1; i >= 0; i--) { 
        // break;
        int rightPixel = (int)img[i * step_width + c];
        // printf("%d \n", rightPixel.x);
        if (!(abs(rightPixel - (int)currentPixel) < threshold)) {
            break;
        }
        unsigned short farDownLabel = (unsigned short) (r - i);
        currentLink.w = farDownLabel;
    }

    // currentLink = make_ushort4(min(1, currentLink.x), min(1, currentLink.y), min(1, currentLink.z), min(1, currentLink.w));

    links[r * img.cols + c] = currentLink;

    // printf("label: %u \n", label);
    // labels[labelIdx] = label;
}

// labels do not contain connection information (rasmusson)
__global__ void Propagate(cuda::PtrStepSzi input, ushort4* links, int* hasUpdated) {
    unsigned int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    unsigned int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    if (x >= input.cols || y >= input.rows) {
        return;
    }
    int step_width = input.step / input.elem_size;

    int outIdx = y * step_width + x;
    
    unsigned int currentLabel = input[outIdx];

    ushort4 currentLink = links[y * input.cols + x];
    unsigned int farRightLabel = input[outIdx + (int) currentLink.x];

    if (farRightLabel > currentLabel) {
        currentLabel = farRightLabel;
        *hasUpdated = 1;
    }

    unsigned int farDownLabel = input[(y + currentLink.y) * step_width + x];

    if (farDownLabel > currentLabel) {
        currentLabel = farDownLabel;
        *hasUpdated = 1;
    }    

    unsigned int farLeftLabel = input[outIdx - currentLink.z];

    if (farLeftLabel > currentLabel) {
        currentLabel = farLeftLabel;
        *hasUpdated = 1;
    }
    
    
    unsigned int farUpLabel = input[(y - currentLink.w) * step_width + x];

    if (farUpLabel > currentLabel) {
        currentLabel = farUpLabel;
        *hasUpdated = 1;
    }
    
    int leftLabel = input[outIdx - min(1, currentLink.z)];

    if (leftLabel > currentLabel) {
        currentLabel = leftLabel;
        *hasUpdated = 1;
    }

    int upLabel = input[(y - min(1, currentLink.w)) * step_width + x];

    if (upLabel > currentLabel) {
        currentLabel = upLabel;
        *hasUpdated = 1;
    }

    input[outIdx] = currentLabel;
}

/*__global__ void End(cuda::PtrStepSzi labels) {

    unsigned global_row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    unsigned global_col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    unsigned labels_index = global_row * (labels.step / labels.elem_size) + global_col;

    if (global_row < labels.rows && global_col < labels.cols) {
        labels.data[labels_index] &= 0x0FFFFFFF;
    }
}*/
}


class FARGATHER: public GpuLabeling2D<Connectivity2D::CONN_8> {
private:
    dim3 grid_size_;
    dim3 block_size_;
    char* d_changed_ptr_;
    ushort4 links;
public:
    FARGATHER() {}

    void PerformLabeling() {
        // printf("start");
        d_img_labels_.create(d_img_.size(), CV_32SC1);
        grid_size_ = dim3((d_img_.cols + BLOCK_SIZE - 1) / BLOCK_SIZE, (d_img_.rows + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
        block_size_ = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);

        unsigned char* links;
        cudaMalloc(&links, d_img_.rows * d_img_.cols * 4 * 2);
        ushort4* uslinks = (ushort4*) links;
 
        Init << <grid_size_, block_size_ >> > (d_img_, d_img_labels_, uslinks);

        // unsigned short* cpu_links = (unsigned short*) malloc(d_img_.rows * d_img_.cols * 4 * 2);
        // cudaMemcpy(cpu_links, uslinks, d_img_.rows * d_img_.cols * 4 * 2, cudaMemcpyDeviceToHost);
        // for (int r = 0; r < d_img_.rows; r++) {
        //     for (int c = 0; c < d_img_.cols * 4; c++) {
        //         // printf("%d, ", cpu_links[r * d_img_.cols + c]);
        //     }
        //     // printf("\n");
        // }
        

        char changed = 1;
        int* d_changed_ptr;
        cudaMalloc(&d_changed_ptr, 1);

        // malloc()

        // cudaFree(d_changed_ptr);
        // cudaFree(links);
        // cudaDeviceSynchronize();
        // return;
        // for (int i = 0; i < 10; i++) {
        //     Propagate << <grid_size_, block_size_ >> > (d_img_labels_, uslinks, d_changed_ptr);
        // }
        while (changed) {     
            changed = 0;
            cudaMemset(d_changed_ptr, 0, 1);

            Propagate << <grid_size_, block_size_ >> > (d_img_labels_, uslinks, d_changed_ptr);

            cudaDeviceSynchronize();
            cudaMemcpy(&changed, d_changed_ptr, 1, cudaMemcpyDeviceToHost);
        }
        // printf("finished");
        // End << <grid_size_, block_size_ >> > (d_img_labels_);
        
        cudaFree(d_changed_ptr);
        cudaFree(links);
        cudaDeviceSynchronize();
    }
};

REGISTER_LABELING(FARGATHER);
