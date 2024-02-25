
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

__global__ void labelWithSharedLinks(const cuda::PtrStepSzb img, cuda::PtrStepSzi labels, ushort4* links) {
    const unsigned r = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    const unsigned c = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    
    const bool in_limits = r < img.rows && c < img.cols;

    if (!in_limits) {
        return;
    }
    
    const unsigned labels_index = r * (labels.step / labels.elem_size) + c;

    int step_width = img.step / img.elem_size;
    int threshold = 1;
    unsigned int connections = 0;

    unsigned char currentPixel = img[r * step_width + c];
    ushort4 currentLink = make_ushort4(0, 0, 0, 0);

    int height = img.rows;
     
    __shared__ ushort4 sharedLinks[32][32];

     // right 
    if (c < img.cols-1) {
        unsigned char rightPixel = img[r * step_width + c + 1];
            
        if (abs(rightPixel - currentPixel) < threshold) {    
            currentLink.x = 1;
        }
    }

    // left
    if (c > 0) {
        unsigned char leftPixel = img[r * step_width + c - 1];

        if (abs(leftPixel - currentPixel) < threshold) {    
            currentLink.z = 1;
        }
    }

    if (r < height -1) {
        // down 
        unsigned char downPixel = img[(r + 1) * step_width + c];

        if (abs(downPixel - currentPixel) < threshold) {    
            currentLink.y = 1;
        }
    }

    if (r > 0) { 
        // up
        unsigned char upPixel = img[(r - 1) * step_width + c];

        if (abs(upPixel - currentPixel) < threshold) {    
            currentLink.w = 1;
        }
    }


    sharedLinks[threadIdx.y][threadIdx.x] = currentLink;
    __syncthreads();

    for (int i=0; i<5; i++) {
        if (threadIdx.x + currentLink.x < 32) {
            // right
            currentLink.x += sharedLinks[threadIdx.y][threadIdx.x + currentLink.x].x;
        }

        if (threadIdx.y + currentLink.y < 32) {
            // down
            currentLink.y += sharedLinks[threadIdx.y + currentLink.y][threadIdx.x].y;
        }

        if ((int)threadIdx.x - (int)currentLink.z >= 0) {
            // left
            currentLink.z += sharedLinks[threadIdx.y][threadIdx.x - currentLink.z].z;
        }

        if ((int)threadIdx.y - (int)currentLink.w >= 0) {
            // up
            currentLink.w += sharedLinks[threadIdx.y - currentLink.w][threadIdx.x].w;
        }

        sharedLinks[threadIdx.y][threadIdx.x] = currentLink;
        __syncthreads();
    }



    links[r * img.cols + c] = sharedLinks[threadIdx.y][threadIdx.x];
    
    labels[labels_index] = labels_index;
}


__global__ void setRootLabelIter(ushort4* links, cuda::PtrStepSzi labels, unsigned char* rootCandidates) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= labels.cols || y >= labels.rows) {
        return;
    } 

    int step_width = labels.step / labels.elem_size;
    int outIdx = y * step_width + x;
    int height = labels.rows;
    
    unsigned int currentLabel = labels[outIdx];

    ushort4 currentLink = links[outIdx];


    unsigned int farRightIdx = outIdx + (int) currentLink.x;
    unsigned int farDownIdx = (y + currentLink.y) * step_width + x;
    // unsigned int farLeftIdx = outIdx - currentLink.z;
    // unsigned int farUpIdx = (y - currentLink.w) * width + x;
    
    if (rootCandidates[y * labels.cols + x + (int) currentLink.x]) {
        labels[outIdx] = labels[farRightIdx];
    }

    if (rootCandidates[(y + currentLink.y) * labels.cols + x]) {
        labels[outIdx] = labels[farDownIdx];
    }
}

__global__ void globalizeLinksVertical(ushort4* links, int active_yd, int active_yu, int width, int height) {
    unsigned int x = x * blockDim.x + threadIdx.x;
    unsigned int yd = active_yd * blockDim.y + threadIdx.y;
    if (x >= width) {
        return;
    } 

    if (yd < height) {
        unsigned short acc_link_y = links[yd * width + x].y;
        unsigned short downMove = acc_link_y;

        while (downMove != 0) {
            downMove = links[(yd + acc_link_y) * width + x].y;
            acc_link_y += downMove;
        }
        links[yd * width + x].y = acc_link_y;
    }
}


__global__ void globalizeLinksHorizontal(ushort4* links, int active_xr, int active_xl, int width, int height) {
    unsigned int xr = active_xr * blockDim.x + threadIdx.x;
    unsigned int xl = active_xl * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (y >= height) {
        return;
    } 


    if (xl < width) {
        unsigned short acc_link_z = links[y * width + xl].z;
        unsigned short leftMove = acc_link_z;

        while (leftMove != 0) {
            leftMove = links[y * width + xl - acc_link_z].z;
            acc_link_z += leftMove;
        }
        links[y * width + xl].z = acc_link_z;
    }


    if (xr < width) {
        unsigned short acc_link_x = links[y * width + xr].x;
        unsigned short rightMove = acc_link_x;

        while (rightMove != 0) {
            rightMove = links[y * width + xr + acc_link_x].x;
            acc_link_x += rightMove;
        }
        links[y * width + xr].x = acc_link_x;
    }
}

__global__ void classifyRootCandidates(cuda::PtrStepSzi labels, ushort4* links, unsigned char* rootCandidates) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= labels.cols || y >= labels.rows) {
        return;
    }

    int step_width = labels.step / labels.elem_size;
    int height = labels.rows;
    int outIdx = y * labels.cols + x;

    int outIdxLabels = y * step_width + x; 
    // int outIdx = outIdxLabels;
    
    unsigned int currentLabel = labels[outIdxLabels];
    ushort4 currentLink = links[outIdx];

    // if (currentLink.x == 0 && currentLink.y == 0) {
    //     rootCandidates[outIdx] = 1;
    // }
    unsigned int farRightLabel = labels[outIdxLabels + (int) currentLink.x];
    unsigned int farDownLabel = labels[(y + currentLink.y) * step_width + x];

    if (farRightLabel > currentLabel || farDownLabel > currentLabel) {
        rootCandidates[outIdxLabels] = 0;
        return;
    }
    rootCandidates[outIdxLabels] = 1;
}


__global__ void Init(const cuda::PtrStepSzb img, cuda::PtrStepSzi labels, ushort4* links) {
    const unsigned r = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    const unsigned c = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const unsigned labels_index = r * (labels.step / labels.elem_size) + c;
    const bool in_limits = r < img.rows && c < img.cols;


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


__global__ void PropagateRoot(unsigned char* rootCandidates, cuda::PtrStepSzi input, ushort4* links, int* hasUpdated) {
    unsigned int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    unsigned int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    if (x >= input.cols || y >= input.rows) {
        return;
    }
    int step_width = input.step / input.elem_size;

    int outIdx = y * step_width + x;
    
    unsigned int currentLabel = input[outIdx];
    ushort4 currentLink = links[y * input.cols + x];

    unsigned int farRightIdx = outIdx + (int) currentLink.x;
    unsigned int farDownIdx = (y + currentLink.y) * step_width + x;
    unsigned int farLeftIdx = outIdx - currentLink.z;
    unsigned int farUpIdx = (y - currentLink.w) * step_width + x;
    
    unsigned int farDownLabel = input[farDownIdx];

    if (farDownLabel > currentLabel) {
        currentLabel = farDownLabel;
        *hasUpdated = 1;
        input[outIdx] =  currentLabel;

        // if a larger label was found downwards, it is (probably) larger than the rest
        return;
    }

    // if (rootCandidates[currentLabel]) {
    //     // unsigned int rootLabel = input[currentLabel - 1];
    //     // if (rootLabel > currentLabel) {
    //     currentLabel = input[currentLabel];
    //         // *hasUpdated = 1;

    //         // input[outIdx] = currentLabel;
    //         // return;

    //     // }
    // }

    unsigned int farRightLabel = input[farRightIdx];

    if (farRightLabel > currentLabel) {
        currentLabel = farRightLabel;
        *hasUpdated = 1;
        input[outIdx] =  currentLabel;
        return;
    }

    unsigned int farLeftLabel = input[farLeftIdx];

    if (farLeftLabel > currentLabel) {
        currentLabel = farLeftLabel;
        *hasUpdated = 1;
    }
    
    unsigned int farUpLabel = input[farUpIdx];

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
    // ushort4 links;
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

        unsigned char* rootCandidates;
        cudaMalloc(&rootCandidates, d_img_.rows * (d_img_.step / 1));

        labelWithSharedLinks<< <grid_size_, block_size_ >> >(d_img_, d_img_labels_, uslinks);

        int max_x = (int) ceil((float) d_img_.cols / (float) 32.0);
        for (int active_x = 0; active_x < max_x; active_x++) {
            globalizeLinksHorizontal<< <dim3(1, 256, 1), dim3(32, 32, 1) >> >(uslinks, active_x, (max_x -active_x), d_img_.cols, d_img_.rows);
        }
        
        int max_y = (int) ceil((float) d_img_.rows / (float) 32.0);
        for (int active_y = 0; active_y < max_y; active_y++) {
            globalizeLinksVertical<< <dim3(256, 1, 1), dim3(32, 32, 1) >> >(uslinks, active_y, (max_y -active_y), d_img_.cols, d_img_.rows);
        }

        classifyRootCandidates<< <grid_size_, block_size_ >> >(d_img_labels_, uslinks, rootCandidates);
 
        // Init << <grid_size_, block_size_ >> > (d_img_, d_img_labels_, uslinks);

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

            // Propagate << <grid_size_, block_size_ >> > (d_img_labels_, uslinks, d_changed_ptr);
            PropagateRoot << <grid_size_, block_size_ >> > (rootCandidates, d_img_labels_, uslinks, d_changed_ptr);

            cudaDeviceSynchronize();
            cudaMemcpy(&changed, d_changed_ptr, 1, cudaMemcpyDeviceToHost);
        }
        // printf("finished");
        // End << <grid_size_, block_size_ >> > (d_img_labels_);
        
        cudaFree(d_changed_ptr);
        cudaFree(links);
        cudaFree(rootCandidates);
        cudaDeviceSynchronize();
    }
};

REGISTER_LABELING(FARGATHER);
