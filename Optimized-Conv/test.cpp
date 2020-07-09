#include <iostream>
#include <memory.h>
#define image_h 512
#define image_w 512
#define INPUT_CHANNEL_TILE     4
#define OUTPUT_CHANNEL_TILE     8
#define INPUT_ROW_TILE          8
#define OUTPUT_ROW_TILE         (INPUT_ROW_TILE-2)
#define kernel                  3
#define kernel_size             kernel*kernel
#define pad                     1
#define stride                  1
#define image_eh                (image_h+2*pad)
#define image_ew                (image_w+2*pad)
#define INPUT_SIZE              (INPUT_CHANNEL_TILE*INPUT_ROW_TILE*image_ew)
#define OUTPUT_SIZE             (OUTPUT_CHANNEL_TILE*OUTPUT_ROW_TILE*image_w)
#define WEIGHT_SIZE             (OUTPUT_CHANNEL_TILE*INPUT_CHANNEL_TILE*kernel*kernel)

typedef float Dtype;
typedef unsigned int uint32;
void top( float  *input,  float *weight, float *output,
		int in_c,int output_c,bool relu);

#include<time.h>
#define kernel					3
#define pad						1
#define stride					1
typedef float Dtype;
using namespace std;

typedef unsigned int uint32;
void sw_conv(Dtype *input, Dtype *filter, Dtype*output, int in_c,int output_c){
		int out_h = image_h;
		int out_w = image_w;
		output_channel:
		for (int m = 0; m < output_c; m++){
			for (int h = 0; h < out_h; h++){
				for (int w = 0; w < out_w; w++){
					output[m*image_eh*image_ew + (h+pad)*image_ew+pad + w] = 0;
					for (int n = 0; n < in_c; n++){
						for (int kernel_h = 0; kernel_h < kernel; kernel_h++){
							for (int kernel_w = 0; kernel_w < kernel; kernel_w++){
								output[m*image_eh*image_ew + (h+pad)*image_ew+pad + w] +=
										filter[m*in_c*kernel_size+n*kernel *kernel + kernel_h*kernel + kernel_w]
									* input[n*image_eh*image_ew + (h+kernel_h)*image_ew+w+kernel_w];
							}
						}
					}
					output[m*image_eh*image_ew + (h+pad)*image_ew+pad + w] += filter[in_c*output_c*kernel_size+m];
				}
			}
		}
}
int main(){
	srand((int)time(0));
	int in_c=16;
	int output_c=16;
	int input_size=in_c*(image_h+2*pad)*(image_w+2*pad);
	union{Dtype value_i;Dtype value_d;}converter;
	Dtype* input=new Dtype[input_size];
	Dtype * hw_input=new Dtype[input_size];
	for(int i=0;i<in_c;i++){
		for (int r=0;r<image_h;r++){
			for (int c=0;c<image_w;c++){
				converter.value_d=Dtype(rand()%10)/100;
				//converter.value_d=1;
				hw_input[i*(image_h+2*pad)*(image_w+2*pad)+(r+pad)*(image_w+2*pad)+pad+c]=converter.value_i;
				input[i*(image_h+2*pad)*(image_w+2*pad)+(r+pad)*(image_w+2*pad)+pad+c]=converter.value_d;
			}
		}
	}
	int weight_size=in_c*output_c*kernel*kernel;
	Dtype *weight=new Dtype[weight_size+20];
	Dtype *hw_weight=new Dtype[weight_size+20];
	for(int i=0;i<weight_size;i++){
		//converter.value_d=1;
		converter.value_d=Dtype(rand()%10)/100;
		hw_weight[i]=converter.value_i;
		weight[i]=converter.value_d;
	}
	for(int i=0;i<output_c;i++){
			//converter.value_d=1;
			converter.value_d=Dtype(rand()%10)/100;
			hw_weight[in_c*output_c*kernel_size+i]=converter.value_i;
			weight[in_c*output_c*kernel_size+i]=converter.value_d;
		}
	int output_size=output_c*(image_h+2*pad)*(image_w+2*pad);
	Dtype *hw_output=new Dtype[output_size];
	Dtype *sw_output=new Dtype[output_size];
	top(hw_input,hw_weight,hw_output,in_c,output_c,false);
	sw_conv(input,weight,sw_output,in_c,output_c);

	int err=0;
    for(int i=0;i<output_c;i++){
      	for (int r=0;r<image_h;r++){
      		for (int c=0;c<image_w;c++){
      			converter.value_i=hw_output[i*image_eh*image_ew+(r+pad)*image_ew+c+pad];
				//printf("  %f  ",converter.value_d);
//      				if(converter.value_d!=9){
//      				    err+=1;
//      				    printf(" %f ",converter.value_d);
//      				}
				if(sw_output[i*image_eh*image_ew+(r+pad)*image_ew+c+pad]-converter.value_d>0.01){
					err+=1;
					cout<<"sw output: "<<sw_output[i*image_eh*image_ew+(r+pad)*image_ew+c+pad]<<"  hw_output: "<<converter.value_d<<endl;
				}
				//cout<<"sw output: "<<sw_output[i*image_eh*image_ew+(r+pad)*image_ew+c+pad]<<"  hw_output: "<<converter.value_d<<endl;
      		}
      	}
      }
    printf("err is %d ",err);
	return err;
	//return 0;
}
