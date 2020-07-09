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
void top(  float *input,  float *weight,   float *output, int in_c, int output_c, bool relu);
void process(  float *input,   float *weight,   float *output, int in_c,int output_c,bool relu);
void calculate_output(  float *input,   float *weight,   float *output, Dtype output_buff[OUTPUT_CHANNEL_TILE][OUTPUT_ROW_TILE][image_w], int in_c,int output_c,int to,int tr);
void post_process(  float *output, Dtype output_buff[OUTPUT_CHANNEL_TILE][OUTPUT_ROW_TILE][image_w], int in_c,int output_c,int to,int tr,bool relu);
void set_zero(Dtype output_buff[OUTPUT_CHANNEL_TILE][OUTPUT_ROW_TILE][image_w]);
void load_input(Dtype input_buff[INPUT_CHANNEL_TILE][INPUT_ROW_TILE][image_ew],   float *input, int ti, int tr, int in_c);
void load_weight(Dtype weight_buff[OUTPUT_CHANNEL_TILE][INPUT_CHANNEL_TILE][kernel][kernel],   float *weight, int to, int ti,int in_c,int output_c);
void conv(Dtype *input, Dtype *weight, Dtype *output, Dtype input_buff[INPUT_CHANNEL_TILE][INPUT_ROW_TILE][image_ew], Dtype weight_buff[OUTPUT_CHANNEL_TILE][INPUT_CHANNEL_TILE][kernel][kernel], Dtype output_buff[OUTPUT_CHANNEL_TILE][OUTPUT_ROW_TILE][image_w],
		int in_c,int output_c,int tr,int ti);
int min(int a,int b);
void conv(  float *input,   float *weight,   float *output, Dtype input_buff[INPUT_CHANNEL_TILE][INPUT_ROW_TILE][image_ew], Dtype weight_buff[OUTPUT_CHANNEL_TILE][INPUT_CHANNEL_TILE][kernel][kernel], Dtype output_buff[OUTPUT_CHANNEL_TILE][OUTPUT_ROW_TILE][image_w],
		int in_c,int output_c,int tr,int ti);
void store_output(Dtype output_buff[OUTPUT_CHANNEL_TILE][OUTPUT_ROW_TILE][image_w],   float *output, int to,int tr, int output_c);



Dtype input_buff0[INPUT_CHANNEL_TILE][INPUT_ROW_TILE][image_ew];
Dtype input_buff1[INPUT_CHANNEL_TILE][INPUT_ROW_TILE][image_ew];
Dtype weight_buff0[OUTPUT_CHANNEL_TILE][INPUT_CHANNEL_TILE][kernel][kernel];
Dtype weight_buff1[OUTPUT_CHANNEL_TILE][INPUT_CHANNEL_TILE][kernel][kernel];
Dtype output_buff0[OUTPUT_CHANNEL_TILE][OUTPUT_ROW_TILE][image_w];
Dtype output_buff1[OUTPUT_CHANNEL_TILE][OUTPUT_ROW_TILE][image_w];
Dtype bias_buff[20];




void top(  float *input,  float *weight,   float *output, int in_c, int output_c, bool relu)
{
#pragma HLS INTERFACE s_axilite port=output_c
#pragma HLS INTERFACE s_axilite port=in_c
#pragma HLS INTERFACE s_axilite port=return
#pragma HLS INTERFACE s_axilite port=relu
	int input_size=INPUT_SIZE;
	int output_size=OUTPUT_SIZE;
	int weight_size=WEIGHT_SIZE+20;//bias_size:20
#pragma HLS INTERFACE m_axi depth=output_size port=output offset=slave bundle=OUTPUT_BUS
#pragma HLS INTERFACE m_axi depth=weight_size port=weight offset=slave bundle=WEIGHT_BUS
#pragma HLS INTERFACE m_axi depth=input_size port=input offset=slave bundle=INPUT_BUS

#pragma HLS ARRAY_PARTITION variable=input_buff0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=input_buff1 complete dim=1

#pragma HLS ARRAY_PARTITION variable=weight_buff0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=weight_buff0 complete dim=2
#pragma HLS ARRAY_PARTITION variable=weight_buff1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=weight_buff1 complete dim=2

#pragma HLS ARRAY_PARTITION variable=bias_buff complete dim=1

#pragma HLS ARRAY_PARTITION variable=output_buff0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=output_buff1 complete dim=1
    process(input, weight, output, in_c, output_c, relu);
}
void process(  float *input,   float *weight,   float *output, int in_c,int output_c,bool relu)
{
    int flag = 0;
    int to, tr;
    memcpy(bias_buff, (uint32*)(weight+in_c*output_c*kernel_size), sizeof(Dtype)*output_c);
    output_channel: for (to=0;to<output_c;to+=OUTPUT_CHANNEL_TILE)
    {
        output_row: for (tr=0;tr<image_h;tr+=OUTPUT_ROW_TILE)
        {
            if (flag == 0)
            {
                calculate_output(input,weight,output,output_buff0,in_c,output_c,to,tr);
                post_process(output,output_buff1,in_c,output_c,to,tr-OUTPUT_ROW_TILE,relu);
            }else
            {
                calculate_output(input,weight,output,output_buff1,in_c,output_c,to,tr);
                post_process(output,output_buff0,in_c,output_c,to,tr-OUTPUT_ROW_TILE,relu);
            }
            flag = 1-flag;
        }
        if (flag == 0)
        {
            post_process(output,output_buff1,in_c,output_c,to,tr-OUTPUT_ROW_TILE,relu);
        }else
        {
            post_process(output,output_buff0,in_c,output_c,to,tr-OUTPUT_ROW_TILE,relu);
        }

    }
}
void calculate_output(  float *input,   float *weight,   float *output, Dtype output_buff[OUTPUT_CHANNEL_TILE][OUTPUT_ROW_TILE][image_w], int in_c,int output_c,int to,int tr)
{
    set_zero(output_buff);
    int flag = 0;
    int ti;
    in_channel: for (ti=0;ti<in_c;ti+=INPUT_CHANNEL_TILE)
    {
        if (flag==0)
        {
            load_input(input_buff0, input, ti, tr, in_c);
            load_weight(weight_buff0, weight, to, ti, in_c, output_c);
            conv(input, weight, output, input_buff1, weight_buff1, output_buff, in_c, output_c, tr, ti-INPUT_CHANNEL_TILE);
        }else
        {
            load_input(input_buff1, input, ti, tr, in_c);
            load_weight(weight_buff1, weight, to, ti, in_c, output_c);
            conv(input, weight, output, input_buff0, weight_buff0, output_buff, in_c, output_c, tr, ti-INPUT_CHANNEL_TILE);
        }
        flag = 1-flag;
    }
    if (flag == 0)
    {
        conv(input, weight, output, input_buff1, weight_buff1, output_buff, in_c, output_c, tr, ti-INPUT_CHANNEL_TILE);
    }else
    {
        conv(input, weight, output, input_buff0, weight_buff0, output_buff, in_c, output_c, tr, ti-INPUT_CHANNEL_TILE);
    }

}
void set_zero(Dtype output_buff[OUTPUT_CHANNEL_TILE][OUTPUT_ROW_TILE][image_w])
{
	SET_R: for (int r = 0; r < OUTPUT_ROW_TILE; r++)
    {
		SET_W: for (int c = 0; c < image_w; c++)
        {
#pragma HLS PIPELINE
            SET_C: for (int i=0; i < OUTPUT_CHANNEL_TILE; i++)
            {
                output_buff[i][r][c] = 0;
            }
        }
    }
}
void load_input(Dtype input_buff[INPUT_CHANNEL_TILE][INPUT_ROW_TILE][image_ew],   float *input, int ti, int tr, int in_c)
{
    load_input_ROW: for (int j=0,trr=tr; j<INPUT_ROW_TILE; trr++,j++)
    {
        load_input_CHANNEL: for (int i=0, tii=ti;i<INPUT_CHANNEL_TILE; tii++,i++)
        {
            if (tii<in_c&&trr<image_eh)
            {
                memcpy(input_buff[i][j], (uint32*)(input+tii*image_eh*image_ew+trr*image_ew),sizeof(uint32)*image_ew);
            }else
            {
                load_input_COL: for (int c=0; c<image_ew; c++)
                {
#pragma HLS  PIPELINE
                    input_buff[i][j][c] = 0;
                }
            }

        }
    }
}
void load_weight(Dtype weight_buff[OUTPUT_CHANNEL_TILE][INPUT_CHANNEL_TILE][kernel][kernel],   float *weight, int to, int ti,int in_c,int output_c)
{
    load_weight_OUTC: for (int i=0, too=to;i<OUTPUT_CHANNEL_TILE;too++,i++)
    {
        load_weight_INC: for (int j=0, tii=ti; j<INPUT_CHANNEL_TILE;tii++,j++)
        {
            if (too<output_c&&tii<in_c)
            {
                memcpy(weight_buff[i][j], (uint32*)(weight+too*in_c*kernel_size+tii*kernel_size),sizeof(uint32)*kernel_size);
            }else
            {
                load_kernel_h: for (int k=0; k<kernel_size; k++){
#pragma HLS PIPELINE
                    weight_buff[i][j][k/kernel][k%kernel]=0;
                }
            }

        }
    }
}
int min(int a,int b){
	return a>b?b:a;
}
void conv(  float *input,   float *weight,   float *output, Dtype input_buff[INPUT_CHANNEL_TILE][INPUT_ROW_TILE][image_ew], Dtype weight_buff[OUTPUT_CHANNEL_TILE][INPUT_CHANNEL_TILE][kernel][kernel], Dtype output_buff[OUTPUT_CHANNEL_TILE][OUTPUT_ROW_TILE][image_w],
		int in_c,int output_c,int tr,int ti)
{
    if (ti<0)
    {
        return;
    }
    kh: for (int kh=0; kh<kernel;kh++){
        kw: for (int kw=0; kw<kernel;kw++){
            tcc: for (int tcc=0; tcc<image_w;tcc++){
                trr: for (int trr=0; trr<OUTPUT_ROW_TILE;trr++){
#pragma HLS LOOP_TRIPCOUNT min=1 max=14
#pragma HLS PIPELINE
                    too: for (int too=0;too<OUTPUT_CHANNEL_TILE;too++){
                        tii: for (int tii=0;tii<INPUT_CHANNEL_TILE;tii++){
                            output_buff[too][trr][tcc] += input_buff[tii][trr*stride+kh][tcc*stride+kw]*weight_buff[too][tii][kh][kw];
                        }
                    }
                }
            }
        }
    }
}
void post_process(  float *output, Dtype output_buff[OUTPUT_CHANNEL_TILE][OUTPUT_ROW_TILE][image_w], int in_c,int output_c,int to,int tr,bool relu)
{
    if (tr<0)
    {
        return;
    }
    Dtype temp;
    output_R: for (int j=0; j<OUTPUT_ROW_TILE;j++){
        output_W: for (int k=0;k<image_w;k++){
#pragma HLS PIPELINE
            output_C: for (int i=0;i<OUTPUT_CHANNEL_TILE;i++){
                temp = output_buff[i][j][k] + bias_buff[to+i];
                if (relu){
                    temp = temp>0?temp:0;
                }
                output_buff[i][j][k] = temp;
            }
        }
    }
    store_output(output_buff,output,to,tr,output_c);
}
void store_output(Dtype output_buff[OUTPUT_CHANNEL_TILE][OUTPUT_ROW_TILE][image_w],   float *output, int to,int tr, int output_c)
{
    store_output_C: for (int i=0, too=to; i<OUTPUT_CHANNEL_TILE; too++,i++){
        store_output_R:for (int j=0, trr=tr; j<OUTPUT_ROW_TILE;trr++,j++){
            if (trr<image_h&&too<output_c){
                memcpy((uint32*)(output+too*image_eh*image_ew+(trr+pad)*image_ew+pad), output_buff[i][j], sizeof(uint32)*image_w);
            }
        }
    }
}

