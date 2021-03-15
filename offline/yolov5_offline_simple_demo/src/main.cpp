#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>
#include <vector>
#include "cnrt.h"
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <errno.h>
#include "processor.hpp"
using namespace std;
using namespace cv;

double GetTickCount()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (ts.tv_sec * 1000000 + ts.tv_nsec / 1000)/1000.0;
}

int main(int argc,char *argv[])
{
	if(argc!=8)
	{
		printf("Usage:%s model_path function_name dev_id dev_channel image_path is_rgb label_filename\n",argv[0]);
		printf("\nex.  ./src/yolov5_offline_simple_demo model/yolov5.cambricon subnet0 0 0 data/000000000872.jpg 2 data/label_map_coco.txt\n\n");
		return -1;
	}
	
	unsigned int 			dev_num=0;
	
	CNRT_CHECK(cnrtInit(0));  
	CNRT_CHECK(cnrtGetDeviceCount(&dev_num));
    if (dev_num == 0)
        return -1;
	
	const char				*model_path=argv[1]; 		//模型路径
	const char 				*function_name=argv[2];		//离线模型中的Function Name
	unsigned    			dev_id=atoi(argv[3]);		//使用的MLU 设备ID
	int 					dev_channel=atoi(argv[4]);	//使用MLU设备的通道号 -1 为自动分配
	const char 				*image_path=argv[5];		//测试图片的路径
	int                     is_rgb=atoi(argv[6]);       //数据预处理选项
    string                  label_filename(argv[7]);    //label文件
	int 				    input_width;  				//网络输入的宽度
	int 				    input_height; 				//网络输入的高度
	int         		    batch_size;  				//网络的batch
														
	int64_t 			    *inputSizeS;  				//网络输入数据大小,用于分配内存
	int64_t					*outputSizeS; 				//网络输出数据量大小,用于分配内存
														
	cnrtDataType_t 		    *inputTypeS;				//网络输入的数据类型
	cnrtDataType_t			*outputTypeS;				//网络输出的数据类型
	vector<int> 			output_count;
    cnrtQueue_t 		    queue;						//cnrt queue
														
    cnrtModel_t 		    model;						//离线模型
    cnrtFunction_t 		    function;					//离线模型中的Function
    cnrtDev_t 				dev;						//MLU设备句柄
	cnrtRuntimeContext_t 	ctx;						//推理上下文

	int						*dimValues;					//保存维度shape				
	int 					dimNum;						//保存维度大小
	
    int 					inputNum;					//输入节点个数
    int 					outputNum;					//输出节点个数

    void 					**param;					//保存推理时,mlu上内存地址的指针
    void 					**inputCpuPtrS;				//输入数据的CPU指针
    void 					**outputCpuPtrS;			//输出数据的CPU指针
	void 					**outputCpuNchwPtrS;		//用来保存transpose后的NCHW数据

    void 					**inputMluPtrS;				//输入数据的MLU指针
    void 					**outputMluPtrS;			//输出数据的MLU指针

	
	//获取指定设备的句柄
    CNRT_CHECK(cnrtGetDeviceHandle(&dev, dev_id));
	
	//设置当前使用的设备,作用于线程上下文
    CNRT_CHECK(cnrtSetCurrentDevice(dev));

	//加载离线模型
    CNRT_CHECK(cnrtLoadModel(&model, model_path));
	
	//创建function
    CNRT_CHECK(cnrtCreateFunction(&function));
	
	//从离线模型中提取指定的function,离线模型可以存储多个function
    CNRT_CHECK(cnrtExtractFunction(&function, model, function_name));
	
	//调用cnrtSetCurrentChannel之后 CNRT 仅在指定的通道上分配MLU内存,否则采用交织的方式分配
	if(dev_channel>=0)
		CNRT_CHECK(cnrtSetCurrentChannel((cnrtChannelType_t)dev_channel));

	//创建运行时
	CNRT_CHECK(cnrtCreateRuntimeContext(&ctx,function,NULL));

	//设置运行时使用的设备ID
    CNRT_CHECK(cnrtSetRuntimeContextDeviceId(ctx,dev_id));
	
	//初始化运行时
    CNRT_CHECK(cnrtInitRuntimeContext(ctx,NULL));

	//创建队列
	CNRT_CHECK(cnrtRuntimeContextCreateQueue(ctx,&queue));
    
	//获取模型输入/输出 的数据大小及节点个数
	CNRT_CHECK(cnrtGetInputDataSize(&inputSizeS,&inputNum,function));
	CNRT_CHECK(cnrtGetOutputDataSize(&outputSizeS,&outputNum,function));
	
	//获取模型输入/输出 的数据类型
	CNRT_CHECK(cnrtGetInputDataType(&inputTypeS,&inputNum,function));
	CNRT_CHECK(cnrtGetOutputDataType(&outputTypeS,&outputNum,function));

	//分配 存放CPU端输入/输出地址的 指针数组
    inputCpuPtrS = (void **)malloc(sizeof(void *) * inputNum);
    outputCpuPtrS = (void **)malloc(sizeof(void *) * outputNum);
	outputCpuNchwPtrS = (void **)malloc(sizeof(void *) * outputNum);

	//分配 存放MLU端输入/输出地址的 指针数组
    outputMluPtrS = (void **)malloc(sizeof(void *) * outputNum);
    inputMluPtrS = (void **)malloc(sizeof(void *) * inputNum);

	//为输入节点 分配CPU/MLU内存
	for (int i = 0; i < inputNum; i++)
	{	
		CNRT_CHECK(cnrtMalloc(&inputMluPtrS[i],inputSizeS[i]));	//分配MLU上内存
        inputCpuPtrS[i] = (void *)malloc(inputSizeS[i]); //分配CPU上的内存
		//printf("%d\n", inputSizeS[i]);
		//获取输入的维度信息 NHWC
		CNRT_CHECK(cnrtGetInputDataShape(&dimValues,&dimNum,i,function));						 
		printf("input shape:\n");
		for(int y=0;y<dimNum;y++)
		{
			printf("%d ",dimValues[y]);
		}
		printf("\n");

		input_width=dimValues[2];
		input_height=dimValues[1];
		batch_size=dimValues[0];
        free(dimValues);
    }

	//为输出节点 分配CPU/MLU内存
    for (int i = 0; i < outputNum; i++) {
		CNRT_CHECK(cnrtMalloc(&outputMluPtrS[i],outputSizeS[i])); //分配MLU上内存	
        outputCpuPtrS[i] = (void *)malloc(outputSizeS[i]); //分配CPU上的内存
		//printf("%d\n", outputSizeS[i]);
		//获取输出的维度信息 NHWC
		CNRT_CHECK(cnrtGetOutputDataShape(&dimValues,&dimNum,i,function));		
		int count=1;
		printf("output shape:\n");
		for(int y=0;y<dimNum;y++)
		{
			printf("%d ",dimValues[y]);
			count=count*dimValues[y];
		}
		printf("\n");		
		outputCpuNchwPtrS[i] = (void *)malloc(count*sizeof(float)); //将输出转为float32类型,方便用户后处理
		output_count.push_back(count);
        free(dimValues);
    }

	//配置MLU输入/输出 地址的指针
    param = (void **)malloc(sizeof(void *) * (inputNum + outputNum));
    for (int i = 0; i < inputNum; i++) {
        param[i] = inputMluPtrS[i];
    }
    for (int i = 0; i < outputNum; i++) {
        param[i + inputNum] = outputMluPtrS[i];
    }
		

	
	
	//设置输入/输出的节点 索引
	int input_idx=0;
	int output_idx=0;

    vector<cv::Mat> imgs;
    vector<string> img_names;
	unsigned char *ptr=(unsigned char *)inputCpuPtrS[input_idx];
	for(int i=0;i<batch_size;i++)
	{
        // 选项 2 是yolov5的数据预处理方式
		img_names.push_back(image_path);
		cv::Mat input_image=cv::imread(image_path);
        imgs.push_back(input_image);
		cv::Mat input_image_resized;
		cv::resize(input_image,input_image_resized,cv::Size(input_width,input_height));		
		if(is_rgb==1)
		{
			cv::Mat net_input_data_rgba(input_height,input_width,CV_8UC4,ptr);	
			cv::cvtColor(input_image_resized, net_input_data_rgba, CV_BGR2RGBA);
			ptr+=(input_height*input_width*4);
		} else if(is_rgb==0) {
			cv::Mat net_input_data_rgba(input_height,input_width,CV_8UC4,ptr);	
			cv::cvtColor(input_image_resized, net_input_data_rgba, CV_BGR2BGRA);
			ptr+=(input_height*input_width*4);
		} else if(is_rgb==2) {
          cv::Mat sample_temp;

          float img_w = input_image.cols;
          float img_h = input_image.rows;
          cv::Mat sample_temp_bgr(input_image.cols, input_image.rows, CV_32FC3);
          float img_scale = img_w < img_h ? (input_height / img_h) : (input_width / img_w);
          int new_w = std::floor(img_w * img_scale);
          int new_h = std::floor(img_h * img_scale);
          cv::cvtColor(input_image, sample_temp_bgr, CV_BGR2RGB);
          cv::resize(sample_temp_bgr, sample_temp, cv::Size(new_w, new_h), CV_INTER_LINEAR);
          
          cv::Mat net_input_data_rgba(input_height,input_width,CV_32FC3,ptr);
          sample_temp.copyTo(net_input_data_rgba(
                                             cv::Range((static_cast<float>(input_height) - new_h) / 2,
                                                       (static_cast<float>(input_height) - new_h) / 2 + new_h),
                                             cv::Range((static_cast<float>(input_width) - new_w) / 2,
                                                       (static_cast<float>(input_width) - new_w) / 2 + new_w)));
          net_input_data_rgba /= 255.0;
          ptr+=(input_height*input_width*4);
        }
    }

	auto t0=GetTickCount();
	//拷贝输入数据到MLU内存
	CNRT_CHECK(cnrtMemcpy(inputMluPtrS[input_idx],inputCpuPtrS[input_idx],inputSizeS[input_idx],CNRT_MEM_TRANS_DIR_HOST2DEV));

	//创建事件
	cnrtNotifier_t 			notifier_start;				//用来记录硬件时间
	cnrtNotifier_t			notifier_end;
    CNRT_CHECK(cnrtRuntimeContextCreateNotifier(ctx,&notifier_start));
    CNRT_CHECK(cnrtRuntimeContextCreateNotifier(ctx,&notifier_end));
    CNRT_CHECK(cnrtPlaceNotifier(notifier_start, queue));
	//设置invoke的参数
	unsigned int 			affinity=1<<dev_channel;    //设置通道亲和性,使用指定的MLU cluster做推理
	cnrtInvokeParam_t 		invokeParam;				//invoke参数
	invokeParam.invoke_param_type=CNRT_INVOKE_PARAM_TYPE_0;
	invokeParam.cluster_affinity.affinity=&affinity;
	
    CNRT_CHECK(cnrtInvokeRuntimeContext_V2(ctx,nullptr,param,queue,&invokeParam));
    CNRT_CHECK(cnrtPlaceNotifier(notifier_end, queue));    
    CNRT_CHECK(cnrtSyncQueue(queue));   

	//拷贝MLU输出到CPU内存
	CNRT_CHECK(cnrtMemcpy(outputCpuPtrS[output_idx],outputMluPtrS[output_idx],outputSizeS[output_idx],CNRT_MEM_TRANS_DIR_DEV2HOST));
	auto t1=GetTickCount();
	
	float hwtime;
	CNRT_CHECK(cnrtNotifierDuration(notifier_start, notifier_end, &hwtime));
	
	printf("HardwareTime:%f(ms) E2ETime:%f(ms)\n",hwtime/1000.0,t1-t0);	
	
    int dim_order[4] = {0, 3, 1, 2};
	CNRT_CHECK(cnrtGetOutputDataShape(&dimValues,&dimNum,output_idx,function));
	
	if(dimNum==4)
	{
		//NHWC->NCHW half->float32
		CNRT_CHECK(cnrtTransOrderAndCast(reinterpret_cast<void*>(outputCpuPtrS[output_idx]), outputTypeS[output_idx],
										 reinterpret_cast<void*>(outputCpuNchwPtrS[output_idx]), CNRT_FLOAT32,
										 nullptr, dimNum, dimValues, dim_order));
	}
	else
	{
		//数据类型转换 half->float32
		CNRT_CHECK(cnrtCastDataType(reinterpret_cast<void*>(outputCpuPtrS[output_idx]),
									outputTypeS[output_idx],
									reinterpret_cast<void*>(outputCpuNchwPtrS[output_idx]),
									CNRT_FLOAT32,
									outputSizeS[output_idx]/2,nullptr));		
	}
	 
	//打印输出结果
	float *output_ptr=(float*)outputCpuNchwPtrS[output_idx];
    cout << "boxnum:" << output_ptr[0] << endl;
    vector<vector<vector<float>>> detections = getResults(output_ptr, dimNum, dimValues);
    cout << "=========================" << endl;
    for(auto& d0:detections)
		for(auto &d1:d0){
			for(auto &re:d1)
				cout << re << " ";
			cout << endl;
        }
    cout << "=========================" << endl;
    vector<string> labels;

    readLabels(label_filename, labels);
    writeVisualizeBBox(imgs, detections,labels,
                      img_names, input_height);
    
    free(dimValues);
    CNRT_CHECK(cnrtSetCurrentDevice(dev));
    CNRT_CHECK(cnrtDestroyQueue(queue));
    CNRT_CHECK(cnrtDestroyFunction(function));
    CNRT_CHECK(cnrtUnloadModel(model));

	cnrtDestroyNotifier(&notifier_start);
	cnrtDestroyNotifier(&notifier_end);

    for (int i = 0; i < inputNum; i++) {
        free(inputCpuPtrS[i]);
		cnrtFree(inputMluPtrS[i]);
    }
    for (int i = 0; i < outputNum; i++) {
        free(outputCpuPtrS[i]);
        free(outputCpuNchwPtrS[i]);
		cnrtFree(outputMluPtrS[i]);
    }
	
    free(param);
    free(inputCpuPtrS);
    free(outputCpuPtrS);
	cnrtDestroyRuntimeContext(ctx);
	
	return 0;	
}


