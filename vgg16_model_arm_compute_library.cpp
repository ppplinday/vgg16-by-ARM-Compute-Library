#include "arm_compute/runtime/NEON/NEFunctions.h"

#include "arm_compute/core/Types.h"
#include "test_helpers/Utils.h"

using namespace arm_compute;
using namespace test_helpers;

void main_neon_dnn(int argc, const char **argv)
{
    /*----------------------------------[init_model_vgg16]-----------------------------------*/

    /*----------------------------------BEGIN:[init_Tensor]----------------------------------*/
    //init_input_tensor
    Tensor input;

    //init_conv_1_tensor
    Tensor weights_1_1;
    Tensor biases_1_1;
    Tensor out_1_1;
    Tensor act_1_1;

    Tensor weights_1_2;
    Tensor biases_1_2;
    Tensor out_1_2;
    Tensor act_1_2;

    Tensor pool_1;

    //init_conv_2_tensor
    Tensor weights_2_1;
    Tensor biases_2_1;
    Tensor out_2_1;
    Tensor act_2_1;

    Tensor weights_2_2;
    Tensor biases_2_2;
    Tensor out_2_2;
    Tensor act_2_2;

    Tensor pool_2;

    //init_conv_3_tensor
    Tensor weights_3_1;
    Tensor biases_3_1;
    Tensor out_3_1;
    Tensor act_3_1;

    Tensor weights_3_2;
    Tensor biases_3_2;
    Tensor out_3_2;
    Tensor act_3_2;

    Tensor weights_3_3;
    Tensor biases_3_3;
    Tensor out_3_3;
    Tensor act_3_3;

    Tensor pool_3;

    //init_conv_4_tensor
    Tensor weights_4_1;
    Tensor biases_4_1;
    Tensor out_4_1;
    Tensor act_4_1;

    Tensor weights_4_2;
    Tensor biases_4_2;
    Tensor out_4_2;
    Tensor act_4_2;

    Tensor weights_4_3;
    Tensor biases_4_3;
    Tensor out_4_3;
    Tensor act_4_3;

    Tensor pool_4;

    //init_conv_5_tensor
    Tensor weights_5_1;
    Tensor biases_5_1;
    Tensor out_5_1;
    Tensor act_5_1;

    Tensor weights_5_2;
    Tensor biases_5_2;
    Tensor out_5_2;
    Tensor act_5_2;

    Tensor weights_5_3;
    Tensor biases_5_3;
    Tensor out_5_3;
    Tensor act_5_3;

    Tensor pool_5;

    //init_fc_6
    Tensor weights_6;
    Tensor biases_6;
    Tensor out_6;
    Tensor act_6;

    //init_fc_7
    Tensor weights_7;
    Tensor biases_7;
    Tensor out_7;
    Tensor act_7;

    //init_fc_8
    Tensor weights_8;
    Tensor biases_8;
    Tensor out_8;

    Tensor softmax_tensor;

    //init_tensor
    constexpr unsigned int input_width  = 224;
    constexpr unsigned int input_height = 224;
    constexpr unsigned int input_fm     = 3;

    const TensorShape input_shape(input_width, input_height, input_fm);
    input.allocator() -> init(TensorInfo(input_shape, 1, DataType::F32));

    //init_conv_1_1
    constexpr unsigned int conv_1_1_kernel_x = 3;
    constexpr unsigned int conv_1_1_kernel_y = 3;
    constexpr unsigned int conv_1_1_fm       = 64;

    const TensorShape conv_1_1_weights_shape(conv_1_1_kernel_x, conv_1_1_kernel_y, input_shape.z(), conv_1_1_fm);
    const TensorShape conv_1_1_biases_shape(conv_1_1_weights_shape[3]);
    const TensorShape conv_1_1_out_shape(input_shape.x(), input_shape.y(), conv_1_1_weights_shape[3]);

    weights_1_1.allocator() -> init(TensorInfo(conv_1_1_weights_shape, 1, DataType::F32));
    biases_1_1.allocator() -> init(TensorInfo(conv_1_1_biases_shape, 1, DataType::F32));
    out_1_1.allocator() -> init(TensorInfo(conv_1_1_out_shape, 1, DataType::F32));

    act_1_1.allocator() -> init(TensorInfo(conv_1_1_out_shape, 1, DataType::F32));

    //init_conv_1_2
    constexpr unsigned int conv_1_2_kernel_x = 3;
    constexpr unsigned int conv_1_2_kernel_y = 3;
    constexpr unsigned int conv_1_2_fm       = 64;

    const TensorShape conv_1_2_weights_shape(conv_1_2_kernel_x, conv_1_2_kernel_y, conv_1_1_out_shape.z(), conv_1_2_fm);
    const TensorShape conv_1_2_biases_shape(conv_1_2_weights_shape[3]);
    const TensorShape conv_1_2_out_shape(conv_1_1_out_shape.x(), conv_1_1_out_shape.y(), conv_1_1_weights_shape[3]);

    weights_1_2.allocator() -> init(TensorInfo(conv_1_2_weights_shape, 1, DataType::F32));
    biases_1_2.allocator() -> init(TensorInfo(conv_1_2_biases_shape, 1, DataType::F32));
    out_1_2.allocator() -> init(TensorInfo(conv_1_2_out_shape, 1, DataType::F32));

    act_1_2.allocator() -> init(TensorInfo(conv_1_2_out_shape, 1, DataType::F32));

    TensorShape conv_1_pool = conv_1_2_out_shape;
    conv_1_pool.set(0, conv_1_pool.x() / 2);
    conv_1_pool.set(1, conv_1_pool.y() / 2);
    pool_1.allocator() -> init(TensorInfo(conv_1_pool, 1, DataType::F32));

    //init_conv_2_1
    constexpr unsigned int conv_2_1_kernel_x = 3;
    constexpr unsigned int conv_2_1_kernel_y = 3;
    constexpr unsigned int conv_2_1_fm       = 128;

    const TensorShape conv_2_1_weights_shape(conv_2_1_kernel_x, conv_2_1_kernel_y, conv_1_pool.z(), conv_2_1_fm);
    const TensorShape conv_2_1_biases_shape(conv_2_1_weights_shape[3]);
    const TensorShape conv_2_1_out_shape(conv_1_pool.x(), conv_1_pool.y(), conv_2_1_weights_shape[3]);

    weights_2_1.allocator() -> init(TensorInfo(conv_2_1_weights_shape, 1, DataType::F32));
    biases_2_1.allocator() -> init(TensorInfo(conv_2_1_biases_shape, 1, DataType::F32));
    out_2_1.allocator() -> init(TensorInfo(conv_2_1_out_shape, 1, DataType::F32));

    act_2_1.allocator() -> init(TensorInfo(conv_2_1_out_shape, 1, DataType::F32));

    //init_conv_2_2
    constexpr unsigned int conv_2_2_kernel_x = 3;
    constexpr unsigned int conv_2_2_kernel_y = 3;
    constexpr unsigned int conv_2_2_fm       = 128;

    const TensorShape conv_2_2_weights_shape(conv_2_2_kernel_x, conv_2_2_kernel_y, conv_2_1_out_shape.z(), conv_2_2_fm);
    const TensorShape conv_2_2_biases_shape(conv_2_2_weights_shape[3]);
    const TensorShape conv_2_2_out_shape(conv_2_1_out_shape.x(), conv_2_1_out_shape.y(), conv_2_2_weights_shape[3]);

    weights_2_2.allocator() -> init(TensorInfo(conv_2_2_weights_shape, 1, DataType::F32));
    biases_2_2.allocator() -> init(TensorInfo(conv_2_2_biases_shape, 1, DataType::F32));
    out_2_2.allocator() -> init(TensorInfo(conv_2_2_out_shape, 1, DataType::F32));

    act_2_2.allocator() -> init(TensorInfo(conv_2_2_out_shape, 1, DataType::F32));

    TensorShape conv_2_pool = conv_2_2_out_shape;
    conv_2_pool.set(0, conv_2_pool.x() / 2);
    conv_2_pool.set(1, conv_2_pool.y() / 2);
    pool_2.allocator() -> init(TensorInfo(conv_2_pool, 1, DataType::F32));

    //init_conv_3_1
    constexpr unsigned int conv_3_1_kernel_x = 3;
    constexpr unsigned int conv_3_1_kernel_y = 3;
    constexpr unsigned int conv_3_1_fm       = 256;

    const TensorShape conv_3_1_weights_shape(conv_3_1_kernel_x, conv_3_1_kernel_y, conv_2_pool.z(), conv_3_1_fm);
    const TensorShape conv_3_1_biases_shape(conv_3_1_weights_shape[3]);
    const TensorShape conv_3_1_out_shape(conv_2_pool.x(), conv_2_pool.y(), conv_3_1_weights_shape[3]);

    weights_3_1.allocator() -> init(TensorInfo(conv_3_1_weights_shape, 1, DataType::F32));
    biases_3_1.allocator() -> init(TensorInfo(conv_3_1_biases_shape, 1, DataType::F32));
    out_3_1.allocator() -> init(TensorInfo(conv_3_1_out_shape, 1, DataType::F32));

    act_3_1.allocator() -> init(TensorInfo(conv_3_1_out_shape, 1, DataType::F32));

    //init_conv_3_2
    constexpr unsigned int conv_3_2_kernel_x = 3;
    constexpr unsigned int conv_3_2_kernel_y = 3;
    constexpr unsigned int conv_3_2_fm       = 256;

    const TensorShape conv_3_2_weights_shape(conv_3_2_kernel_x, conv_3_2_kernel_y, conv_3_1_out_shape.z(), conv_3_2_fm);
    const TensorShape conv_3_2_biases_shape(conv_3_2_weights_shape[3]);
    const TensorShape conv_3_2_out_shape(conv_3_1_out_shape.x(), conv_3_1_out_shape.y(), conv_3_2_weights_shape[3]);

    weights_3_2.allocator() -> init(TensorInfo(conv_3_2_weights_shape, 1, DataType::F32));
    biases_3_2.allocator() -> init(TensorInfo(conv_3_2_biases_shape, 1, DataType::F32));
    out_3_2.allocator() -> init(TensorInfo(conv_3_2_out_shape, 1, DataType::F32));

    act_3_2.allocator() -> init(TensorInfo(conv_3_2_out_shape, 1, DataType::F32));

    //init_conv_3_3
    constexpr unsigned int conv_3_3_kernel_x = 3;
    constexpr unsigned int conv_3_3_kernel_y = 3;
    constexpr unsigned int conv_3_3_fm       = 256;

    const TensorShape conv_3_3_weights_shape(conv_3_3_kernel_x, conv_3_3_kernel_y, conv_3_2_out_shape.z(), conv_3_3_fm);
    const TensorShape conv_3_3_biases_shape(conv_3_3_weights_shape[3]);
    const TensorShape conv_3_3_out_shape(conv_3_2_out_shape.x(), conv_3_2_out_shape.y(), conv_3_3_weights_shape[3]);

    weights_3_3.allocator() -> init(TensorInfo(conv_3_3_weights_shape, 1, DataType::F32));
    biases_3_3.allocator() -> init(TensorInfo(conv_3_3_biases_shape, 1, DataType::F32));
    out_3_3.allocator() -> init(TensorInfo(conv_3_3_out_shape, 1, DataType::F32));

    act_3_3.allocator() -> init(TensorInfo(conv_3_3_out_shape, 1, DataType::F32));

    TensorShape conv_3_pool = conv_3_3_out_shape;
    conv_3_pool.set(0, conv_3_pool.x() / 2);
    conv_3_pool.set(1, conv_3_pool.y() / 2);
    pool_3.allocator() -> init(TensorInfo(conv_3_pool, 1, DataType::F32));

    //init_conv_4_1
    constexpr unsigned int conv_4_1_kernel_x = 3;
    constexpr unsigned int conv_4_1_kernel_y = 3;
    constexpr unsigned int conv_4_1_fm       = 512;

    const TensorShape conv_4_1_weights_shape(conv_4_1_kernel_x, conv_4_1_kernel_y, conv_3_pool.z(), conv_4_1_fm);
    const TensorShape conv_4_1_biases_shape(conv_4_1_weights_shape[3]);
    const TensorShape conv_4_1_out_shape(conv_3_pool.x(), conv_3_pool.y(), conv_4_1_weights_shape[3]);

    weights_4_1.allocator() -> init(TensorInfo(conv_4_1_weights_shape, 1, DataType::F32));
    biases_4_1.allocator() -> init(TensorInfo(conv_4_1_biases_shape, 1, DataType::F32));
    out_4_1.allocator() -> init(TensorInfo(conv_4_1_out_shape, 1, DataType::F32));

    act_4_1.allocator() -> init(TensorInfo(conv_4_1_out_shape, 1, DataType::F32));

    //init_conv_4_2
    constexpr unsigned int conv_4_2_kernel_x = 3;
    constexpr unsigned int conv_4_2_kernel_y = 3;
    constexpr unsigned int conv_4_2_fm       = 512;

    const TensorShape conv_4_2_weights_shape(conv_4_2_kernel_x, conv_4_2_kernel_y, conv_4_1_out_shape.z(), conv_4_2_fm);
    const TensorShape conv_4_2_biases_shape(conv_4_2_weights_shape[3]);
    const TensorShape conv_4_2_out_shape(conv_4_1_out_shape.x(), conv_4_1_out_shape.y(), conv_4_2_weights_shape[3]);

    weights_4_2.allocator() -> init(TensorInfo(conv_4_2_weights_shape, 1, DataType::F32));
    biases_4_2.allocator() -> init(TensorInfo(conv_4_2_biases_shape, 1, DataType::F32));
    out_4_2.allocator() -> init(TensorInfo(conv_4_2_out_shape, 1, DataType::F32));

    act_4_2.allocator() -> init(TensorInfo(conv_4_2_out_shape, 1, DataType::F32));

    //init_conv_4_3
    constexpr unsigned int conv_4_3_kernel_x = 3;
    constexpr unsigned int conv_4_3_kernel_y = 3;
    constexpr unsigned int conv_4_3_fm       = 512;

    const TensorShape conv_4_3_weights_shape(conv_4_3_kernel_x, conv_4_3_kernel_y, conv_4_2_out_shape.z(), conv_4_3_fm);
    const TensorShape conv_4_3_biases_shape(conv_4_3_weights_shape[3]);
    const TensorShape conv_4_3_out_shape(conv_4_2_out_shape.x(), conv_4_2_out_shape.y(), conv_4_3_weights_shape[3]);

    weights_4_3.allocator() -> init(TensorInfo(conv_4_3_weights_shape, 1, DataType::F32));
    biases_4_3.allocator() -> init(TensorInfo(conv_4_3_biases_shape, 1, DataType::F32));
    out_4_3.allocator() -> init(TensorInfo(conv_4_3_out_shape, 1, DataType::F32));

    act_4_3.allocator() -> init(TensorInfo(conv_4_3_out_shape, 1, DataType::F32));

    TensorShape conv_4_pool = conv_4_3_out_shape;
    conv_4_pool.set(0, conv_4_pool.x() / 2);
    conv_4_pool.set(1, conv_4_pool.y() / 2);
    pool_4.allocator() -> init(TensorInfo(conv_4_pool, 1, DataType::F32));

    //init_conv_5_1
    constexpr unsigned int conv_5_1_kernel_x = 3;
    constexpr unsigned int conv_5_1_kernel_y = 3;
    constexpr unsigned int conv_5_1_fm       = 512;

    const TensorShape conv_5_1_weights_shape(conv_5_1_kernel_x, conv_5_1_kernel_y, conv_4_pool.z(), conv_5_1_fm);
    const TensorShape conv_5_1_biases_shape(conv_5_1_weights_shape[3]);
    const TensorShape conv_5_1_out_shape(conv_4_pool.x(), conv_4_pool.y(), conv_5_1_weights_shape[3]);

    weights_5_1.allocator() -> init(TensorInfo(conv_5_1_weights_shape, 1, DataType::F32));
    biases_5_1.allocator() -> init(TensorInfo(conv_5_1_biases_shape, 1, DataType::F32));
    out_5_1.allocator() -> init(TensorInfo(conv_5_1_out_shape, 1, DataType::F32));

    act_5_1.allocator() -> init(TensorInfo(conv_5_1_out_shape, 1, DataType::F32));

    //init_conv_5_2
    constexpr unsigned int conv_5_2_kernel_x = 3;
    constexpr unsigned int conv_5_2_kernel_y = 3;
    constexpr unsigned int conv_5_2_fm       = 512;

    const TensorShape conv_5_2_weights_shape(conv_5_2_kernel_x, conv_5_2_kernel_y, conv_5_1_out_shape.z(), conv_5_2_fm);
    const TensorShape conv_5_2_biases_shape(conv_5_2_weights_shape[3]);
    const TensorShape conv_5_2_out_shape(conv_5_1_out_shape.x(), conv_5_1_out_shape.y(), conv_5_2_weights_shape[3]);

    weights_5_2.allocator() -> init(TensorInfo(conv_5_2_weights_shape, 1, DataType::F32));
    biases_5_2.allocator() -> init(TensorInfo(conv_5_2_biases_shape, 1, DataType::F32));
    out_5_2.allocator() -> init(TensorInfo(conv_5_2_out_shape, 1, DataType::F32));

    act_5_2.allocator() -> init(TensorInfo(conv_5_2_out_shape, 1, DataType::F32));

    //init_conv_5_3
    constexpr unsigned int conv_5_3_kernel_x = 3;
    constexpr unsigned int conv_5_3_kernel_y = 3;
    constexpr unsigned int conv_5_3_fm       = 512;

    const TensorShape conv_5_3_weights_shape(conv_5_3_kernel_x, conv_5_3_kernel_y, conv_5_2_out_shape.z(), conv_5_3_fm);
    const TensorShape conv_5_3_biases_shape(conv_5_3_weights_shape[3]);
    const TensorShape conv_5_3_out_shape(conv_5_2_out_shape.x(), conv_5_2_out_shape.y(), conv_5_3_weights_shape[3]);

    weights_5_3.allocator() -> init(TensorInfo(conv_5_3_weights_shape, 1, DataType::F32));
    biases_5_3.allocator() -> init(TensorInfo(conv_5_3_biases_shape, 1, DataType::F32));
    out_5_3.allocator() -> init(TensorInfo(conv_5_3_out_shape, 1, DataType::F32));

    act_5_3.allocator() -> init(TensorInfo(conv_5_3_out_shape, 1, DataType::F32));

    TensorShape conv_5_pool = conv_5_3_out_shape;
    conv_5_pool.set(0, conv_5_pool.x() / 2);
    conv_5_pool.set(1, conv_5_pool.y() / 2);
    pool_5.allocator() -> init(TensorInfo(conv_5_pool, 1, DataType::F32));

    //init_fc_6
    constexpr unsigned int fc_6_numoflabel = 4096;

    const TensorShape fc_6_weights_shape(conv_5_pool.x() * conv_5_pool.y() * conv_5_pool.z(), fc_6_numoflabel);
    const TensorShape fc_6_biases_shape(fc_6_numoflabel);
    const TensorShape fc_6_out_shape(fc_6_numoflabel);

    weights_6.allocator() -> init(TensorInfo(fc_6_weights_shape, 1, DataType::F32));
    biases_6.allocator() -> init(TensorInfo(fc_6_biases_shape, 1, DataType::F32));
    out_6.allocator() -> init(TensorInfo(fc_6_out_shape, 1, DataType::F32));

    act_6.allocator() -> init(TensorInfo(fc_6_out_shape, 1, DataType::F32));

    //init_fc_7
    constexpr unsigned int fc_7_numoflabel = 4096;

    const TensorShape fc_7_weights_shape(fc_6_out_shape.x(), fc_7_numoflabel);
    const TensorShape fc_7_biases_shape(fc_7_numoflabel);
    const TensorShape fc_7_out_shape(fc_7_numoflabel);

    weights_7.allocator() -> init(TensorInfo(fc_7_weights_shape, 1, DataType::F32));
    biases_7.allocator() -> init(TensorInfo(fc_7_biases_shape, 1, DataType::F32));
    out_7.allocator() -> init(TensorInfo(fc_7_out_shape, 1, DataType::F32));

    act_7.allocator() -> init(TensorInfo(fc_7_out_shape, 1, DataType::F32));

    //init_fc_8
    constexpr unsigned int fc_8_numoflabel = 1000;

    const TensorShape fc_8_weights_shape(fc_7_out_shape.x(), fc_8_numoflabel);
    const TensorShape fc_8_biases_shape(fc_8_numoflabel);
    const TensorShape fc_8_out_shape(fc_8_numoflabel);

    weights_8.allocator() -> init(TensorInfo(fc_8_weights_shape, 1, DataType::F32));
    biases_8.allocator() -> init(TensorInfo(fc_8_biases_shape, 1, DataType::F32));
    out_8.allocator() -> init(TensorInfo(fc_8_out_shape, 1, DataType::F32));

    const TensorShape softmax_shape(fc_8_out_shape.x());
    softmax_tensor.allocator() -> init(TensorInfo(softmax_shape, 1, DataType::F32));

    /*----------------------------------END:[init_Tensor]----------------------------------*/


    /*-----------------------------BEGIN:[Configure Functions]-----------------------------*/
    //init_layer
    NEConvolutionLayer    conv_1_1;
    NEConvolutionLayer    conv_1_2;
    NEConvolutionLayer    conv_2_1;
    NEConvolutionLayer    conv_2_2;
    NEConvolutionLayer    conv_3_1;
    NEConvolutionLayer    conv_3_2;
    NEConvolutionLayer    conv_3_3;
    NEConvolutionLayer    conv_4_1;
    NEConvolutionLayer    conv_4_2;
    NEConvolutionLayer    conv_4_3;
    NEConvolutionLayer    conv_5_1;
    NEConvolutionLayer    conv_5_2;
    NEConvolutionLayer    conv_5_3;
    NEActivationLayer     Nact_1_1;
    NEActivationLayer     Nact_1_2;
    NEActivationLayer     Nact_2_1;
    NEActivationLayer     Nact_2_2;
    NEActivationLayer     Nact_3_1;
    NEActivationLayer     Nact_3_2;
    NEActivationLayer     Nact_3_3;
    NEActivationLayer     Nact_4_1;
    NEActivationLayer     Nact_4_2;
    NEActivationLayer     Nact_4_3;
    NEActivationLayer     Nact_5_1;
    NEActivationLayer     Nact_5_2;
    NEActivationLayer     Nact_5_3;
    NEActivationLayer     Nact_6;
    NEActivationLayer     Nact_7;
    NEPoolingLayer        Npool_1;
    NEPoolingLayer        Npool_2;
    NEPoolingLayer        Npool_3;
    NEPoolingLayer        Npool_4;
    NEPoolingLayer        Npool_5;
    NEFullyConnectedLayer fc_6;
    NEFullyConnectedLayer fc_7;
    NEFullyConnectedLayer fc_8;
    NESoftmaxLayer        softmax;

    //conv_1
    //in: 224 * 224 * 3, kernel: 3 * 3 * 3 * 64, out: 224 * 224 * 64
    conv_1_1.configure(&input, &weights_1_1, &biases_1_1, &out_1_1, PadStrideInfo(1, 1, 1, 1));

    //in: 224 * 224 * 64, out: 224 * 224 * 64
    Nact_1_1.configure(&out_1_1, &act_1_1, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

    //in: 224 × 224 × 64, kernel: 3 * 3 * 64 * 64, out: 224 * 224 * 64
    conv_1_2.configure(&act_1_1, &weights_1_2, &biases_1_2, &out_1_2, PadStrideInfo(1, 1, 1, 1));

    //in: 224 * 224 * 64, out: 224 * 224 * 64
    Nact_1_2.configure(&out_1_2, &act_1_2, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

    //in: 224 * 224 * 64, out: 112 * 112 * 64
    Npool_1.configure(&act_1_2, &pool_1, PoolingLayerInfo(PoolingType::MAX, 2, PadStrideInfo(2, 2)));

    //conv_2
    //in: 112 * 112 * 64, kernel: 3 * 3 * 64 * 128. out: 112 * 112 * 128
    conv_2_1.configure(&pool_1, &weights_2_1, &biases_2_1, &out_2_1, PadStrideInfo(1, 1, 1, 1));

    //in: 112 * 112 * 128, out: 112 * 112 * 128
    Nact_2_1.configure(&out_2_1, &act_2_1, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

    //in: 112 * 112 * 128, kernel: 3 * 3 * 128 * 128, out: 112 * 112 * 128
    conv_2_2.configure(&act_2_1, &weights_2_2, &biases_2_2, &out_2_2, PadStrideInfo(1, 1, 1, 1));

    //in: 112 * 112 * 128, out: 112 * 112 * 128
    Nact_2_2.configure(&out_2_2, &act_2_2, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

    //in: 112 * 112 * 128, out: 56 * 56 * 128
    Npool_2.configure(&act_2_2, &pool_2, PoolingLayerInfo(PoolingType::MAX, 2, PadStrideInfo(2, 2)));

    //conv_3
    //in: 56 * 56 * 128, kernel: 3 * 3 * 128 * 256, out: 56 * 56 * 256
    conv_3_1.configure(&pool_2, &weights_3_1, &biases_3_1, &out_3_1, PadStrideInfo(1, 1, 1, 1));

    //in: 56 * 56 * 256, out: 56 * 56 * 256
    Nact_3_1.configure(&out_3_1, &act_3_1, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

    //in: 56 * 56 * 256, kernel: 3 * 3 * 256 * 256, out: 56 * 56 * 256
    conv_3_2.configure(&act_3_1, &weights_3_2, &biases_3_2, &out_3_2, PadStrideInfo(1, 1, 1, 1));

    //in: 56 * 56 * 256, out: 56 * 56 * 256
    Nact_3_2.configure(&out_3_2, &act_3_2, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

    //in: 56 * 56 * 256, kernel: 3 * 3 * 256 * 256, out: 56 * 56 * 256
    conv_3_3.configure(&act_3_2, &weights_3_3, &biases_3_3, &out_3_3, PadStrideInfo(1, 1, 1, 1));

    //in: 56 * 56 * 256, out: 56 * 56 * 256
    Nact_3_3.configure(&out_3_3, &act_3_3, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

    //in: 56 * 56 * 256, out: 28 * 28 * 256
    Npool_3.configure(&act_3_3, &pool_3, PoolingLayerInfo(PoolingType::MAX, 2, PadStrideInfo(2, 2)));

    //conv_4
    //in: 28 * 28 * 256, kernel: 3 * 3 * 256 * 512, out: 28 * 28 * 512
    conv_4_1.configure(&pool_3, &weights_4_1, &biases_4_1, &out_4_1, PadStrideInfo(1, 1, 1, 1));

    //in: 28 * 28 * 512, out: 28 * 28 * 512
    Nact_4_1.configure(&out_4_1, &act_4_1, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

    //in: 28 * 28 * 512, kernel: 3 * 3 * 512 * 512, out: 28 * 28 * 512
    conv_4_2.configure(&act_4_1, &weights_4_2, &biases_4_2, &out_4_2, PadStrideInfo(1, 1, 1, 1));

    //in: 28 * 28 * 512, out: 28 * 28 * 512
    Nact_4_2.configure(&out_4_2, &act_4_2, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

    //in: 28 * 28 * 512, kernel: 3 * 3 * 512 * 512, out: 28 * 28 * 512
    conv_4_3.configure(&act_4_2, &weights_4_3, &biases_4_3, &out_4_3, PadStrideInfo(1, 1, 1, 1));

    //in: 28 * 28 * 512, out: 28 * 28 * 512
    Nact_4_3.configure(&out_4_3, &act_4_3, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

    //in: 28 * 28 * 512, out: 14 * 14 * 512
    Npool_4.configure(&act_4_3, &pool_4, PoolingLayerInfo(PoolingType::MAX, 2, PadStrideInfo(2, 2)));

    //conv_5
    //in: 14 * 14 * 512, kernel: 3 * 3 * 512 * 512, out: 14 * 14 * 512
    conv_5_1.configure(&pool_4, &weights_5_1, &biases_5_1, &out_5_1, PadStrideInfo(1, 1, 1, 1));

    //in: 14 * 14 * 512, out: 14 * 14 * 512
    Nact_5_1.configure(&out_5_1, &act_5_1, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

    //in: 14 * 14 * 512, kernel: 3 * 3 * 512 * 512, out: 14 * 14 * 512
    conv_5_2.configure(&act_5_1, &weights_5_2, &biases_5_2, &out_5_2, PadStrideInfo(1, 1, 1, 1));

    //in: 14 * 14 * 512, out: 14 * 14 * 512
    Nact_5_2.configure(&out_5_2, &act_5_2, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

    //in: 14 * 14 * 512 kernel: 3 * 3 * 512 * 512, out: 14 * 14 * 512
    conv_5_3.configure(&act_5_2, &weights_5_3, &biases_5_3, &out_5_3, PadStrideInfo(1, 1, 1, 1));

    //in: 14 * 14 * 512, out: 14 * 14 * 512
    Nact_5_3.configure(&out_5_3, &act_5_3, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

    //in: 14 * 14 * 512, out: 7 * 7 * 512
    Npool_5.configure(&act_5_3, &pool_5, PoolingLayerInfo(PoolingType::MAX, 2, PadStrideInfo(2, 2)));

    //fc_6
    //in: 7 * 7 * 512, out: 4096
    fc_6.configure(&pool_5, &weights_6, &biases_6, &out_6);

    //in: 4096, out: 4096
    Nact_6.configure(&out_6, &act_6, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

    //fc_7
    //in: 4096, out: 4096
    fc_7.configure(&act_6, &weights_7, &biases_7, &out_7);

    //in:4096, out: 4096
    Nact_7.configure(&out_7, &act_7, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

    //fc_8
    //in: 4096, out: 1000
    fc_8.configure(&act_7, &weights_8, &biases_8, &out_8);

    //softmax layer: 1000
    softmax.configure(&out_8, &softmax_tensor);

    /*------------------------------END:[Configure Functions]------------------------------*/

    /*------------------------------BEGIN:[Allocate tensors]-------------------------------*/

    //input
    input.allocator() -> allocate();

    //conv_1
    weights_1_1.allocator() -> allocate();
    biases_1_1.allocator() -> allocate();
    out_1_1.allocator() -> allocate();
    act_1_1.allocator() -> allocate();

    weights_1_2.allocator() -> allocate();
    biases_1_2.allocator() -> allocate();
    out_1_2.allocator() -> allocate();
    act_1_2.allocator() -> allocate();

    pool_1.allocator() -> allocate();

    //conv_2
    weights_2_1.allocator() -> allocate();
    biases_2_1.allocator() -> allocate();
    out_2_1.allocator() -> allocate();
    act_2_1.allocator() -> allocate();

    weights_2_2.allocator() -> allocate();
    biases_2_2.allocator() -> allocate();
    out_2_2.allocator() -> allocate();
    act_2_2.allocator() -> allocate();

    pool_2.allocator() -> allocate();

    //conv_3
    weights_3_1.allocator() -> allocate();
    biases_3_1.allocator() -> allocate();
    out_3_1.allocator() -> allocate();
    act_3_1.allocator() -> allocate();

    weights_3_2.allocator() -> allocate();
    biases_3_2.allocator() -> allocate();
    out_3_2.allocator() -> allocate();
    act_3_2.allocator() -> allocate();

    weights_3_3.allocator() -> allocate();
    biases_3_3.allocator() -> allocate();
    out_3_3.allocator() -> allocate();
    act_3_3.allocator() -> allocate();

    pool_3.allocator() -> allocate();

    //conv_4
    weights_4_1.allocator() -> allocate();
    biases_4_1.allocator() -> allocate();
    out_4_1.allocator() -> allocate();
    act_4_1.allocator() -> allocate();

    weights_4_2.allocator() -> allocate();
    biases_4_2.allocator() -> allocate();
    out_4_2.allocator() -> allocate();
    act_4_2.allocator() -> allocate();

    weights_4_3.allocator() -> allocate();
    biases_4_3.allocator() -> allocate();
    out_4_3.allocator() -> allocate();
    act_4_3.allocator() -> allocate();

    pool_4.allocator() -> allocate();

    //conv_5
    weights_5_1.allocator() -> allocate();
    biases_5_1.allocator() -> allocate();
    out_5_1.allocator() -> allocate();
    act_5_1.allocator() -> allocate();

    weights_5_2.allocator() -> allocate();
    biases_5_2.allocator() -> allocate();
    out_5_2.allocator() -> allocate();
    act_5_2.allocator() -> allocate();

    weights_5_3.allocator() -> allocate();
    biases_5_3.allocator() -> allocate();
    out_5_3.allocator() -> allocate();
    act_5_3.allocator() -> allocate();

    pool_5.allocator() -> allocate();

    //fc_6
    weights_6.allocator() -> allocate();
    biases_6.allocator() -> allocate();
    out_6.allocator() -> allocate();
    act_6.allocator() -> allocate();

    //fc_7
    weights_7.allocator() -> allocate();
    biases_7.allocator() -> allocate();
    out_7.allocator() -> allocate();
    act_7.allocator() -> allocate();

    //fc_8
    weights_8.allocator() -> allocate();
    biases_8.allocator() -> allocate();
    out_8.allocator() -> allocate();
    softmax_tensor.allocator() -> allocate();

    /*------------------------------END:[Allocate tensors]-------------------------------*/

    /*--------------------------BEGIN:[Execute the functions]----------------------------*/

    //conv_1
    conv_1_1.run();
    Nact_1_1.run();
    conv_1_2.run();
    Nact_1_2.run();
    Npool_1.run();

    //conv_2
    conv_2_1.run();
    Nact_2_1.run();
    conv_2_2.run();
    Nact_2_2.run();
    Npool_2.run();

    //conv_3
    conv_3_1.run();
    Nact_3_1.run();
    conv_3_2.run();
    Nact_3_2.run();
    conv_3_3.run();
    Nact_3_3.run();
    Npool_3.run();

    //conv_4
    conv_4_1.run();
    Nact_4_1.run();
    conv_4_2.run();
    Nact_4_2.run();
    conv_4_3.run();
    Nact_4_3.run();
    Npool_4.run();

    //conv_5
    conv_5_1.run();
    Nact_5_1.run();
    conv_5_2.run();
    Nact_5_2.run();
    conv_5_3.run();
    Nact_5_3.run();
    Npool_5.run();

    //fc_6
    fc_6.run();
    Nact_6.run();

    //fc_7
    fc_7.run();
    Nact_7.run();

    //fc_8
    fc_8.run();
    softmax.run();

    /*---------------------------END:[Execute the functions]-----------------------------*/

    //test
    std::cout << "fine! Jason!" << std::endl;
}


/** Main program for convolution test
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments ( [optional] Path to PPM image to process )
 */
int main(int argc, const char **argv)
{
    return test_helpers::run_example(argc, argv, main_neon_dnn);
}