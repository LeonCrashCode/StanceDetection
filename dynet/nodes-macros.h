#ifndef DYNET_NODE_MACROS_H_
#define DYNET_NODE_MACROS_H_

#include "dynet/dim.h"
#include <iostream>

namespace dynet {

inline bool LooksLikeVector(const Dim& d) {
  if (d.ndims() == 1) return true;
  if (d.ndims() > 1) {
    for (unsigned i = 1; i < d.ndims(); ++i)
      if (d[i] != 1) return false;
  }
  return true;
}

template <class T>
inline std::string print_vec(const std::vector<T> & vec) {
  std::string sep = "[";
  std::ostringstream oss;
  for(auto f : vec) {
    oss << sep << f; sep = ",";
  }
  oss << "]";
  return oss.str();
}

template <class T>
inline std::string print_vecs(const std::vector<std::vector<T> > & vec) {
  std::string sep = "[";
  std::ostringstream oss;
  for(auto & f : vec) {
    oss << sep << print_vec(f); sep = ",";
  }
  oss << "]";
  return oss.str();
}

}

// A macro to dispatch things to the appropriate device
#define DYNET_NODE_DEFINE_DEV_IMPL() \
  std::string as_string(const std::vector<std::string>& arg_names) const override; \
  Dim dim_forward(const std::vector<Dim>& xs) const override; \
  void forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const override; \
  template <class MyDevice> \
  void forward_dev_impl(const MyDevice & dev, const std::vector<const Tensor*>& xs, Tensor& fx) const; \
  void backward_impl(const std::vector<const Tensor*>& xs, \
                const Tensor& fx, \
                const Tensor& dEdf, \
                unsigned i, \
                Tensor& dEdxi) const override; \
  template <class MyDevice> \
  void backward_dev_impl( \
                const MyDevice & dev, \
                const std::vector<const Tensor*>& xs, \
                const Tensor& fx, \
                const Tensor& dEdf, \
                unsigned i, \
                Tensor& dEdxi) const;

// A macro to instantiate templated device functions
// If the implementation is the same for both devices (using Eigen Tensors),
//  then this will instantiate both CPU and GPU implementations, and the
//  code can be the same.
// If the implementation is different for both devices, use #ifdef __CUDACC__
//  within the function, and create alternative code paths for CPU and GPU implementations
#ifdef __CUDACC__
#define DYNET_NODE_INST_DEV_IMPL(MyNode) \
  template void MyNode::forward_dev_impl<Device_GPU>(const Device_GPU & dev, const vector<const Tensor*>& xs, Tensor& fx) const; \
  template void MyNode::backward_dev_impl<Device_GPU>(const Device_GPU & dev, \
                                           const vector<const Tensor*>& xs, \
                                           const Tensor& fx, \
                                           const Tensor& dEdf, \
                                           unsigned i, \
                                           Tensor& dEdxi) const;
#elif HAVE_CUDA
// When we have CUDA, make sure we extern the GPU code to make sure that it is only 
// generated by CUDACC
#define DYNET_NODE_INST_DEV_IMPL(MyNode) \
  extern template void MyNode::forward_dev_impl<Device_GPU>(const Device_GPU & dev, const vector<const Tensor*>& xs, Tensor& fx) const; \
  extern template void MyNode::backward_dev_impl<Device_GPU>(const Device_GPU & dev, \
                                                             const vector<const Tensor*>& xs, \
                                                             const Tensor& fx, \
                                                             const Tensor& dEdf, \
                                                             unsigned i, \
                                                             Tensor& dEdxi) const; \
  template void MyNode::forward_dev_impl<Device_CPU>(const Device_CPU & dev, const vector<const Tensor*>& xs, Tensor& fx) const; \
  template void MyNode::backward_dev_impl<Device_CPU>(const Device_CPU & dev, \
                                           const vector<const Tensor*>& xs, \
                                           const Tensor& fx, \
                                           const Tensor& dEdf, \
                                           unsigned i, \
                                           Tensor& dEdxi) const; \
  void MyNode::forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const { \
    DYNET_ASSERT(fx.device, "Device not allocated for expression"); \
    if(fx.device->type == DeviceType::CPU) { forward_dev_impl<dynet::Device_CPU>(*(dynet::Device_CPU*)fx.device,xs,fx); } \
    else if(fx.device->type == DeviceType::GPU) { forward_dev_impl<dynet::Device_GPU>(*(dynet::Device_GPU*)fx.device,xs,fx); } \
    else { throw std::runtime_error("Invalid device in MyNode::forward_impl"); } \
  } \
  void MyNode::backward_impl(const std::vector<const Tensor*>& xs, \
                const Tensor& fx, \
                const Tensor& dEdf, \
                unsigned i, \
                Tensor& dEdxi) const { \
    DYNET_ASSERT(fx.device, "Device not allocated for expression"); \
    if(fx.device->type == DeviceType::CPU) { backward_dev_impl<dynet::Device_CPU>(*(dynet::Device_CPU*)fx.device,xs,fx,dEdf,i,dEdxi); } \
    else if(fx.device->type == DeviceType::GPU) { backward_dev_impl<dynet::Device_GPU>(*(dynet::Device_GPU*)fx.device,xs,fx,dEdf,i,dEdxi); } \
    else { throw std::runtime_error("Invalid device in MyNode::backward_impl"); } \
  }
#else
#define DYNET_NODE_INST_DEV_IMPL(MyNode) \
  template void MyNode::forward_dev_impl<Device_CPU>(const Device_CPU & dev, const vector<const Tensor*>& xs, Tensor& fx) const; \
  template void MyNode::backward_dev_impl<Device_CPU>(const Device_CPU & dev, \
                                           const vector<const Tensor*>& xs, \
                                           const Tensor& fx, \
                                           const Tensor& dEdf, \
                                           unsigned i, \
                                           Tensor& dEdxi) const; \
  void MyNode::forward_impl(const std::vector<const Tensor*>& xs, Tensor& fx) const { \
    DYNET_ASSERT(fx.device, "Device not allocated for expression"); \
    if(fx.device->type == DeviceType::CPU) { forward_dev_impl<dynet::Device_CPU>(*(dynet::Device_CPU*)fx.device,xs,fx); } \
    else { throw std::runtime_error("Invalid device in MyNode::forward_impl"); } \
  } \
  void MyNode::backward_impl(const std::vector<const Tensor*>& xs, \
                const Tensor& fx, \
                const Tensor& dEdf, \
                unsigned i, \
                Tensor& dEdxi) const { \
    DYNET_ASSERT(fx.device, "Device not allocated for expression"); \
    if(fx.device->type == DeviceType::CPU) { backward_dev_impl<dynet::Device_CPU>(*(dynet::Device_CPU*)fx.device,xs,fx,dEdf,i,dEdxi); } \
    else { throw std::runtime_error("Invalid device in MyNode::backward_impl"); } \
  }
#endif

#endif
