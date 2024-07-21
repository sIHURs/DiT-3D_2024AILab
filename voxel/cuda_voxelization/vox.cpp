#include "vox.hpp"
#include "vox_cuda.cuh"

#include "include/utils.hpp"

/*
  Function: average pool voxelization (forward)
  Args:
    features: features, FloatTensor[b, c, n]
    coords  : coords of each point, IntTensor[b, 3, n]
    resolution : voxel resolution
  Return:
    out : outputs, FloatTensor[b, c, s], s = r ** 3
    ind : voxel index of each point, IntTensor[b, n]
    cnt : #points in each voxel index, IntTensor[b, s]
*/
std::vector<at::Tensor> avg_voxelize_forward(const at::Tensor features,
                                             const at::Tensor coords,
                                             const int resolution) {
  CHECK_CUDA(features);
  CHECK_CUDA(coords);
  CHECK_CONTIGUOUS(features);
  CHECK_CONTIGUOUS(coords);
  CHECK_IS_FLOAT(features);
  CHECK_IS_INT(coords);

  int b = features.size(0);
  int c = features.size(1);
  int n = features.size(2);
  int r = resolution;
  int r2 = r * r;
  int r3 = r2 * r;
  at::Tensor ind = torch::zeros(
      {b, n}, at::device(features.device()).dtype(at::ScalarType::Int));
  at::Tensor out = torch::zeros(
      {b, c, r3}, at::device(features.device()).dtype(at::ScalarType::Float));
  at::Tensor cnt = torch::zeros(
      {b, r3}, at::device(features.device()).dtype(at::ScalarType::Int));
  avg_voxelize(b, c, n, r, r2, r3, coords.data_ptr<int>(),
               features.data_ptr<float>(), ind.data_ptr<int>(),
               cnt.data_ptr<int>(), out.data_ptr<float>());
  return {out, ind, cnt};
}

/*
  Function: average pool voxelization (backward)
  Args:
    grad_y : grad outputs, FloatTensor[b, c, s]
    indices: voxel index of each point, IntTensor[b, n]
    cnt    : #points in each voxel index, IntTensor[b, s]
  Return:
    grad_x : grad inputs, FloatTensor[b, c, n]
*/
at::Tensor avg_voxelize_backward(const at::Tensor grad_y,
                                 const at::Tensor indices,
                                 const at::Tensor cnt) {
  CHECK_CUDA(grad_y);
  CHECK_CUDA(indices);
  CHECK_CUDA(cnt);
  CHECK_CONTIGUOUS(grad_y);
  CHECK_CONTIGUOUS(indices);
  CHECK_CONTIGUOUS(cnt);
  CHECK_IS_FLOAT(grad_y);
  CHECK_IS_INT(indices);
  CHECK_IS_INT(cnt);

  int b = grad_y.size(0);
  int c = grad_y.size(1);
  int s = grad_y.size(2);
  int n = indices.size(1);
  at::Tensor grad_x = torch::zeros(
      {b, c, n}, at::device(grad_y.device()).dtype(at::ScalarType::Float));
  avg_voxelize_grad(b, c, n, s, indices.data_ptr<int>(), cnt.data_ptr<int>(),
                    grad_y.data_ptr<float>(), grad_x.data_ptr<float>());
  return grad_x;
}


// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
// //   m.def("gather_features_forward", &gather_features_forward,
// //         "Gather Centers' Features forward (CUDA)");
// //   m.def("gather_features_backward", &gather_features_backward,
// //         "Gather Centers' Features backward (CUDA)");
// //   m.def("furthest_point_sampling", &furthest_point_sampling_forward,
// //         "Furthest Point Sampling (CUDA)");
// //   m.def("ball_query", &ball_query_forward, "Ball Query (CUDA)");
// //   m.def("grouping_forward", &grouping_forward,
// //         "Grouping Features forward (CUDA)");
// //   m.def("grouping_backward", &grouping_backward,
// //         "Grouping Features backward (CUDA)");
// //   m.def("three_nearest_neighbors_interpolate_forward",
// //         &three_nearest_neighbors_interpolate_forward,
// //         "3 Nearest Neighbors Interpolate forward (CUDA)");
// //   m.def("three_nearest_neighbors_interpolate_backward",
// //         &three_nearest_neighbors_interpolate_backward,
// //         "3 Nearest Neighbors Interpolate backward (CUDA)");

// //   m.def("trilinear_devoxelize_forward", &trilinear_devoxelize_forward,
// //         "Trilinear Devoxelization forward (CUDA)");
// //   m.def("trilinear_devoxelize_backward", &trilinear_devoxelize_backward,
// //         "Trilinear Devoxelization backward (CUDA)");
//   m.def("avg_voxelize_forward", &avg_voxelize_forward,
//         "Voxelization forward with average pooling (CUDA)");
//   m.def("avg_voxelize_backward", &avg_voxelize_backward,
//         "Voxelization backward (CUDA)");
// }