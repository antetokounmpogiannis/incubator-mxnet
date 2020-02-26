/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 *  Copyright (c) 2019 by Contributors
 * \file np_insert_op_scalar.cc
 * \brief CPU Implementation of numpy insert operations
 */
#include "./np_insert_op-inl.h"
#include "./np_insert_op_scalar-inl.h"

namespace mxnet {
namespace op {

bool NumpyInsertScalarType(const nnvm::NodeAttrs& attrs,
                           std::vector<int> *in_type,
                           std::vector<int> *out_type) {
  const NumpyInsertParam& param = nnvm::get<NumpyInsertParam>(attrs.parsed);
  int input_count = param.val.has_value() ? 1 : 2;
  int insize = input_count;
  CHECK_EQ(in_type->size(), insize);
  CHECK_EQ(out_type->size(), 1U);
  TYPE_ASSIGN_CHECK(*out_type, 0, (*in_type)[0]);  // output type equals to input arr's
  TYPE_ASSIGN_CHECK(*in_type, 0, (*out_type)[0]);
  return (*in_type)[0] != -1;
}

bool NumpyInsertScalarShape(const nnvm::NodeAttrs& attrs,
                            mxnet::ShapeVector *in_shape,
                            mxnet::ShapeVector *out_shape) {
  using namespace mshadow;
  const NumpyInsertParam& param = nnvm::get<NumpyInsertParam>(attrs.parsed);
  int input_count = param.val.has_value() ? 1 : 2;
  int insize = input_count;
  const int arr_pos = 0;
  const int val_pos = param.val.has_value() ? 0 : 1;
  CHECK_EQ(in_shape->size(), insize);
  mxnet::TShape scale_shape(0, 1);
  mxnet::TShape &arrshape = (*in_shape)[arr_pos];
  mxnet::TShape &valshape = param.val.has_value() ? scale_shape : (*in_shape)[val_pos];

  out_shape->clear();

  int ndim = arrshape.ndim();
  int axis = param.axis.has_value() ? param.axis.value() : 0;
  if (!(param.axis.has_value())) {
    arrshape = Shape1(arrshape.Size());
    ndim = 1;
  } else if (ndim == 0) {
    if (param.val.has_value()) {
      out_shape->push_back(scale_shape);
    } else {
      CHECK_EQ(valshape.ndim(), 0)
        << "'arr' is a 0-d array, 'values' can not assign to it. "
        << "alueError: assignment to 0-d array.";
      out_shape->push_back(valshape);
    }
    return shape_is_known(out_shape[0]);
  } else {
    CHECK(axis >= -1 * arrshape.ndim() && axis < arrshape.ndim())
      << "Axis should be in the range of [-r, r-1] where r is the rank of input tensor";
    axis += (axis < 0) ? arrshape.ndim() : 0;
  }

  mxnet::TShape newshape(arrshape);
  mxnet::TShape val_newshape(arrshape.ndim(), -1);
  int numnew = 0;  // amount of new column insert to 'arr' in 'axis'
  // modify values's ndim to arr's ndim, for broadcast easily later
  // e.g. value shape: (2,) arr shape: (3, 2) => value shape: (1, 2)
  for (int i = valshape.ndim() - 1, j = arrshape.ndim() - 1; i >= 0 || j >= 0; --i, --j) {
    if (i >= 0 && j >= 0) {
      val_newshape[j] = valshape[i];
    } else if (i >= 0) {
      CHECK_EQ(valshape[i], 1) << "index exceed limits.";
    } else {
      val_newshape[j] = 1;
    }
  }
  valshape.assign(val_newshape.begin(), val_newshape.end());

  // because of moveaxis(values, 0, axis)
  numnew = valshape[0];

  newshape[axis] += numnew;
  out_shape->push_back(newshape);
  return shape_is_known(newshape);
}

NNVM_REGISTER_OP(_npi_insert_scalar)
.describe(R"code(Insert values along the given axis before the given indices.)code" ADD_FILELINE)
.set_attr_parser(ParamParser<NumpyInsertParam>)
.set_num_inputs([](const NodeAttrs& attrs) {
    const NumpyInsertParam& params = nnvm::get<NumpyInsertParam>(attrs.parsed);
    int input_count = params.val.has_value() ? 1 : 2;
    return input_count;
})
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    const NumpyInsertParam& params = nnvm::get<NumpyInsertParam>(attrs.parsed);
    if (params.val.has_value()) {
      return std::vector<std::string>{"arr"};
    } else {
      return std::vector<std::string>{"arr", "values"};
    }
})
.set_attr<mxnet::FInferShape>("FInferShape", NumpyInsertScalarShape)
.set_attr<nnvm::FInferType>("FInferType", NumpyInsertScalarType)
.set_attr<mxnet::FCompute>("FCompute<cpu>", NumpyInsertScalarCompute<cpu>)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.add_argument("arr", "NDArray-or-Symbol", "Input ndarray")
.add_argument("values", "NDArray-or-Symbol", "Input ndarray")
.add_arguments(NumpyInsertParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
