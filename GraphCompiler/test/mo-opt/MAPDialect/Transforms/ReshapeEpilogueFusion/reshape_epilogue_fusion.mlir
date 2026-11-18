// RUN: mo-opt --mlir-print-local-scope %s --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" --mo-to-map --map-fuse-kernels | FileCheck %s

// A reshape between a (non-accumulating) fused-output kernel and an add fuses
// into the producer's epilogue: ElementwiseFuser folds the reshape into the
// add, then ReshapeEpilogueFuser relocates it onto the producer's epilogue as
// a swapped `map.index.reshape` on the STORE index. Only the store DESTINATION
// is retyped to [64]; the epilogue RESULT (what the kernel binds) keeps the
// producer's [8, 8] shape. The bias load and store both use the reshaped
// index. (An accumulating producer -- matmul/conv/concat -- is declined, since
// it would touch the null-pointer fused-output slice; see ReshapeEpilogueFuser.)

// CHECK-LABEL: mo.graph @custom_reshape_add
// CHECK:       %[[OUT:.*]] = map.iter.opaque
// CHECK:       outputs (%{{.*}} : !mo.tensor<[64], f32>) {
// CHECK:       map.iter.epilogue (%[[VAL:.*]] : f32, %[[IDX:.*]] : !map.index<2>)
// CHECK:       %[[SIDX:.*]] = map.index.reshape<#map.index.reshape.spec<[64] -> [8, 8]>>
// CHECK:       %[[BV:.*]] = map.load %{{.*}}[%[[SIDX]]] : !mo.tensor<[64], f32>
// CHECK:       %[[ADD:.*]] = map.call.elementwise @{{.*}}.Add(%{{.*}}, %[[VAL]], %[[BV]])
// CHECK:       map.store %{{.*}}[%[[SIDX]]] = %[[ADD]] : !mo.tensor<[64], f32>
// CHECK:       map.call.opaque @{{.*}}MapLowerOneFusedOutput
// CHECK:       fusionInfo = #mogg.fusion<#mogg.fusion_epilogue<[#mogg.fusion_leaf<"custom__map_lower_one_fused_output"
// CHECK-NOT:   map.iter.elementwise
// CHECK:       mo.output %[[OUT]] : !mo.tensor<[64], f32>
mo.graph @custom_reshape_add(
  %x: !mo.tensor<[8, 8], f32>,
  %z: !mo.tensor<[64], f32>
) -> !mo.tensor<[64], f32> {
  %c = mo.custom [symbol = "map_lower_one_fused_output", device = #M.device_ref<"cpu", 0>](%x) : (!mo.tensor<[8, 8], f32>) -> !mo.tensor<[8, 8], f32>
  %r = mo.static.reshape(%c) : !mo.tensor<[8, 8], f32> -> !mo.tensor<[64], f32>
  %out = mo.add(%r, %z) : !mo.tensor<[64], f32>
  mo.output %out : !mo.tensor<[64], f32>
}

// A pure reshape (no downstream elementwise op to unlock) is left alone -- a
// no-op relabel in MGP -- so it stays its own view kernel.

// CHECK-LABEL: mo.graph @custom_reshape_only
// CHECK:       map.iter.opaque
// CHECK:       map.iter.elementwise
// CHECK:       map.index.reshape<#map.index.reshape.spec<[8, 8] -> [64]>>
mo.graph @custom_reshape_only(
  %x: !mo.tensor<[8, 8], f32>
) -> !mo.tensor<[64], f32> {
  %c = mo.custom [symbol = "map_lower_one_fused_output", device = #M.device_ref<"cpu", 0>](%x) : (!mo.tensor<[8, 8], f32>) -> !mo.tensor<[8, 8], f32>
  %r = mo.static.reshape(%c) : !mo.tensor<[8, 8], f32> -> !mo.tensor<[64], f32>
  mo.output %r : !mo.tensor<[64], f32>
}

// The kernel calls its epilogue on runs of its own rows, here 25 long. Once
// the reshape is on the store index, a run of 25 spans five consumer rows of
// 5, and the body reads each lane after the first as the next element of the
// consumer row. A transposed read takes those lanes from the wrong elements,
// so the mul is not fused and runs as its own kernel at the consumer's index.

// CHECK-LABEL: mo.graph @custom_reshape_split_rows_transposed_mul
// CHECK:       map.iter.opaque
// CHECK:       map.iter.epilogue
// CHECK-NOT:   map.index.reshape
// CHECK:       map.call.opaque @{{.*}}MapLowerOneFusedOutput
// CHECK:       map.iter.elementwise<[8, 5, 5]>
// CHECK-DAG:   map.index.reshape<#map.index.reshape.spec<[8, 25] -> [8, 5, 5]>>
// CHECK-DAG:   map.index.transpose
// CHECK:       map.call.elementwise @{{.*}}.Mul
mo.graph @custom_reshape_split_rows_transposed_mul(
  %x: !mo.tensor<[8, 25], f32>,
  %z: !mo.tensor<[8, 5, 5], f32>
) -> !mo.tensor<[8, 5, 5], f32> {
  %c = mo.custom [symbol = "map_lower_one_fused_output", device = #M.device_ref<"cpu", 0>](%x) : (!mo.tensor<[8, 25], f32>) -> !mo.tensor<[8, 25], f32>
  %r = mo.static.reshape(%c) : !mo.tensor<[8, 25], f32> -> !mo.tensor<[8, 5, 5], f32>
  %perm = mo.constant <#M.dense_array<0, 2, 1> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %t = mo.transpose(%z, %perm) : (!mo.tensor<[8, 5, 5], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[8, 5, 5], f32>
  %out = mo.mul(%t, %r) : !mo.tensor<[8, 5, 5], f32>
  mo.output %out : !mo.tensor<[8, 5, 5], f32>
}

// The same reshape still fuses when the body reads only at its own index: a
// run that crosses into the next consumer row reads on into the next row of
// the dense operand, which is what it needs.

// CHECK-LABEL: mo.graph @custom_reshape_split_rows_add
// CHECK:       map.iter.epilogue
// CHECK:       %[[SIDX:.*]] = map.index.reshape<#map.index.reshape.spec<[8, 5, 5] -> [8, 25]>>
// CHECK:       map.load %{{.*}}[%[[SIDX]]] : !mo.tensor<[8, 5, 5], f32>
// CHECK:       map.store %{{.*}}[%[[SIDX]]]
// CHECK:       map.call.opaque @{{.*}}MapLowerOneFusedOutput
// CHECK-NOT:   map.iter.elementwise
// CHECK:       mo.output
mo.graph @custom_reshape_split_rows_add(
  %x: !mo.tensor<[8, 25], f32>,
  %z: !mo.tensor<[8, 5, 5], f32>
) -> !mo.tensor<[8, 5, 5], f32> {
  %c = mo.custom [symbol = "map_lower_one_fused_output", device = #M.device_ref<"cpu", 0>](%x) : (!mo.tensor<[8, 25], f32>) -> !mo.tensor<[8, 25], f32>
  %r = mo.static.reshape(%c) : !mo.tensor<[8, 25], f32> -> !mo.tensor<[8, 5, 5], f32>
  %out = mo.add(%r, %z) : !mo.tensor<[8, 5, 5], f32>
  mo.output %out : !mo.tensor<[8, 5, 5], f32>
}

// A transposed read still fuses when the kernel's rows nest in the
// consumer's: every run of 8 lies inside one consumer row of 32.

// CHECK-LABEL: mo.graph @custom_reshape_nested_rows_transposed_mul
// CHECK:       map.iter.epilogue
// CHECK:       %[[SIDX:.*]] = map.index.reshape<#map.index.reshape.spec<[2, 32] -> [8, 8]>>
// CHECK:       map.index.transpose{{.*}}(%[[SIDX]],
// CHECK:       map.call.elementwise @{{.*}}.Mul
// CHECK:       map.call.opaque @{{.*}}MapLowerOneFusedOutput
// CHECK-NOT:   map.iter.elementwise
// CHECK:       mo.output
mo.graph @custom_reshape_nested_rows_transposed_mul(
  %x: !mo.tensor<[8, 8], f32>,
  %z: !mo.tensor<[32, 2], f32>
) -> !mo.tensor<[2, 32], f32> {
  %c = mo.custom [symbol = "map_lower_one_fused_output", device = #M.device_ref<"cpu", 0>](%x) : (!mo.tensor<[8, 8], f32>) -> !mo.tensor<[8, 8], f32>
  %r = mo.static.reshape(%c) : !mo.tensor<[8, 8], f32> -> !mo.tensor<[2, 32], f32>
  %perm = mo.constant <#M.dense_array<1, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %t = mo.transpose(%z, %perm) : (!mo.tensor<[32, 2], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 32], f32>
  %out = mo.mul(%t, %r) : !mo.tensor<[2, 32], f32>
  mo.output %out : !mo.tensor<[2, 32], f32>
}
