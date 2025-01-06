//===- TppPasses.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef EINSUM_PASSES_H
#define EINSUM_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
class ModuleOp;

namespace affine {
class AffineDialect;
} // namespace affine

namespace arith {
class ArithDialect;
} // namespace arith

namespace bufferization {
class BufferizationDialect;
} // namespace bufferization

namespace func {
class FuncOp;
class FuncDialect;
} // namespace func

namespace gpu {
class GPUModuleOp;
class GPUDialect;
} // namespace gpu

namespace linalg {
class LinalgDialect;
} // namespace linalg

namespace LLVM {
class LLVMDialect;
} // namespace LLVM

namespace math {
class MathDialect;
} // namespace math

namespace memref {
class MemRefDialect;
} // namespace memref


namespace scf {
class SCFDialect;
} // namespace scf

namespace spirv {
class SPIRVDialect;
}
namespace tensor {
class TensorDialect;
} // namespace tensor

namespace vector {
class VectorDialect;
} // namespace vector

namespace xegpu {
class XeGPUDialect;
} // namespace xegpu

} // namespace mlir

namespace mlir {
namespace einsum {
// TODO: This should be per-pass so that pass can live
// in their namespace (xsmm, check...). All the passes
// are now in tpp.
#define GEN_PASS_DECL
#include "Einsum/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "Einsum/Passes.h.inc"
} // namespace einsum
} // namespace mlir

#endif // EINSUM_PASSES_H
