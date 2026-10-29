#pragma once
#include "ggml.h"

namespace yue2::detail {
// Row norms of W + s*B*A, without forming the dense adapted matrix.
// This uses full normalization gradients, as in our SA3 path. ReLU plus epsilon
// provides a differentiable nonnegative guard using existing GGML primitives.
inline ggml_tensor *dora_norm_from_projection(ggml_context *ctx, ggml_tensor *wta,
                                  ggml_tensor *a, ggml_tensor *b,
                                  ggml_tensor *base_norm_sq, float scale = 1.0F) {
    const auto out = b->ne[1];
    auto *bt = ggml_cont(ctx, ggml_transpose(ctx, b));
    auto *cross = ggml_reshape_1d(ctx, ggml_sum_rows(ctx,
        ggml_cont(ctx, ggml_transpose(ctx, ggml_mul(ctx, wta, bt)))), out);
    auto *ata = ggml_mul_mat(ctx, a, a);
    auto *square = ggml_reshape_1d(ctx,
        ggml_sum_rows(ctx, ggml_mul(ctx, b, ggml_mul_mat(ctx, ata, b))), out);
    auto *nsq = ggml_add(ctx, base_norm_sq,
        ggml_add(ctx, ggml_scale(ctx, cross, 2 * scale),
                      ggml_scale(ctx, square, scale * scale)));
    return ggml_sqrt(ctx, ggml_scale_bias(ctx, ggml_relu(ctx, nsq), 1.0F, 1e-12F));
}
inline ggml_tensor *dora_row_norm(ggml_context *ctx, ggml_tensor *w,
                                  ggml_tensor *a, ggml_tensor *b,
                                  ggml_tensor *base_norm_sq, float scale = 1.0F) {
    return dora_norm_from_projection(ctx, ggml_mul_mat(ctx,w,a), a,b,base_norm_sq,scale);
}
} // namespace yue2::detail
