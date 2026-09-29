#include "loss_functions_gpu_helper.h"

namespace {
    struct TFocalSamples {
        TVector<float> Targets;
        TVector<TVector<float>> Cursor{1};
        TVector<float> Weights;
    };

    TFocalSamples GenerateFocalSamples(ui64 seed) {
        constexpr ui32 SIZE = 100;

        TRandom random(seed);
        TFocalSamples samples;
        GenerateSamples(random, SIZE, samples.Targets, samples.Cursor, samples.Weights);

        // Confident predictions of both signs for both classes, including values beyond the
        // probability clamp, where (1 - pt) underflows in float unless computed directly.
        for (float target : {0.0f, 1.0f}) {
            for (float approx : {5.0f, 15.0f, 29.0f, 40.0f}) {
                for (float sign : {-1.0f, 1.0f}) {
                    samples.Targets.push_back(target);
                    samples.Cursor[0].push_back(sign * approx);
                }
            }
        }

        samples.Weights.resize(samples.Targets.size());
        for (float& weight : samples.Weights) {
            weight = 0.5f + random.NextUniform();
        }
        return samples;
    }
}

Y_UNIT_TEST_SUITE(FocalMetricOnGpuTest) {

    Y_UNIT_TEST(FocalTest) {
        auto stopCudaManagerGuard = StartCudaManager();
        for (ui64 seed : {0, 42, 100}) {
            const TFocalSamples samples = GenerateFocalSamples(seed);
            for (double alpha : {0.25, 0.5, 0.75}) {
                for (double gamma : {0.5, 1.0, 2.0, 3.0}) {
                    const auto approximate = [&](const auto& targets, const auto& weights, const auto& cursor, TVec* score, TVec* der, TVec* der2) {
                        ApproximateFocal(targets, weights, cursor, alpha, gamma, score, der, der2);
                    };
                    const auto params = TLossParams::FromVector({{"focal_alpha", ToString(alpha)}, {"focal_gamma", ToString(gamma)}});
                    CompareLossAndDerivatives(
                        CalculateLossAndDerivativesOnCpu(samples.Targets, samples.Cursor, samples.Weights, ELossFunction::Focal, TFocalError(alpha, gamma, /*isExpApprox=*/false), params),
                        CalculateLossAndDerivativesOnGpu(samples.Targets, samples.Cursor, samples.Weights, approximate));
                }
            }
        }
    }
}
