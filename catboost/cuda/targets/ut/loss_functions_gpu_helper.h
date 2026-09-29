#pragma once

#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_buffer_helpers/all_reduce.h>
#include <catboost/cuda/cuda_util/fill.h>
#include <catboost/cuda/cuda_util/helpers.h>
#include <catboost/cuda/cuda_util/partitions_reduce.h>
#include <catboost/cuda/targets/kernel.h>
#include <catboost/private/libs/algo_helpers/ders_holder.h>
#include <catboost/private/libs/algo_helpers/error_functions.h>
#include <catboost/libs/helpers/cpu_random.h>
#include <catboost/libs/metrics/metric.h>
#include <catboost/libs/metrics/metric_holder.h>
#include <catboost/libs/metrics/sample.h>

#include <library/cpp/testing/unittest/registar.h>

#include <util/generic/algorithm.h>
#include <util/generic/array_ref.h>
#include <util/generic/ymath.h>
#include <util/system/info.h>

#include <cmath>
#include <functional>

using TVec = TSingleBuffer<float>;
using Derivatives = std::pair<TVector<float>, TVector<float>>;
using CpuResult = std::pair<float, TVector<TDers>>;
using GpuResult = std::pair<float, Derivatives>;

inline void GenerateSamples(TRandom & random,
                            ui64 size,
                            TVector<float>& classes,
                            TVector<TVector<float>>& predictions,
                            TVector<float>& weights,
                            double rate = 0.6) {
    classes.resize(size);
    predictions[0].resize(size);
    weights.resize(size, 1.0f);

    double positiveApproxMean = 1;
    double negativeApproxMean = -1;

    for (ui64 i = 0; i < size; ++i) {
        classes[i] = random.NextUniformL() % 2;

        const double classifierClass = (random.NextUniform() < rate) ? classes[i] : 1.0 - classes[i];

        double mean = classifierClass ? positiveApproxMean : negativeApproxMean;
        predictions[0][i] = random.NextGaussian() + mean;
    }
}

// Fills score with the negated weighted loss sum, der with weighted first derivatives and der2 with
// weighted second derivatives, using the GPU sign conventions.
using TGpuLossApproximator = std::function<void(const TSingleBuffer<const float>& targets,
                                                const TSingleBuffer<const float>& weights,
                                                const TSingleBuffer<const float>& cursor,
                                                TVec* score,
                                                TVec* der,
                                                TVec* der2)>;

inline CpuResult CalculateLossAndDerivativesOnCpu(const TVector<float>& targets,
                                                  const TVector<TVector<float>>& cursor,
                                                  const TVector<float>& weights,
                                                  const ELossFunction& lossFunction,
                                                  const IDerCalcer& error,
                                                  const TLossParams& params) {

    const auto metric = std::move(CreateSingleTargetMetric(lossFunction,
                                                           params,
                                                           /*approxDimension=*/1)[0]);
    NPar::TLocalExecutor executor;

    TVector<TVector<double>> approxes(1, TVector<double>(cursor[0].begin(), cursor[0].end()));

    TMetricHolder score = metric->Eval(approxes, targets, weights, {}, 0, targets.size(), executor);

    const ui32 objectsCount = targets.size();
    TVector<TDers> derivatives(objectsCount);
    error.CalcDersRange(
            /*start=*/0,
                      objectsCount,
            /*calcThirdDer=*/false,
                      approxes[0].data(),
            /*approxDeltas=*/nullptr,
                      targets.data(),
                      weights.data(),
                      derivatives.data()
    );

    return std::make_pair(metric->GetFinalError(score), std::move(derivatives));
}

inline GpuResult CalculateLossAndDerivativesOnGpu(const TVector<float>& targets,
                                                  const TVector<TVector<float>>& cursor,
                                                  const TVector<float>& weights,
                                                  const TGpuLossApproximator& approximate) {
    auto docsMapping = NCudaLib::TSingleMapping(0, targets.size());
    auto upload = [&](const TVector<float>& values) {
        auto tmp = TVec::Create(docsMapping);
        tmp.Write(values);
        return tmp.ConstCopyView();
    };
    auto targetsGpu = upload(targets);
    auto weightsGpu = upload(weights);
    auto cursorGpu = upload(cursor[0]);

    auto tmp = TVec::Create(cursorGpu.GetMapping().RepeatOnAllDevices(1));
    auto der = TVec::CopyMapping(cursorGpu);
    auto der2 = TVec::CopyMapping(cursorGpu);
    approximate(targetsGpu, weightsGpu, cursorGpu, &tmp, &der, &der2);

    TVector<float> derVec;
    TVector<float> der2Vec;
    der.Read(derVec);
    der2.Read(der2Vec);

    const double totalWeight = Accumulate(weights, 0.0);
    const float score = -ReadReduce(tmp)[0] / totalWeight;
    return std::make_pair(score, std::make_pair(derVec, der2Vec));
}

inline void CompareLossAndDerivatives(const CpuResult& cpuResult,
                                      const GpuResult& gpuResult,
                                      float precision = 1e-4f) {
    UNIT_ASSERT(std::isfinite(gpuResult.first));
    UNIT_ASSERT_DOUBLES_EQUAL(cpuResult.first, gpuResult.first, precision);
    for (ui32 index = 0; index < cpuResult.second.size(); ++index) {
        const float gpuDer = gpuResult.second.first[index];
        const float gpuDer2 = gpuResult.second.second[index];
        UNIT_ASSERT(std::isfinite(gpuDer));
        UNIT_ASSERT(std::isfinite(gpuDer2));
        UNIT_ASSERT_DOUBLES_EQUAL(cpuResult.second[index].Der1, gpuDer, precision);
        UNIT_ASSERT_DOUBLES_EQUAL(cpuResult.second[index].Der2, -gpuDer2, precision); // 2nd der signs are opposite
    }
}

inline void TestLossFunctionImpl(ui64 seed,
                                 double param,
                                 TString paramName,
                                 const ELossFunction& lossFunction,
                                 const IDerCalcer& error) {
    constexpr ui32 SIZE = 100;

    TRandom random(seed);

    auto stopCudaManagerGuard = StartCudaManager();
    {
        TVector<float> targets;
        TVector<TVector<float>> cursor(1);
        TVector<float> weights;

        GenerateSamples(random, SIZE, targets, cursor, weights);

        const auto approximate = [&](const auto& targetsGpu, const auto& weightsGpu, const auto& cursorGpu, TVec* score, TVec* der, TVec* der2) {
            ApproximatePointwise(targetsGpu, weightsGpu, cursorGpu, lossFunction, param, score, der, der2);
        };
        CompareLossAndDerivatives(
            CalculateLossAndDerivativesOnCpu(targets, cursor, weights, lossFunction, error, TLossParams::FromVector({{paramName, ToString(param)}})),
            CalculateLossAndDerivativesOnGpu(targets, cursor, weights, approximate));
    }
}
