#pragma once

#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/data/binarizations_manager.h>
#include <catboost/cuda/gpu_data/grid_policy.h>
#include <catboost/libs/helpers/cpu_random.h>

#include <util/generic/ptr.h>
#include <util/generic/serialized_enum.h>
#include <util/generic/vector.h>

namespace NCatboostCuda {
    /*
     * Per-level feature subsampling (rsm, aka colsample_bylevel) for GPU split search.
     *
     * As on CPU, at every split selection step each feature is kept with probability rsm and only splits of kept
     * features compete for the best split. Every GPU feature is sampled on its own: a float or one-hot feature, an
     * exclusive features bundle or a CTR. Tree CTRs are not sampled.
     */
    class TRsmFeatureSampler {
    public:
        TRsmFeatureSampler(const TBinarizedFeaturesManager& featuresManager,
                           double rsm,
                           TVector<ui32> candidateFeatureIds);

        // Draws a new sample and returns whether it has a split candidate. As on CPU, it can be empty: then there
        // is no split at this level
        bool NextLevel(TRandom& random);

        const TMirrorBuffer<ui8>* GetFeatureMask() const {
            return &FeatureMask;
        }

    private:
        const TBinarizedFeaturesManager& FeaturesManager;
        const double Rsm;
        const TVector<ui32> CandidateFeatureIds;
        TVector<ui8> FeatureMaskCpu;
        TMirrorBuffer<ui8> FeatureMask;
    };

    template <class TDataSet>
    inline void AddFeaturesForSplitSearch(const TDataSet& dataSet, TVector<ui32>* featureIds) {
        for (auto policy : GetEnumAllValues<EFeaturesGroupingPolicy>()) {
            if (dataSet.HasFeaturesForPolicy(policy)) {
                const auto& grid = dataSet.GetCpuGrid(policy);
                for (ui32 i = 0; i < grid.FeatureIds.size(); ++i) {
                    // a feature without a scored bin never gives a split candidate
                    if (grid.Folds[i] > (grid.SkipFirstBucketWhileScoring[i] ? 1u : 0u)) {
                        featureIds->push_back(grid.FeatureIds[i]);
                    }
                }
            }
        }
    }

    // Returns nullptr when rsm keeps every feature
    template <class TDataSet>
    inline THolder<TRsmFeatureSampler> CreateRsmFeatureSampler(const TBinarizedFeaturesManager& featuresManager,
                                                               double rsm,
                                                               const TDataSet& dataSet) {
        if (rsm >= 1.0) {
            return nullptr;
        }
        TVector<ui32> candidateFeatureIds;
        if (dataSet.HasFeatures()) {
            AddFeaturesForSplitSearch(dataSet.GetFeatures(), &candidateFeatureIds);
        }
        if (dataSet.HasPermutationDependentFeatures()) {
            AddFeaturesForSplitSearch(dataSet.GetPermutationFeatures(), &candidateFeatureIds);
        }
        return MakeHolder<TRsmFeatureSampler>(featuresManager, rsm, std::move(candidateFeatureIds));
    }
}
