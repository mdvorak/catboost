#include "rsm_helper.h"

#include <catboost/libs/helpers/exception.h>

namespace NCatboostCuda {
    TRsmFeatureSampler::TRsmFeatureSampler(const TBinarizedFeaturesManager& featuresManager,
                                           double rsm,
                                           TVector<ui32> candidateFeatureIds)
        : FeaturesManager(featuresManager)
        , Rsm(rsm)
        , CandidateFeatureIds(std::move(candidateFeatureIds))
    {
        CB_ENSURE(Rsm > 0 && Rsm <= 1, "rsm should be in (0, 1]");
    }

    bool TRsmFeatureSampler::NextLevel(TRandom& random) {
        const ui32 featureCount = FeaturesManager.GetFeatureCount();
        FeatureMaskCpu.assign(featureCount, 0);

        bool hasSampledFeatures = false;
        for (ui32 featureId : CandidateFeatureIds) {
            const bool isSampled = random.NextUniform() <= Rsm;
            FeatureMaskCpu[featureId] = isSampled;
            hasSampledFeatures |= isSampled;
        }

        if (FeatureMask.GetObjectsSlice().Size() != featureCount) {
            FeatureMask.Reset(NCudaLib::TMirrorMapping(featureCount));
        }
        FeatureMask.Write(FeatureMaskCpu);
        return hasSampledFeatures;
    }
}
