# Rule for PoseBeliefPrior :out — message from the prior toward the pose variable
#
# Simply returns a PoseBelief constructed from the given mean and covariance.

@rule PoseBeliefPrior(:out, Marginalisation) (q_Tmean::PointMass, q_Σ::PointMass) = begin
    return PoseBelief(mean(q_Tmean), mean(q_Σ))
end
